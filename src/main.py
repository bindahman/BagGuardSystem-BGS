# src/main.py
from ultralytics import YOLO
import cv2
import time
import os
import argparse
from collections import Counter, defaultdict

# COCO class IDs for YOLOv8 pretrained models
# 0=person, 24=backpack, 26=handbag, 28=suitcase
TARGET_CLASS_IDS = {0, 24, 26, 28}
BAG_CLASS_IDS = {24, 26, 28}

# -----------------------------
# Phase B4: Visual styling
# -----------------------------
COLOR_PERSON = (255, 200, 0)    # cyan-ish (BGR)
COLOR_BAG    = (255, 0, 255)    # magenta (BGR)
COLOR_TEXT   = (255, 255, 255)  # white
COLOR_BG     = (0, 0, 0)        # black
COLOR_COUNT  = (0, 255, 0)      # green
COLOR_OWNER  = (0, 180, 255)    # orange-ish (BGR)

# -----------------------------
# Helpers
# -----------------------------
def now_sec() -> float:
    return time.time()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def bbox_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(1.0, float((x2 - x1) * (y2 - y1)))

def euclid(p1, p2) -> float:
    dx = float(p1[0] - p2[0])
    dy = float(p1[1] - p2[1])
    return (dx * dx + dy * dy) ** 0.5

def frame_diag(frame):
    h, w = frame.shape[:2]
    return (w * w + h * h) ** 0.5

def normalized_distance(p1, p2, frame):
    """Distance divided by frame diagonal. (roughly 0..1+)"""
    d = euclid(p1, p2)
    diag = max(1.0, frame_diag(frame))
    return d / diag

# -----------------------------
# Simple Tracker (Centroid + TTL)
# -----------------------------
class SimpleTracker:
    def __init__(self, match_dist_px=200, max_missed_sec=3.0, use_area_penalty=False):
        self.match_dist_px = float(match_dist_px)
        self.max_missed_sec = float(max_missed_sec)
        self.use_area_penalty = bool(use_area_penalty)

        self.next_id = 1
        self.tracks = {}  # id -> dict

    def _cleanup_stale(self, now_t):
        stale = []
        for tid, tr in self.tracks.items():
            if (now_t - tr["last_seen"]) > self.max_missed_sec:
                stale.append(tid)
        for tid in stale:
            del self.tracks[tid]

    def update(self, detections, now_t):
        """
        detections: list of dicts:
          {
            "cls_id": int,
            "label": str,
            "conf": float,
            "bbox": (x1,y1,x2,y2),
            "centroid": (cx,cy)
          }
        returns: list of tuples (track_id, det_dict)
        """
        self._cleanup_stale(now_t)

        # sort by confidence (high -> low) to reduce ID swaps
        detections = sorted(detections, key=lambda d: d["conf"], reverse=True)

        used_track_ids = set()
        assigned = []

        for det in detections:
            best_id = None
            best_score = float("inf")

            det_c = det["centroid"]
            det_a = bbox_area(det["bbox"])

            for tid, tr in self.tracks.items():
                if tid in used_track_ids:
                    continue

                d = euclid(det_c, tr["bbox_centroid"])

                if d > self.match_dist_px:
                    continue

                if self.use_area_penalty:
                    tr_a = tr.get("bbox_area", det_a)
                    ratio = det_a / max(tr_a, 1e-6)
                    ratio = max(ratio, 1.0 / max(ratio, 1e-6))  # >= 1
                    score = d * ratio
                else:
                    score = d

                if score < best_score:
                    best_score = score
                    best_id = tid

            if best_id is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "bbox_centroid": det_c,
                    "bbox": det["bbox"],
                    "bbox_area": det_a,
                    "last_seen": now_t,
                }
                used_track_ids.add(tid)
                assigned.append((tid, det))
            else:
                tr = self.tracks[best_id]
                tr["bbox_centroid"] = det_c
                tr["bbox"] = det["bbox"]
                tr["bbox_area"] = det_a
                tr["last_seen"] = now_t

                used_track_ids.add(best_id)
                assigned.append((best_id, det))

        return assigned

# -----------------------------
# Phase C3: Ownership persistence (centroid->centroid, normalized distance)
# -----------------------------
class OwnershipManager:
    """
    Keeps stable bag->owner assignments over time using evidence + hysteresis.

    Evidence: accumulate score when person is near bag (based on normalized distance).
    Persistence: locks + switching rules + occlusion grace.
    """
    def __init__(
        self,
        near_norm=0.12,              # "near" threshold in normalized units
        min_score=1.0,               # score needed to claim ownership
        decay_per_sec=0.60,          # evidence decay (0.0..1.0): lower = faster decay
        lock_sec=1.0,                # lock owner for a short time after assignment
        switch_ratio=1.35,           # challenger must beat current by this ratio
        switch_min_sec=0.8,          # challenger dominance duration required
        occlusion_grace_sec=2.0,     # keep owner briefly if person disappears
        margin=0.25                  # best must be >= second*(1+margin)
    ):
        self.near_norm = float(near_norm)
        self.min_score = float(min_score)
        self.decay_per_sec = float(decay_per_sec)
        self.lock_sec = float(lock_sec)
        self.switch_ratio = float(switch_ratio)
        self.switch_min_sec = float(switch_min_sec)
        self.occlusion_grace_sec = float(occlusion_grace_sec)
        self.margin = float(margin)

        # bag_id -> state
        self.state = {}

    def _get_state(self, bag_id):
        if bag_id not in self.state:
            self.state[bag_id] = {
                "owner_id": None,
                "scores": defaultdict(float),   # person_id -> evidence score
                "lock_until": 0.0,
                "owner_seen_until": 0.0,
                "switch_start": None,          # (challenger_id, start_t) or None
                "last_best_dn": None
            }
        return self.state[bag_id]

    def update(self, bag_id, bag_centroid, people_centroids, frame, t, dt):
        """
        people_centroids: dict {person_id: (cx,cy)}
        returns: (owner_id, debug_best_dn)
        """
        st = self._get_state(bag_id)

        # Decay all evidence scores with time
        if dt > 0:
            decay = self.decay_per_sec ** dt
            for pid in list(st["scores"].keys()):
                st["scores"][pid] *= decay
                if st["scores"][pid] < 1e-4:
                    del st["scores"][pid]

        # If we have people, accumulate evidence for those near
        best_pid = None
        best_dn = None

        for pid, pc in people_centroids.items():
            dn = normalized_distance(bag_centroid, pc, frame)

            # closeness: 1 when dn=0, 0 when dn>=near_norm
            closeness = 1.0 - clamp(dn / self.near_norm, 0.0, 1.0)

            if closeness > 0.0:
                st["scores"][pid] += closeness * max(dt, 1/30)  # dt fallback

            if best_dn is None or dn < best_dn:
                best_dn = dn
                best_pid = pid

        st["last_best_dn"] = best_dn

        # If current owner exists, maintain "seen" grace when owner visible
        owner = st["owner_id"]
        if owner is not None and owner in people_centroids:
            st["owner_seen_until"] = t + self.occlusion_grace_sec

        # If owner vanished and grace expired -> drop owner
        if owner is not None and owner not in people_centroids and t > st["owner_seen_until"]:
            st["owner_id"] = None
            st["lock_until"] = 0.0
            st["switch_start"] = None
            owner = None

        # Choose top candidates by score
        if st["scores"]:
            ranked = sorted(st["scores"].items(), key=lambda kv: kv[1], reverse=True)
            top_id, top_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        else:
            top_id, top_score, second_score = None, 0.0, 0.0

        # If no owner, try to assign
        if owner is None:
            if top_id is not None:
                strong_enough = top_score >= self.min_score
                separated = top_score >= second_score * (1.0 + self.margin)
                if strong_enough and separated:
                    st["owner_id"] = top_id
                    st["lock_until"] = t + self.lock_sec
                    st["owner_seen_until"] = t + self.occlusion_grace_sec
                    st["switch_start"] = None
            return st["owner_id"], st["last_best_dn"]

        # If owner exists, only allow switch after lock
        if t < st["lock_until"]:
            return st["owner_id"], st["last_best_dn"]

        # Switch logic: challenger must dominate
        owner_score = st["scores"].get(owner, 0.0001)
        if top_id is not None and top_id != owner:
            if top_score >= owner_score * self.switch_ratio:
                # start or continue dominance timer
                if st["switch_start"] is None or st["switch_start"][0] != top_id:
                    st["switch_start"] = (top_id, t)
                else:
                    _, start_t = st["switch_start"]
                    if (t - start_t) >= self.switch_min_sec:
                        st["owner_id"] = top_id
                        st["lock_until"] = t + self.lock_sec
                        st["owner_seen_until"] = t + self.occlusion_grace_sec
                        st["switch_start"] = None
            else:
                st["switch_start"] = None
        else:
            st["switch_start"] = None

        return st["owner_id"], st["last_best_dn"]

# -----------------------------
# Drawing helpers (Phase B4 + C3 labels)
# -----------------------------
def draw_label_with_bg(frame, x, y, text, box_color=COLOR_BG, text_color=COLOR_TEXT):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x1 = max(0, x)
    y2 = max(0, y)
    y1 = max(0, y2 - th - 10)
    x2 = x1 + tw + 8

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, -1)
    cv2.putText(frame, text, (x1 + 4, y2 - 4), font, scale, text_color, thickness)

def draw_assigned(frame, assigned, color, person=False, ownership=None, owner_debug=None):
    """
    ownership: dict bag_id -> owner_id (only used for bags)
    owner_debug: dict bag_id -> best_dn
    """
    for tid, det in assigned:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        label = det["label"]

        thickness = 3 if person else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if person:
            text = f"{label} #{tid} {conf:.2f}"
            draw_label_with_bg(frame, x1, max(25, y1), text, box_color=COLOR_BG, text_color=COLOR_TEXT)
        else:
            owner_txt = ""
            if ownership is not None:
                oid = ownership.get(tid, None)
                if oid is not None:
                    owner_txt = f" | owner: P#{oid}"
            dn_txt = ""
            if owner_debug is not None and owner_debug.get(tid) is not None:
                dn_txt = f" | dn={owner_debug[tid]:.3f}"

            text = f"{label} #{tid} {conf:.2f}{owner_txt}{dn_txt}"
            draw_label_with_bg(frame, x1, max(25, y1), text, box_color=COLOR_BG, text_color=COLOR_TEXT)

def draw_counts_panel(frame, counts: Counter):
    panel_x, panel_y = 8, 8
    panel_w, panel_h = 260, 130

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), COLOR_BG, -1)
    cv2.putText(
        frame, "Counts",
        (panel_x + 10, panel_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        COLOR_TEXT, 2
    )

    y = panel_y + 55
    for name in ["person", "backpack", "handbag", "suitcase"]:
        cv2.putText(
            frame,
            f"{name}: {counts.get(name, 0)}",
            (panel_x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            COLOR_COUNT,
            2
        )
        y += 26

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BagGuard System - Phase A Detection + Phase B Tracking + Phase C Ownership (persistent, normalized distance)"
    )
    parser.add_argument("--source", default="0", help="0 for webcam OR path to video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path/name")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Show window")
    parser.add_argument("--save", action="store_true", help="Save output video to outputs/")
    parser.add_argument("--out", default="outputs/detection_output.mp4", help="Output path if --save")
    args = parser.parse_args()

    model = YOLO(args.model)

    # Source handling
    source = 0 if str(args.source).strip() == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        print("Tip: use --source 0 for webcam or provide a valid video path.")
        return

    # Video writer (optional)
    writer = None
    if args.save:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-3:
            fps = 30.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    # Trackers
    person_tracker = SimpleTracker(match_dist_px=200, max_missed_sec=3.0, use_area_penalty=False)
    bag_tracker    = SimpleTracker(match_dist_px=220, max_missed_sec=5.0, use_area_penalty=True)

    # Ownership manager (C3)
    ownership_mgr = OwnershipManager(
        near_norm=0.12,
        min_score=1.0,
        decay_per_sec=0.60,
        lock_sec=1.0,
        switch_ratio=1.35,
        switch_min_sec=0.8,
        occlusion_grace_sec=2.0,
        margin=0.25
    )

    window_title = "BagGuard System – Phase A+B+C"

    prev_t = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = now_sec()
        if prev_t is None:
            dt = 1.0 / 30.0
        else:
            dt = max(1e-3, t - prev_t)
        prev_t = t

        results = model(frame, verbose=False, conf=args.conf)[0]

        det_person = []
        det_bags = []
        counts = Counter()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id not in TARGET_CLASS_IDS:
                continue

            label = model.names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            c = bbox_centroid(bbox)

            det = {
                "cls_id": cls_id,
                "label": label,
                "conf": conf,
                "bbox": bbox,
                "centroid": c,
            }

            counts[label] += 1

            if cls_id == 0:
                det_person.append(det)
            else:
                det_bags.append(det)

        assigned_people = person_tracker.update(det_person, t)
        assigned_bags   = bag_tracker.update(det_bags, t)

        # Build centroid maps
        people_centroids = {pid: det["centroid"] for pid, det in assigned_people}
        bag_centroids    = {bid: det["centroid"] for bid, det in assigned_bags}

        # Ownership update per bag
        bag_to_owner = {}
        bag_best_dn = {}
        for bid, bc in bag_centroids.items():
            owner_id, best_dn = ownership_mgr.update(
                bag_id=bid,
                bag_centroid=bc,
                people_centroids=people_centroids,
                frame=frame,
                t=t,
                dt=dt
            )
            bag_to_owner[bid] = owner_id
            bag_best_dn[bid] = best_dn

        # Draw tracked boxes + IDs
        draw_assigned(frame, assigned_people, COLOR_PERSON, person=True)
        draw_assigned(frame, assigned_bags, COLOR_BAG, person=False, ownership=bag_to_owner, owner_debug=bag_best_dn)

        # Draw counts
        draw_counts_panel(frame, counts)

        # Save frame (optional)
        if writer is not None:
            writer.write(frame)

        # Show (optional)
        if args.show:
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved output to: {args.out}")
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
