# src/main.py
from ultralytics import YOLO
import cv2
import time
import os
import argparse
from collections import Counter

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

# -----------------------------
# Phase C2: Nearest-person association (per frame)
# -----------------------------
MAX_ASSOC_DIST_PX = 180  # if nearest person farther than this, bag is UNASSIGNED

# -----------------------------
# Helpers (time, geometry, distance)
# -----------------------------
def now_sec() -> float:
    return time.time()

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
    """0.0–1.0-ish scale (distance divided by frame diagonal)."""
    d = euclid(p1, p2)
    diag = max(1.0, frame_diag(frame))
    return d / diag

def nearest_person_for_bag(bag_centroid, assigned_people, max_dist_px):
    """
    assigned_people: list of (person_id, det_dict)
    returns: (best_person_id or None, best_dist_px or None)
    """
    best_pid = None
    best_d = float("inf")

    for pid, pdet in assigned_people:
        d = euclid(bag_centroid, pdet["centroid"])
        if d < best_d:
            best_d = d
            best_pid = pid

    if best_pid is None:
        return None, None

    if best_d > max_dist_px:
        return None, best_d  # too far -> treat as unassigned, but keep distance info
    return best_pid, best_d

# -----------------------------
# Simple Tracker (Centroid + TTL)
# Upgrades included:
#  - confidence sorting (Fix 2)
#  - optional bbox-area penalty (Fix 3)
# -----------------------------
class SimpleTracker:
    def __init__(self, match_dist_px=200, max_missed_sec=3.0, use_area_penalty=False):
        self.match_dist_px = float(match_dist_px)
        self.max_missed_sec = float(max_missed_sec)
        self.use_area_penalty = bool(use_area_penalty)

        self.next_id = 1
        self.tracks = {}  # id -> dict {centroid, bbox, area, last_seen}

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

        # Sort by confidence (high -> low) to reduce ID swaps
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

                d = euclid(det_c, tr["centroid"])

                # too far -> skip
                if d > self.match_dist_px:
                    continue

                if self.use_area_penalty:
                    tr_a = tr.get("area", det_a)
                    ratio = det_a / max(tr_a, 1e-6)
                    ratio = max(ratio, 1.0 / max(ratio, 1e-6))  # make >= 1
                    score = d * ratio
                else:
                    score = d

                if score < best_score:
                    best_score = score
                    best_id = tid

            if best_id is None:
                # Create new track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "centroid": det_c,
                    "bbox": det["bbox"],
                    "area": det_a,
                    "last_seen": now_t,
                }
                used_track_ids.add(tid)
                assigned.append((tid, det))
            else:
                # Update track
                tr = self.tracks[best_id]
                tr["centroid"] = det_c
                tr["bbox"] = det["bbox"]
                tr["area"] = det_a
                tr["last_seen"] = now_t

                used_track_ids.add(best_id)
                assigned.append((best_id, det))

        return assigned

# -----------------------------
# Drawing helpers (Phase B4 + Phase C2 labels)
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

def draw_assigned(frame, assigned, color, person=False, bag_assoc=None):
    """
    bag_assoc: dict bag_id -> (owner_pid or None, dist_px or None)
    """
    for tid, det in assigned:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        label = det["label"]

        thickness = 3 if person else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if (not person) and bag_assoc is not None:
            owner_pid, dpx = bag_assoc.get(tid, (None, None))
            if owner_pid is None:
                extra = f"owner: -  d:{dpx:.0f}px" if dpx is not None else "owner: -"
            else:
                extra = f"owner: P{owner_pid}  d:{dpx:.0f}px"
            text = f"{label} #{tid} {conf:.2f} | {extra}"
        else:
            text = f"{label} #{tid} {conf:.2f}"

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
        description="BagGuard System - Phase A Detection + Phase B Tracking + Phase C (C1+C2)"
    )
    parser.add_argument("--source", default="0", help="0 for webcam OR path to video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path/name")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Show window")
    parser.add_argument("--save", action="store_true", help="Save output video to outputs/")
    parser.add_argument("--out", default="outputs/detection_output.mp4", help="Output path if --save")
    args = parser.parse_args()

    model = YOLO(args.model)

    source = 0 if str(args.source).strip() == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        print("Tip: use --source 0 for webcam or provide a valid video path.")
        return

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

    person_tracker = SimpleTracker(match_dist_px=200, max_missed_sec=3.0, use_area_penalty=False)
    bag_tracker    = SimpleTracker(match_dist_px=220, max_missed_sec=5.0, use_area_penalty=True)

    window_title = "BagGuard System – Phase A+B+C (C1+C2)"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = now_sec()
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

        # Phase C2: per-frame bag -> nearest person association
        bag_associations = {}  # bag_id -> (person_id or None, dist_px or None)
        for bid, bdet in assigned_bags:
            owner_pid, dpx = nearest_person_for_bag(bdet["centroid"], assigned_people, MAX_ASSOC_DIST_PX)
            bag_associations[bid] = (owner_pid, dpx)

        draw_assigned(frame, assigned_people, COLOR_PERSON, person=True)
        draw_assigned(frame, assigned_bags, COLOR_BAG, person=False, bag_assoc=bag_associations)
        draw_counts_panel(frame, counts)

        if writer is not None:
            writer.write(frame)

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
