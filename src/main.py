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
# Helpers
# -----------------------------
def now_sec() -> float:
    return time.time()

def dist(p1, p2) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5

def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(1.0, float((x2 - x1) * (y2 - y1)))

# -----------------------------
# Simple Tracker (Centroid + TTL)
# Upgrades included:
#  - confidence sorting (Fix 2)
#  - optional bbox-area penalty (Fix 3) 
# -----------------------------

class SimpleTracker: # Phase B3: Maintains track IDs across frames, Removes stale IDs after max_missed_sec and Reduces ID flicker during short occlusions

    
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

        # Fix 2: sort by confidence (high -> low) to reduce ID swaps
        detections = sorted(detections, key=lambda d: d["conf"], reverse=True)

        used_track_ids = set()
        assigned = []

        for det in detections:
            best_id = None
            best_score = float("inf")

            det_c = det["centroid"]
            det_a = area(det["bbox"])

            for tid, tr in self.tracks.items():
                if tid in used_track_ids:
                    continue

                d = dist(det_c, tr["centroid"])

                # If centroid is too far, skip quickly
                # (we still use d threshold even with area penalty)
                if d > self.match_dist_px:
                    continue

                if self.use_area_penalty:
                    # Fix 3: penalize mismatch in bbox area (helps with bags swapping)
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
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="BagGuard System - Phase A Detection + Simple Tracking (less flicker)")
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
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-3:
            fps = 30.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    # Fix 1: make bags "stickier" (bigger match dist + longer missed TTL)
    # Phase B2: Separate trackers for people and bags
    person_tracker = SimpleTracker(match_dist_px=200, max_missed_sec=3.0, use_area_penalty=False)
    bag_tracker    = SimpleTracker(match_dist_px=220, max_missed_sec=5.0, use_area_penalty=True)  # Fix 3 enabled for bags

    window_title = "BagGuard System – Phase A Detection + Tracking"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = now_sec()

        results = model(frame, verbose=False, conf=args.conf)[0]

        # Collect detections filtered to target classes
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
            c = centroid(bbox)

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

        # Update trackers separately
        assigned_people = person_tracker.update(det_person, t)
        assigned_bags   = bag_tracker.update(det_bags, t)

        # Draw tracked boxes + IDs
        def draw_assigned(assigned, color=(255, 0, 255), person=False):
            for tid, det in assigned:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["conf"]
                label = det["label"]

                thickness = 3 if person else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                text = f"{label} #{tid} {conf:.2f}"
                cv2.putText(
                    frame, text,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

        draw_assigned(assigned_people, person=True)
        draw_assigned(assigned_bags, person=False)

        # Show counts on top-left
        y = 28
        for name in ["person", "backpack", "handbag", "suitcase"]:
            cv2.putText(
                frame,
                f"{name}: {counts.get(name, 0)}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )
            y += 26

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
