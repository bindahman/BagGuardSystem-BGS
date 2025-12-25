from ultralytics import YOLO
import cv2
from collections import Counter
import argparse
import os
from datetime import datetime

TARGET_CLASS_IDS = {0, 24, 26, 28}  # person, backpack, handbag, suitcase

def parse_args():
    p = argparse.ArgumentParser("BagGuard System - Phase A Detection")
    p.add_argument("--source", type=str, default="0",
                   help="0 for webcam OR path to a video file")
    p.add_argument("--save", action="store_true",
                   help="Save annotated output video to outputs/")
    return p.parse_args()

def open_capture(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)

def make_writer(frame, out_path, fps):
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))

def main():
    args = parse_args()
    model = YOLO("yolov8n.pt")

    cap = open_capture(args.source)
    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        return

    # Create outputs folder if not exists
    os.makedirs("outputs", exist_ok=True)

    writer = None
    out_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        counts = Counter()

        # Draw filtered boxes
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id not in TARGET_CLASS_IDS:
                continue

            label = model.names[cls_id]
            counts[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            thickness = 3 if cls_id == 0 else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), thickness)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

        # Count panel
        cv2.rectangle(frame, (5, 5), (260, 135), (0, 0, 0), -1)
        y = 30
        for name in ["person", "backpack", "handbag", "suitcase"]:
            cv2.putText(frame, f"{name.capitalize()}: {counts.get(name, 0)}",
                        (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30

        # Setup writer on first frame (only if saving AND source is a video file)
        if args.save and writer is None and not args.source.isdigit():
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps == 0:
                fps = 25.0

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join("outputs", f"annotated_{stamp}.mp4")
            writer = make_writer(frame, out_path, fps)
            print(f"Saving output to: {out_path}")

        # Write frame if saving
        if writer is not None:
            writer.write(frame)

        cv2.imshow("BagGuard System – Phase A Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved: {out_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
