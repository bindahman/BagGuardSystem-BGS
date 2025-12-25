from ultralytics import YOLO
import cv2
from collections import Counter
import argparse
import os

# COCO class IDs for YOLOv8 pretrained models
# 0 = person, 24 = backpack, 26 = handbag, 28 = suitcase
TARGET_CLASS_IDS = {0, 24, 26, 28}

def parse_args():
    parser = argparse.ArgumentParser(description="BagGuard System - Phase A Detection")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: 0 for webcam OR path to a video file (e.g. data/airport_demo.mp4)"
    )
    return parser.parse_args()

def open_capture(source: str):
    # If source is "0", "1", etc. treat as webcam index
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    # Otherwise treat as a video file path
    return cv2.VideoCapture(source)

def main():
    args = parse_args()
    model = YOLO("yolov8n.pt")

    cap = open_capture(args.source)

    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        print("Tip: use --source 0 for webcam or provide a valid video path.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        # Count filtered detections
        counts = Counter()

        # Draw filtered bounding boxes
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

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 255),
                2
            )

        # ===== Live count panel (top-left) =====
        cv2.rectangle(frame, (5, 5), (260, 135), (0, 0, 0), -1)

        y = 30
        for name in ["person", "backpack", "handbag", "suitcase"]:
            cv2.putText(
                frame,
                f"{name.capitalize()}: {counts.get(name, 0)}",
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y += 30

        cv2.imshow("BagGuard System – Phase A Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
