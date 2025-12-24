from ultralytics import YOLO
import cv2
from collections import Counter

# COCO class IDs for YOLOv8 pretrained models
# 0=person, 24=backpack, 26=handbag, 28=suitcase
TARGET_CLASS_IDS = {0, 24, 26, 28}

def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)  # webcam for now
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        # Count filtered detections
        counts = Counter()

        # Draw boxes manually (filtered)
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id not in TARGET_CLASS_IDS:
                continue

            label = model.names[cls_id]
            counts[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Different thickness for people vs bags (readability)
            is_person = (cls_id == 0)
            thickness = 3 if is_person else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), thickness)

            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 255),
                2
            )

        # Show counts at top-left
        y = 25
        for name in ["person", "backpack", "handbag", "suitcase"]:
            cv2.putText(
                frame,
                f"{name}: {counts.get(name, 0)}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y += 25

        cv2.imshow("BagGuard System - Phase A Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
