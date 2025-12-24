from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8n.pt")  # lightweight model to start
    cap = cv2.VideoCapture(0)   # webcam (change to video later)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("BagGuard System - YOLO Test", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
