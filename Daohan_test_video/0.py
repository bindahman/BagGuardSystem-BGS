from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model(
    source="Daohan_test_video/airport_demo_clip1.mp4",
    save=True,
    project=r"Daohan_test_video/outputs",
    name="airport_detect"
)