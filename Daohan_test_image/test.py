from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model(
    source="Daohan_test_image/bus.jpg",
    save=True,
    project="Daohan_test_image/outputs",
    name="bus_detect"
)