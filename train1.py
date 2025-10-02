from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data='DatasetTask1/data.yaml',
    epochs=20,
    imgsz=640,
    batch=16,
    name="building_numbers"
)