from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data='DatasetTask2/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name="split_digits"
)