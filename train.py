from ultralytics import YOLO

model = YOLO("yolov8n.pt")
print(model.model)

# model.train(
#     data='Dataset/data.yaml',
#     epochs=20,
#     imgsz=640,
#     batch=16,
#     name="building_numbers"
# )