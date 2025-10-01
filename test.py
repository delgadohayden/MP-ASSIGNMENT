from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

model = YOLO("runs/detect/building_numbers3/weights/best.pt")

img = cv2.imread("Dataset/valid/images/280FE49C-63BC-47F6-A047-A12627771931_jpg.rf.fe78771b2e660fd327ad4f5efa6dd32e.jpg")

results = model(img, conf=0.5)
    
boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
confidences = results[0].boxes.conf.cpu().numpy()

if len(results[0].boxes) == 0:
    print("No building numbers detected.")
else:
   cv2.imwrite("output.png", results[0].plot(line_width=2))

print("Saved output.png with detections")