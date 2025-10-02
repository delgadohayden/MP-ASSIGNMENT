from ultralytics import YOLO
import cv2
import os
from glob import glob

# Load your trained YOLOv8 model
model = YOLO("runs/detect/split_digits/weights/best.pt")

# Input and output folders
input_folder = "DatasetTask2/train/images/"
output_folder = "DatasetTask3/"
os.makedirs(output_folder, exist_ok=True)

counter = 1
max_images = 50  # only process the first 100 images

# Get all image filenames
all_images = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg"))]

# Loop through only the first max_images files
for filename in all_images[:max_images]:
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read {img_path}")
        continue

    # Run YOLO
    results = model(img, conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print(f"No digits detected in {img_path}")
        continue

    # Sort boxes left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])

    # Save each digit as c0001.png, c0002.png, ...
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        digit_img = img[y1:y2, x1:x2]
        crop_filename = os.path.join(output_folder, f"c{counter:04d}.png")
        cv2.imwrite(crop_filename, digit_img)
        counter += 1

    print(f"Processed {img_path}, total crops saved: {counter-1}")

print(f"✅ Finished! Total digits saved: {counter-1}")