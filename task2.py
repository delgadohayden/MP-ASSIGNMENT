

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Hayden Delgado
# Last Modified: 2025-03-10

import os
import cv2
import numpy as np
from ultralytics import YOLO

# # Load YOLO model using weights trained on dataset simulating Task 1 outputs
model = YOLO("data/split_digits/weights/best.pt")

def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")


def run_task2(image_path, config):
    # Loop through all images in input folder
    if os.path.isdir(image_path):
        image_files = [os.path.join(image_path, f)
                       for f in os.listdir(image_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_files = [image_path] # Pass through single image

    for img_path in image_files:
        try:
            # Read in image
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Could not read {img_path}")
                continue

            # Run detection with confidence threshold of 0.5
            # (filters out detections with <50% confidence)
            results = model(img, conf=0.5)

            # Extract detected bounding boxes as numpy array
            # Format: [x1, y1, x2, y2]
            boxes = results[0].boxes.xyxy.cpu().numpy()

            # Negative input
            if len(boxes) == 0:
                print(f"No characters detected in {os.path.basename(img_path)}")
                continue

            # Sort detection boxes by leftmost x coordinate
            # Ensures characters are saved left-to-right
            boxes = sorted(boxes, key=lambda b: b[0])

            # Save input images to their corresponding output folder
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            output_dir = os.path.join("output/task2", base_name)  # Save inside bnX folder

            # Loop through each detected character
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                char_img = img[y1:y2, x1:x2]

                char_filename = os.path.join(output_dir, f"c{idx+1}.png")
                cv2.imwrite(char_filename, char_img)
                print(f"Successfully saved {char_filename}")

        except Exception as e:
            # Catch unexpected errors so loop continues
            print(f"Error processing {img_path}: {e}")
