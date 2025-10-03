

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

# Load YOLO model using weights trained on dataset
model = YOLO("data/building_numbers3/weights/best.pt")

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


def run_task1(image_path, config):

    # Check if output folder exists
    output_dir = "output/task1/"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all images in input folder
    if os.path.isdir(image_path):
        image_files = [os.path.join(image_path, f) 
                       for f in os.listdir(image_path) 
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_files = [image_path]  # Pass through single image

    for img_path in image_files:
        try:
            # Read in image
            img = cv2.imread(img_path)

            # Run detection with confidence threshold of 0.5
            # (filters out detections with <50% confidence)
            results = model(img, conf=0.5)
            
            # Extract detected bounding boxes as numpy array
            # Format: [x1, y1, x2, y2]
            boxes = results[0].boxes.xyxy.cpu().numpy()

             # If no bounding boxes detected, skip (negative input image)
            if len(boxes) == 0:
                print(f"No building numbers detected in {os.path.basename(img_path)}")
                continue

            # Convert coordinates of the first bounding box to integers
            # Assumes one building number per image
            x1, y1, x2, y2 = map(int, boxes[0])
            bn_img = img[y1:y2, x1:x2] # Crop image using its bounding box

            # Extract digits from filename (used for output naming)
            filename = os.path.basename(img_path)
            index = ''.join(filter(str.isdigit, filename))
            bn_filename = os.path.join(output_dir, f"bn{index}.png") # Adds 'bn' prefix to file name

            cv2.imwrite(bn_filename, bn_img)
            print(f"Successfully saved {bn_filename}")

        except Exception as e:
            print(f"Error processing file {img_path}: {e}")