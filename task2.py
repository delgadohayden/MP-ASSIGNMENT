

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
    
    # Read in image
    img = cv2.imread(image_path)

    # Set confidence threshold to 0.5
    # to filter out false positives (detections falling below 50%)
    results = model(img, conf=0.5)

    # Extract detected bounding boxes as numpy array
    # x1, y1, x2, y2 format
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2

    # Extract and save each detected building number
    if len(boxes) == 0:
        print(f"No building numbers detected")

    # Prepare output folder
    output_dir = "output/task2/"
    os.makedirs(output_dir, exist_ok=True)

    # Sort detection boxes by leftmost x coordinate
    # Ensures left-to-right order
    boxes = sorted(boxes, key=lambda b: b[0])

    # Loop through all bounding boxes with index
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box) # Convert floating point coordinates to integers
        char_img = img[y1:y2, x1:x2]  # Crop digit character from image
        char_filename = os.path.join(output_dir, f"c{idx+1}.png")
        cv2.imwrite(char_filename, char_img)
        print(f"Image saved successfully.")
