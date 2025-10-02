

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
    # TODO: Implement task 1 here
    
    # Read in image
    img = cv2.imread(image_path)
    
    # Set confidence threshold to 0.5
    # to filter out false positives (detections falling below 50%)
    results = model(img, conf=0.5)
    
    # Extract detected bounding boxes as numpy array
    # x1, y1, x2, y2 format
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # If no bounding boxes detected, skip (negative input)
    if len(boxes) == 0:
        print(f"No building numbers detected")
        return

    # Prepare output folder
    output_dir = "output/task1/"
    os.makedirs(output_dir, exist_ok=True)

    # Extract and save each detected building number
    if len(boxes) == 0:
        print(f"No building numbers detected")

    # Convert coordinates of bounding box to integers
    x1, y1, x2, y2 = map(int, boxes[0])
    bn_img = img[y1:y2, x1:x2] # Use slicing to crop image to bounding box

    # Extract file name
    filename = os.path.basename(image_path)
    index = ''.join(filter(str.isdigit, filename)) # Extract digits from filename

    # Generate output filename using extracted index
    bn_filename = os.path.join(output_dir, f"bn{index}.png")
    cv2.imwrite(bn_filename, bn_img) # Write cropped image to designated output path
    print(f"File saved successfully.")