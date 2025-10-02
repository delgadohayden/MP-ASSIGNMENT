

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
# Last Modified: 2025-15-09

import os
import cv2 
import numpy as np
from ultralytics import YOLO

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
    img = cv2.imread(image_path)
    
    results = model(img, conf=0.5)
    
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2

    # If no boxes detected, skip
    if len(boxes) == 0:
        print(f"No building numbers detected in {image_path}")
        return

    # Prepare output folder
    output_dir = "output/task1/"
    os.makedirs(output_dir, exist_ok=True)

    # Extract and save each detected building number
    if len(boxes) == 0:
        print(f"No building numbers detected in {image_path}")

    # Only take the first detected region (Task 1 expects a single bn file per image)
    x1, y1, x2, y2 = map(int, boxes[0])
    bn_img = img[y1:y2, x1:x2]

    # Get input image index from filename
    filename = os.path.basename(image_path)       # e.g., "img1.jpg"
    index = ''.join(filter(str.isdigit, filename)) # "1"

    bn_filename = os.path.join(output_dir, f"bn{index}.png")
    cv2.imwrite(bn_filename, bn_img)
    print(f"✅ Saved {bn_filename}")

    # for idx, box in enumerate(boxes):
    #     x1, y1, x2, y2 = map(int, box)
    #     bn_img = img[y1:y2, x1:x2]  # crop the building number
    #     bn_filename = os.path.join(output_dir, f"bn{idx+1}.png")
    #     cv2.imwrite(bn_filename, bn_img)
    #     print(f"✅ Saved {bn_filename}")
        
    output_path = f"output/task1/result.txt"
    save_output(output_path, "Task 1 output", output_type='txt')