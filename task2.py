

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
# Last Modified: 2025-30-09

import os
import cv2
import numpy as np
from ultralytics import YOLO

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
    
    # Load the image
    img = cv2.imread(image_path)

    # Run detection
    results = model(img, conf=0.5)

    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2

    # If no digits detected, skip
    if len(boxes) == 0:
        print(f"No characters detected in {image_path}")
        return

    # Prepare output folder
    output_dir = "output/task2/"
    os.makedirs(output_dir, exist_ok=True)

    # Sort boxes left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])

    # Extract and save each detected character
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        char_img = img[y1:y2, x1:x2]  # crop character
        char_filename = os.path.join(output_dir, f"c{idx+1}.png")
        cv2.imwrite(char_filename, char_img)
        print(f"✅ Saved {char_filename}")

    # Write a small result file (required by marking script)
    output_path = "output/task2/result.txt"
    save_output(output_path, "Task 2 output", output_type='txt')
    
    # # TODO: Implement task 2 here
    # filename = os.path.basename(image_path)
    # file_index = ''.join(filter(str.isdigit, filename))
    # output_dir = "output/task2/"
    # os.makedirs(output_dir, exist_ok=True)

    # # Read in image and convert to grayscale for thresholding
    # img = cv2.imread(image_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # # Otsu thresholding - best for outdoor conditions
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # # Crop to bounding box of all digits
    # coords = cv2.findNonZero(binary) # Return belonging pixels to foreground
    # x, y, w, h = cv2.boundingRect(coords) # Finds minimal rectangle that contains all foreground pixels
    # roi = binary[y:y+h, x:x+w] # Slicing for precise digit extraction

    # # Split ROI into 3 equal parts (column-based split)
    # digit_width = roi.shape[1] // 3
    # digits = []
    # for i in range(3):
    #     digit_img = roi[:, i*digit_width:(i+1)*digit_width]
    #     digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
    #     digits.append(digit_img)

    # # Save each digit
    # for idx, d in enumerate(digits):
    #     char_filename = os.path.join(output_dir, f"c{idx+1:02d}.png")
    #     save_output(char_filename, d, output_type='image')

    # # Save debug preview (all 3 digits side by side)
    # preview = cv2.hconcat(digits)
    # debug_filename = os.path.join(output_dir, f"bn{file_index}_debug.png")
    # save_output(debug_filename, preview, output_type='image')