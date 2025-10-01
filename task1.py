

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
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.color import label2rgb

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
        
def run_task1(image_path, config=None):
    filename = os.path.basename(image_path)
    file_index = ''.join(filter(str.isdigit, filename))
    output_path = f"output/task1/bn{file_index}.png"

    # Step 1: Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    thresh = cv2.adaptiveThreshold(enhanced,255,1,1,11,2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    debug_img = img.copy()


    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Heuristic filtering
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)

        if area < 300 or w < 20 or h < 20:
            continue  # too small

        if 1.0 < aspect_ratio < 10.0 and h > 20 and w > 20:
            candidates.append((x, y, w, h))

    if not candidates:
        print(f"[INFO] No valid number region found in {filename}")
        return  # Negative image — do not output anything


    # Step 4: Pick the largest candidate by area
    best_box = max(candidates, key=lambda b: b[2] * b[3])
    x, y, w, h = best_box

    # Optional: Add padding
    pad = 5
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])

    # Step 5: Crop and save
    cropped = img[y1:y2, x1:x2]
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Debug: show image with all contours and selected region
    cv2.imshow(f"[DEBUG] Detection in {filename}", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save_output(output_path, cropped, output_type='image')