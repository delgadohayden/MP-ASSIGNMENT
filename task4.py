

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


# Author: [Your Name]
# Last Modified: 2024-09-09

import os
import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# ---------------- TASK 1 MODEL ----------------
task1_model = YOLO("runs/detect/building_numbers3/weights/best.pt")

# ---------------- TASK 3 MODEL ----------------
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task3_model = DigitCNN().to(device)
task3_model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
task3_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------- RUN TASK 4 ----------------
def run_task4(image_path, config):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return

    # Task 1: detect building number regions
    results = task1_model(img, conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        # negative image → no output
        return

    # Only handle first detected building number (can extend if multiple)
    x1, y1, x2, y2 = map(int, boxes[0])
    bn_img = img[y1:y2, x1:x2]

    # Task 2: detect and crop individual digits using Task2 YOLO model
    task2_model = YOLO("runs/detect/split_digits/weights/best.pt")
    digit_results = task2_model(bn_img, conf=0.5)
    digit_boxes = digit_results[0].boxes.xyxy.cpu().numpy()
    if len(digit_boxes) == 0:
        return  # No digits detected

    # Sort digits left to right
    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

    building_number = ""
    for box in digit_boxes:
        x1d, y1d, x2d, y2d = map(int, box)
        digit_img = bn_img[y1d:y2d, x1d:x2d]

        # Task 3: recognize digit
        tensor_img = transform(digit_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = task3_model(tensor_img)
            pred = output.argmax(dim=1).item()
            building_number += str(pred)

    # Save the building number
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"output/task4/{filename}.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(building_number)

    print(f"✅ {image_path} -> {output_path}: {building_number}")

