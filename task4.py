

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
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Load YOLO model used in Task 1
task1_model = YOLO("data/building_numbers3/weights/best.pt")

# Load CNN model used in Task 3
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
task3_model.load_state_dict(torch.load("data/digit_cnn.pth", map_location=device))
task3_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def run_task4(image_path, config):
    img = cv2.imread(image_path)

    # Task 1: detect building number regions
    results = task1_model(img, conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        # negative image → no output
        return

    # Extract bounding box of building number
    x1, y1, x2, y2 = map(int, boxes[0])
    bn_img = img[y1:y2, x1:x2]

    # Crop individual digits using Task2 YOLO model
    task2_model = YOLO("data/split_digits/weights/best.pt")
    digit_results = task2_model(bn_img, conf=0.5) # Execute Task 2 model on cropped image
    digit_boxes = digit_results[0].boxes.xyxy.cpu().numpy() # Split and extract individual digits
    if len(digit_boxes) == 0:
        return  # No digits detected

    # Sort digits left to right
    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

    # Recognize digits using CNN
    building_number = "" # String to store full number
    for box in digit_boxes:
        
        # Extract each digit's bounding box and crop
        x1d, y1d, x2d, y2d = map(int, box)
        digit_img = bn_img[y1d:y2d, x1d:x2d]

        # Preprocess digit for CNN
        tensor_img = transform(digit_img).unsqueeze(0).to(device)
        
        # Execute CNN used in Task 3 and append predicted digit to string
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

