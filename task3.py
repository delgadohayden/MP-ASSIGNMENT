

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

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

# CNN architecture for digit classification trained using dataset made from Task 2
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        
        # Convolution layers: (channel, feature maps, kernel size, strides)
        
        # First convolution layer 
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128) # Outputs 128 neurons
        self.fc2 = nn.Linear(128, 10) # Outputs 10 neurons (digits 0-9)

    # Forward pass
    def forward(self, x):
        x = F.relu(self.conv1(x)) # First convolution + ReLU activation
        x = F.relu(self.conv2(x)) # Second convolution + ReLU activation
        x = F.max_pool2d(x, 2) # Downsample using 2x2 max pooling
        x = torch.flatten(x, 1) # Flatten feature maps into single vector (preserve batch size)
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available, otherwise use CPU
model = DigitCNN().to(device)
model.load_state_dict(torch.load("data/digit_cnn.pth", map_location=device)) # Load trained weights
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),    # Convert numpy/OpenCV image to PIL format as required by torchvision transforms
    transforms.Grayscale(),     # Convert to grayscale
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def run_task3(image_path, config):
    
    # Determine if input is a folder
    if os.path.isdir(image_path):
        bn_folders = [os.path.join(image_path, f) for f in os.listdir(image_path)
                  if os.path.isdir(os.path.join(image_path, f))]

    for bn_folder in bn_folders:
        try:
        # Loop through images in each subfolder
            image_files = [os.path.join(bn_folder, f)
                        for f in os.listdir(bn_folder)
                        if f.lower().endswith(".png")]

            for img_path in image_files:
                # Read in image (grayscale)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"⚠️ Could not read {img_path}")
                    continue

                # Preprocess and predict
                tensor_img = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(tensor_img)
                    pred = output.argmax(dim=1, keepdim=True).item()

                # Prepare output folder: output/task3/bnX
                bn_name = os.path.basename(bn_folder)  # bn1, bn2, ...
                output_dir = os.path.join("output/task3", bn_name)
                os.makedirs(output_dir, exist_ok=True)

                # Save prediction as txt
                base_name = os.path.splitext(os.path.basename(img_path))[0]  
                txt_filename = os.path.join(output_dir, f"{base_name}.txt")
                with open(txt_filename, "w") as f:
                    f.write(str(pred))

                print(f"Saved {txt_filename} (digit {pred})")

        except Exception as e:
            print(f"Error processing: {e}")