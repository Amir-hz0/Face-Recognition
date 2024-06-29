import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
from PIL import Image
import os

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = len(os.listdir('dArchive\\train'))

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load('resnet50_vggface2.pth'))
model = model.to(device)
model.eval()

# Data Transformation
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
class_names = os.listdir('dArchive\\train')

def predict_frame(frame):
    image = Image.fromarray(frame)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = class_names[predicted.item()]
    return class_name

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Predict the frame
        class_name = predict_frame(rgb_frame)
        
        # Display the prediction on the frame
        cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show the frame
        cv2.imshow('Webcam', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
