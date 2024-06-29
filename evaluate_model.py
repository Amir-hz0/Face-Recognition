import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import precision_score, mean_squared_error

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
NUM_CLASSES = len(os.listdir('dArchive\\train'))

# Data Transformations for Test Set
test_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Test Dataset
test_dataset = ImageFolder(root='dArchive\\test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Model
model = models.resnet50()  # Initialize without pre-trained weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

# Load Pre-trained Weights
model.load_state_dict(torch.load('resnet50_vggface2.pth'))
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct_test / total_test

    # Calculate Precision
    precision = precision_score(all_labels, all_preds, average='weighted')
    
    
    # Calculate MSE
    mse = mean_squared_error(all_labels, all_preds)

    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'MSE: {mse:.4f}')
    
    return avg_test_loss, test_accuracy, precision, mse


if __name__ == '__main__':
    # Evaluate the model
    test_loss, test_accuracy, precision, mse = evaluate_model(model, test_loader, criterion)
    
    # Plot confusion matrix
    class_names = test_dataset.classes  # Class names from the dataset