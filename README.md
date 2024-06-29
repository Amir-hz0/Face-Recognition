# Face Recognition using VGGFace2 and ResNet50

This project implements face recognition using the VGGFace2 dataset and ResNet50 model. The project includes dataset preparation, training, validation, and real-time face recognition using a webcam.

## Abstract

This project focuses on face recognition using deep learning techniques. The VGGFace2 dataset is utilized, and a ResNet50 model is trained to recognize faces. The model is evaluated on a test dataset and used for real-time face recognition via webcam. The results demonstrate high accuracy and efficiency in face recognition tasks.

## Introduction

Face recognition is a crucial application in computer vision with significant importance in security, authentication, and social media. This project explores the capabilities of deep learning models in accurately identifying faces from images and videos. We employ the VGGFace2 dataset and train a ResNet50 model for this purpose. 

## Materials and Methods

### Dataset
First Download VggFace2 dataset via Kaggle.com Then run this file "Dataset Preparation.py" to divide dataset for you.
The VGGFace2 dataset is used for training, validation, and testing. The dataset is divided into three subsets:
- Training (70%)
- Validation (20%)
- Test (10%)

### Tools and Libraries

The following tools and libraries are used in this project:
- Python
- PyTorch
- OpenCV
- TensorFlow

### Methods

1. **Data Preparation**: The dataset is preprocessed and divided into training, validation, and test sets.
2. **Model Training**: ResNet50 model is trained on the VGGFace2 dataset using PyTorch.
3. **Evaluation**: The trained model is evaluated on the test set for accuracy, weighted accuracy, and mean squared error.
4. **Real-time Recognition**: The model is used for real-time face recognition using a webcam.

## How to Run the Project

### Prerequisites

Ensure you have the following libraries installed:

- Python 3.7+
- PyTorch
- torchvision
- OpenCV
- numpy
- scikit-learn
- matplotlib

You can install the required libraries using the following command:

```bash
pip install torch torchvision opencv-python numpy scikit-learn matplotlib
```

### Running the Code

1. **Dataset Preparation**

```bash
python dataset_preparation.py
```

2. **Model Training**

```bash
python train_model.py
```

3. **Model Evaluation**

```bash
python evaluate_model.py
```

4. **Real-time Face Recognition**

```bash
python real_time_recognition.py
```

### Files and Directories

- `dataset_preparation.py`: Script for preparing the dataset.
- `train_model.py`: Script for training the ResNet50 model.
- `evaluate_model.py`: Script for evaluating the trained model.
- `real_time_recognition.py`: Script for real-time face recognition using a webcam.
- `archive/`: Example directory containing the VGGFace2 dataset.

## Results

The results of the project include:
- High accuracy in face recognition on the VGGFace2 dataset.
- Successful real-time face recognition using a webcam.

## Conclusion

This project demonstrates the effectiveness of deep learning models, specifically ResNet50, in face recognition tasks. The use of the VGGFace2 dataset ensures a diverse and comprehensive training set, leading to robust and accurate face recognition in real-world scenarios.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
