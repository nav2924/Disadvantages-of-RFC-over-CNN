
# Tomato Leaf Detection using Random Forest , CNN and Densenet CNN

This project involves building and comparing different machine learning models to classify tomato leaf diseases. The dataset consists of images of tomato leaves, and the goal is to correctly classify the type of disease present in each image. The project uses transfer learning with DenseNet121, a custom Convolutional Neural Network (CNN), and a Random Forest classifier.

## Introduction
Tomato diseases can significantly impact agricultural productivity. This project aims to develop a model that can accurately classify tomato leaf diseases from images using various machine learning techniques.

## Dataset
The dataset used in this project consists of images of tomato leaves with different diseases. The dataset is organized into train and validation directories, with each directory containing subdirectories for each disease category.


## Installation

Clone the repository

```bash
git clone https://github.com/nav2924/Tomato-Disease-Classification.git
cd Tomato-Disease-Classification

```

Install the required packages:


```bash
pip install -r requirements.txt
```

Ensure the dataset is placed in the appropriate directory structure:

```bash
tomato/
├── train/
│   ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   ├── Tomato___Bacterial_spot/
│   └── ...
└── val/
    ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
    ├── Tomato___Bacterial_spot/
    └── ...
```
## Model Architectures
### DenseNet121 with Random Forest
Feature Extraction: DenseNet121 is used as a feature extractor.
Classification: A Random Forest classifier is used to classify the extracted features.
### Custom CNN
A custom Convolutional Neural Network (CNN) is built with the following architecture:

Conv2D and MaxPooling2D layers for feature extraction.
BatchNormalization and Dropout layers for regularization.
Dense layers for classification.
### Training
DenseNet121 with Random Forest:

Extract features from images using DenseNet121.
Train a Random Forest classifier on the extracted features.
Custom CNN:

Train a custom CNN model end-to-end on the image dataset.
### Evaluation
The models are evaluated based on their accuracy on the validation set. The training and validation accuracies are plotted to visualize the model performance over epochs.

### Results
The final accuracies of the models are compared and visualized in a bar plot:

Random Forest Accuracy , 
DenseNet121 CNN Accuracy , 
Regular CNN Accuracy .
### Conclusion
This project demonstrates the effectiveness of transfer learning and custom CNN architectures in classifying tomato leaf diseases. The DenseNet121 with Random Forest model and the custom CNN both showed promising results.
    
