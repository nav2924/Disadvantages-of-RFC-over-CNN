import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image
import requests
from io import BytesIO
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_data = tf.keras.utils.image_dataset_from_directory(
    'tomato/train',
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32
).map(lambda x, y: (x / 255.0, y))

val_data = tf.keras.utils.image_dataset_from_directory(
    'tomato/val',
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32
).map(lambda x, y: (x / 255.0, y))

# Define DenseNet121 model
conv_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3), pooling='avg')
conv_base.trainable = False

# Feature extraction function
def extract_features(data):
    features, labels = [], []
    for images, label in data:
        features.append(conv_base.predict(images))
        labels.append(label)
    return np.vstack(features), np.vstack(labels)

# Extract features
train_features, train_labels = extract_features(train_data)
val_features, val_labels = extract_features(val_data)

# Flatten labels for compatibility
train_labels = np.argmax(train_labels, axis=1)
val_labels = np.argmax(val_labels, axis=1)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(train_features, train_labels)

# Predictions and confusion matrix for RF
rf_predictions = rf_classifier.predict(val_features)
print("Random Forest Accuracy:", accuracy_score(val_labels, rf_predictions))
print("Confusion Matrix (Random Forest):")
print(confusion_matrix(val_labels, rf_predictions))

# Simple CNN model
def create_simple_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# Compile and train simple CNN
simple_cnn = create_simple_cnn()
simple_cnn.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
simple_cnn.fit(train_data, epochs=10, validation_data=val_data, callbacks=[EarlyStopping(patience=3)])

# Evaluate simple CNN
simple_cnn_eval = simple_cnn.evaluate(val_data)
print("Simple CNN Accuracy:", simple_cnn_eval[1])

# DenseNet-based CNN
def create_densenet_cnn():
    model = Sequential([
        conv_base,
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.35),
        BatchNormalization(),
        Dense(120, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Compile and train DenseNet-based CNN
densenet_cnn = create_densenet_cnn()
densenet_cnn.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
densenet_cnn.fit(train_data, epochs=10, validation_data=val_data, callbacks=[EarlyStopping(patience=3)])

# Evaluate DenseNet-based CNN
densenet_cnn_eval = densenet_cnn.evaluate(val_data)
print("DenseNet CNN Accuracy:", densenet_cnn_eval[1])

# Image prediction function for Random Forest, Simple CNN, and DenseNet CNN
def predict_from_url(url, model, model_type='densenet'):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).resize((256, 256))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    if model_type == 'rf':
        # Random Forest Model expects the features extracted via DenseNet
        features = conv_base.predict(img_array)
        prediction = model.predict(features)
        predicted_label = prediction  # Random Forest gives a label directly
    elif model_type == 'simple_cnn':
        # Simple CNN directly predicts from the image
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction, axis=-1)  # Ensure axis=-1 for a single dimension output
    elif model_type == 'densenet':
        # DenseNet-based CNN directly predicts from the image
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction, axis=-1)  # Ensure axis=-1 for a single dimension output

    return predicted_label

# Example URL prediction for all models
example_url = "https://example.com/image.jpg"  # Replace with an actual image URL
predicted_rf_disease = predict_from_url(example_url, rf_classifier, model_type='rf')
predicted_simple_cnn_disease = predict_from_url(example_url, simple_cnn, model_type='simple_cnn')
predicted_densenet_disease = predict_from_url(example_url, densenet_cnn, model_type='densenet')

print("Predicted Disease Label (Random Forest):", predicted_rf_disease)
print("Predicted Disease Label (Simple CNN):", predicted_simple_cnn_disease)
print("Predicted Disease Label (DenseNet CNN):", predicted_densenet_disease)

# Confusion matrix for DenseNet-based CNN
densenet_predictions = np.argmax(densenet_cnn.predict(val_data), axis=1)
cm = confusion_matrix(val_labels, densenet_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - DenseNet CNN')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
