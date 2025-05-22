import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("densenet_tomato_model.h5")

class_names = [
    "Bacterial Spot", 
    "Early Blight",   
    "Healthy",   
    "Late Blight",
    "Leaf Mold",
    "Septoria leaf spot",
    "Two Spotted Spider mite",
    "Target spot",
    "mosaic virus",
    "yellow leaf virus"
] 

def preprocess_image(img_path):
    """Loads and preprocesses image for prediction"""
    img = Image.open(img_path).resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path):
    """Predict disease using DenseNet model"""
    preprocessed = preprocess_image(img_path)
    prediction = model.predict(preprocessed)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).resize((256, 256))
        tk_img = ImageTk.PhotoImage(img)
        panel.config(image=tk_img)
        panel.image = tk_img

        result = predict_image(file_path)
        result_label.config(text="Predicted Disease: " + result)

root = tk.Tk()
root.title("Tomato Disease Classifier")
root.geometry("400x500")

panel = Label(root)
panel.pack(pady=20)

upload_btn = Button(root, text="Upload Image", command=upload_image)
upload_btn.pack()

result_label = Label(root, text="Prediction appears here", font=("Helvetica", 14))
result_label.pack(pady=20)

root.mainloop()
