import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load classification model
model = tf.keras.models.load_model("densenet_tomato_model.h5")

# Class names for predictions
class_names = [
    "Bacterial Spot", "Early Blight", "Healthy", "Late Blight",
    "Leaf Mold", "Septoria leaf spot", "Two Spotted Spider mite",
    "Target spot", "mosaic virus", "yellow leaf virus"
]

def apply_dummy_mask(image):
    """Simulates a leaf mask by cropping the center of the image"""
    img_array = np.array(image)
    h, w, _ = img_array.shape
    crop_size = 180
    startx = w // 2 - crop_size // 2
    starty = h // 2 - crop_size // 2
    cropped = img_array[starty:starty+crop_size, startx:startx+crop_size]
    resized = Image.fromarray(cropped).resize((256, 256))
    return np.array(resized)

def predict_image_with_dummy_mask(img_path):
    img = Image.open(img_path).resize((256, 256)).convert("RGB")
    
    # Apply dummy mask
    masked_img_array = apply_dummy_mask(img) / 255.0
    masked_img_array = np.expand_dims(masked_img_array, axis=0)

    # Make prediction
    prediction = model.predict(masked_img_array)
    predicted_class = np.argmax(prediction)

    # Convert back to image for display
    display_img = Image.fromarray((masked_img_array[0] * 255).astype(np.uint8))
    return class_names[predicted_class], display_img

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result, display_img = predict_image_with_dummy_mask(file_path)

        tk_img = ImageTk.PhotoImage(display_img)
        panel.config(image=tk_img)
        panel.image = tk_img

        result_label.config(text="Predicted Disease: " + result)

# GUI setup
root = tk.Tk()
root.title("Tomato Disease Classifier with Dummy Masking")
root.geometry("400x500")

panel = Label(root)
panel.pack(pady=20)

upload_btn = Button(root, text="Upload Image", command=upload_image)
upload_btn.pack()

result_label = Label(root, text="Prediction appears here", font=("Helvetica", 14))
result_label.pack(pady=20)

root.mainloop()
