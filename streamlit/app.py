import streamlit as st 
import os
import pathlib
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import dvc.api
import requests

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_file


model = None
class_name = None

current_dir = os.path.dirname(os.path.abspath(__file__))

class_name_URL = "https://drive.google.com/uc?id=1p1fGmfIgoXWevO1l-cSRavLbcfSU1SXY&export=download"
model_URL = "https://drive.google.com/uc?id=1ofu7smGB7D2rwce_-1Tiz4tk8Wfwjpqn&export=download"
model_file = os.path.join(current_dir, "model.h5")
class_name_file = os.path.join(current_dir, "class_name.pkl")


def preprocessing(filename) -> np.ndarray:
    img = Image.open(BytesIO(filename))
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    return img_array


def prediction(filename, model, class_name):
    img_array = preprocessing(filename)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])

    predicted_class = class_name[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class, confidence


def main():
    picture = st.camera_input("Capture")

    global model
    global class_name

    if model is None:
        response = requests.get(model_URL)
        if response.status_code == 200:
            with open(model_file, "wb") as f:
                f.write(response.content)
            st.write("Model downloaded successfully!")
            model = tf.keras.models.load_model(model_file)

    if class_name is None:
        response = requests.get(class_name_URL)
        if response.status_code == 200:
            with open(class_name_file, "wb") as f:
                f.write(response.content)
            st.write("Classes downloaded successfully!")
            with open(class_name_file, "rb") as f:
                class_name = pickle.load(f)
        else:
            st.write("Failed to download class names.")
            return

    if st.button("Predict"):
        if model is not None and class_name is not None:
            predicted_class, confidence = prediction(picture, model, class_name)
            st.success(f"This is {predicted_class} and the model is {confidence} confident.")

if __name__ == "__main__":
    main()
