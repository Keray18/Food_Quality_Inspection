import os
import pathlib
import numpy as np  
from PIL import Image 
from io import BytesIO 
import pickle
import dvc.api

import tensorflow as tf
from tensorflow.keras.preprocessing import image


current_dir = os.path.dirname(os.path.abspath(__file__))
class_names_file = os.path.join(current_dir, "..", "notebook", "class_name.pkl")
# model_dir = os.path.join(current_dir, "..", "model", "2")



def preprocessing(filename) -> np.ndarray:
    img = Image.open(BytesIO(filename))
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    return img_array


def prediction(filename):
    img_array = preprocessing(filename)
    img_array = np.expand_dims(img_array, axis=0)

    dagshub_model_path = "trial_model/model.h5"

    dagshub_repo = "https://dagshub.com/Keray18/Food_Quality_Inspection"

    with dvc.api.open(f"{dagshub_model_path}", repo=dagshub_repo, mode="rb") as f:
        model = tf.keras.models.load_model(f)

    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])

    if os.path.exists(class_names_file):
        with open(class_names_file, "rb") as f:
            class_name = pickle.load(f)

    predicted_class = class_name[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class, confidence
