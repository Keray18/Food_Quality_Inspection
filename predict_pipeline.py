import os
import pathlib
import numpy as np
from PIL import Image
from io import BytesIO
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing import image


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
