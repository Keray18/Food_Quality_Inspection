import numpy as np
import os
import pathlib
import pickle
import uvicorn
import requests

from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from predict_pipeline import prediction

app = FastAPI()

model = None
current_dir = os.path.dirname(os.path.abspath(__file__))
class_names_file = os.path.join(
    current_dir, "notebook", "class_name.pkl")
# model_dir = os.path.join(current_dir, "..", "testing_model", "model.h5")

URL = "https://drive.usercontent.google.com/u/0/uc?id=1ofu7smGB7D2rwce_-1Tiz4tk8Wfwjpqn&export=download"
model_file = os.path.join(current_dir, "model.h5")


@app.get("/")
def home():
    return "X marks the spot."


@app.post("/preds")
async def predict_image(file: UploadFile = File(...)):
    if os.path.exists(class_names_file):
        with open(class_names_file, "rb") as f:
            class_name = pickle.load(f)

    global model
    if model is None:
        response = requests.get(URL)
        if response.status_code == 200:
            with open(model_file, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully!")  
        else:
            print("Failed to download model.") 
            return {"error": "Model not available"}

        model = tf.keras.models.load_model(model_file)

    bytes = await file.read()
    predicted_class, confidence = prediction(bytes, model, class_name)
    return {"prediction": predicted_class}


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
