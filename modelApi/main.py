import numpy as np 
import os
import pathlib
import pickle
import uvicorn

from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from predict_pipeline import prediction

app = FastAPI()


current_dir = os.path.dirname(os.path.abspath(__file__))
class_names_file = os.path.join(current_dir, "..", "notebook", "class_name.pkl")
model_dir = os.path.join(current_dir, "..", "testing_model", "model.h5")


@app.post("/preds")
async def predict_image(file: UploadFile = File(...)):
    if os.path.exists(class_names_file):
        with open(class_names_file, "rb") as f:
            class_name = pickle.load(f)

    bytes = await file.read()
    model = tf.keras.models.load_model(model_dir)

    predicted_class, confidence = prediction(bytes, model, class_name)
    return {"prediction": predicted_class}



if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)