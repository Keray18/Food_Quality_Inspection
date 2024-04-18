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



@app.post("/preds")
async def predict_image(file: UploadFile = File(...)):
    bytes = await file.read()
    predicted_class, confidence = prediction(bytes)
    return {"prediction": predicted_class}

    




if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)