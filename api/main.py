from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = FastAPI()

MODEL = None  # load later

def load_my_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model("potatoes.h5", custom_objects={"RandomFlip": RandomFlip})

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    load_my_model()  # load when needed
    
    image = Image.open(file.file).convert("RGB")
    input_array = preprocess_image(image)

    prediction = MODEL.predict(input_array)
    predicted_index = int(np.argmax(prediction))

    class_names = ["Early Blight", "Late Blight", "Healthy"]

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": float(np.max(prediction))
    }