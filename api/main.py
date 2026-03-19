from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL = load_model(
    "../potatoes.h5",
    custom_objects={"RandomFlip": RandomFlip}
)

class_names = ["Early Blight", "Late Blight", "Healthy"]

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    input_array = preprocess_image(image)

    prediction = MODEL.predict(input_array)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": confidence
    }