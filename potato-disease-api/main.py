from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Potato Disease API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((256, 256))
    image = np.array(image)

    # Dummy prediction (for now)
    return {"prediction": "Potato___Healthy"}