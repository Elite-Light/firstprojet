# ------------------------------------------------------- 
# Requirements
# ------------------------------------------------------- 
from fastapi import FastAPI, UploadFile
#from tensorflow.keras.models import load_model
import numpy as np
import io
#from PIL import Image

# ------------------------------------------------------- 
# App
# ------------------------------------------------------- 
app = FastAPI()

# ------------------------------------------------------- 
# Utils
# ------------------------------------------------------- 

def load():
    model_path = "3-projet/Forêt Aleatoire.pkl"
    model = load_model(model_path)
    return model

# ------------------------------------------------------- 
# Load the model on app setup
# ------------------------------------------------------- 
model = load()

# ------------------------------------------------------- 
# First route
# ------------------------------------------------------- 
@app.get("/")
def api_info():
    return {"info": "Welcome carapuce"}

# ------------------------------------------------------- 
# Second route
# ------------------------------------------------------- 
@app.post("/predict")
async def predict(file: UploadFile):
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data))
    img_processed = preprocess(img)
    predictions = model.predict(img_processed)
    print(predictions)
    proba = float(predictions[0][0])
    return {
        "cat_proba": 1 - proba,
        "dog_proba": proba,
        "predict_class": "dog" if proba > 0.5 else "cat"
    }