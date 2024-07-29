# ------------------------------------------------------- 
# Requirements
# ------------------------------------------------------- 
import uvicorn
from fastapi import FastAPI
#from tensorflow.keras.models import load_model
import Forêt Aleatoire
import numpy as np
import io
#from PIL import Image
import pickle
from pydantic import BaseModel
# ------------------------------------------------------- 
# App
# ------------------------------------------------------- 
app = FastAPI()

# ------------------------------------------------------- 
# Utils
# ------------------------------------------------------- 

#def load():
   # model_path = "3-projet/Forêt Aleatoire.pkl"
    #model = joblib.load(model_path)
   # return model
# 
# model = joblib.load('Forêt Aleatoire.pkl')
# ------------------------------------------------------- 
# Load the model on app setup
# ------------------------------------------------------- 
#model = load()
# Chargement du modèle
pickle_in = open("Forêt Aleatoire.pkl", "rb")
model = pickle.load(open(pickle_in)

# ------------------------------------------------------- 
# First route
# ------------------------------------------------------- 
@app.get("/")
def api_info():
    return {"info": "Welcome carapuce"}

# ------------------------------------------------------- 
# Second route
# ------------------------------------------------------- 

class CustomerData(BaseModel):
    gender: int
    tenure: int
    MultipleLines_yes: int
    InternetService_Fiber: int
    OnlineSecurity_Yes: int
    DeviceProtection_No: int
    StreamingTV_No: int
    Contract_Two: int
    Contract_One: int
    PaperlessBilling: int
    PaymentMethod_Electronic: int
    MonthlyCharges: float
    TotalCharges: float
    TechSupport_Yes: int

@app.post("/predict")
def predict(data: CustomerData):
    # Conversion des données en tableau numpy
    data_array = np.array([[data.TotalCharges, data.tenure, data.MonthlyCharges,
       data.PaymentMethod_Electronic, data.InternetService_Fiber,
       data.Contract_Two, data.gender, data.OnlineSecurity_Yes, data.PaperlessBilling,
       data.TechSupport_Yes, data.Contract_One, data.MultipleLines_Yes,
       data.StreamingTV_No,
       frame_two.DeviceProtection_No]])

    # Application  du scaler
    data_scaled = scaler.transform(data_array)
    
    # Prédiction
    prediction = model.predict(data_scaled)
    
    return {"Churn": "Yes" if prediction[0] == 1 else "No"}

if __name__ =="__backend__":
    uvicorn.run("backend:app", host='127.0.0.1', port=10000, log_level='info', reload=True)