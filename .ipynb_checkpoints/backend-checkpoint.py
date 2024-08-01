# ------------------------------------------------------- 
# Requirements
# ------------------------------------------------------- 
import uvicorn
from fastapi import FastAPI, HTTPExeption
#from tensorflow.keras.models import load_model
import Forêt_Aleatoire
import numpy as np
import io
#from PIL import Image
import joblib
from pydantic import BaseModel
from pathlib import Path
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


# ------------------------------------------------------- 
# First route
# ------------------------------------------------------- 
'''@app.get("/")
def api_info():
    return {"info": "Welcome carapuce"}'''

# ------------------------------------------------------- 
# Second route
# ------------------------------------------------------- 

'''class CustomerData(BaseModel):
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
    TechSupport_Yes: int'''

# load model

def load_model(model_path):
    return joblib.load(model_path)


'''@app.post("/predict")
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
    uvicorn.run("backend:app", host='127.0.0.1', port=10000, log_level='info', reload=True)'''
def preprocess(data: CustomerData):
    # Exemple de prétraitement - ajustez-le en fonction des besoins de votre modèle
    feature_order = [
        'tenure', 'MultipleLines_Yes', 'InternetService_Fiber', 'OnlineSecurity_Yes', 'gender' 
        'DeviceProtection_No', 'TechSupport_Yes', 'StreamingTV_No', 
        'Contract_Two', 'Contract_One', 'PaperlessBilling', 'PaymentMethod_Electronic', 'MonthlyCharges', 'TotalCharges'
    ]

    categorical_features = [
        'MultipleLines_Yes', 'InternetService_Fiber',
        'OnlineSecurity_Yes', 'DeviceProtection_No', 'TechSupport_Yes',
        'PaymentMethod_Electronic', 'Contract_One', 'Contract_Two'
    ]

    # Convertir l'objet Pydantic en dictionnaire
    data_dict = data.dict()

    # Encodeur fictif (à remplacer par votre encodeur réel)
    encoded_features = []
    for feature in feature_order:
        if feature in categorical_features:
            # Encoder la caractéristique catégorielle (exemple de transformation)
            encoded_feature = encode_categorical_feature(feature, data_dict[feature])
            encoded_features.extend(encoded_feature)
        else:
            encoded_features.append(data_dict[feature])
    
    return np.array(encoded_features).reshape(1, -1)

def encode_categorical_feature(feature, value):
    # Exemple d'encodage pour une caractéristique catégorielle
    # Remplacez-le par votre méthode d'encodage réelle (par exemple, One-Hot Encoding)
    encoding_map = {
        'Yes': 1,
        'No': 0,
        # Ajoutez ici toutes les valeurs possibles pour chaque caractéristique catégorielle
    }
    return [encoding_map.get(value, -1)]  # Retourne -1 si la valeur n'est pas trouvée (à ajuster)

# ------------------------------------------------------- 
# Load the model on app setup
# ------------------------------------------------------- 
model_path = Path(__file__).parent / "Forêt_Aleatoire.pkl"
model = load_model(model_path)
# ------------------------------------------------------- 
# First route
# ------------------------------------------------------- 
@app.get("/")
def api_info():
    return {"info": "Welcome to the churn prediction API"}

# ------------------------------------------------------- 
# Second route
# ------------------------------------------------------- 
@app.post("/predict")
async def predict(data: CustomerData):
    try:
        # Prétraiter les données
        processed_data = preprocess(data)
        
        # Faire la prédiction
        prediction = model.predict(processed_data)
        
        # Retourner la prédiction
        return {
            "Churn": "Yes" if prediction[0] else "No"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))