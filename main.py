# Librairies
from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
# Chargement de données
data = pd.read_csv('churn_predictor.csv')

# chargement ddu modèle
loaded_model = load('Forêt Aleatoire')

# Création d'une nouvelle instance FastAPI
app = FastAPI()

# Définir un objet (une classe) pour réaliser des requêtes
# dot nutation (.)
class reuest(BaseModel):
    train_features: float
    train_labels: float


# Définition du point de terminaison
@app.post('/predict') # local: http:/127.0.0.1:8890/predict
# Définition de la fonction de prediction
