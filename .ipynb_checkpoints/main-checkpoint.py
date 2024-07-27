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