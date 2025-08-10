import os

import joblib

from fastapi import FastAPI
import logging
from pydantic import BaseModel

logging.basicConfig(filename='C:/Users/Harish Kumar/Downloads/mlops-iris-pipeline/api/logs/api.log', level=logging.INFO)

app = FastAPI()
 
# --- Load the model ---

# This works both locally and inside Docker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
 
if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
 
model = joblib.load(MODEL_PATH)
 
# --- Input Schema ---

class InputData(BaseModel):

    sepal_length: float

    sepal_width: float

    petal_length: float

    petal_width: float
 
# --- Routes ---

@app.get("/")

def root():

    return {"message": "Iris Prediction API is running!"}
 
@app.post("/predict")

def predict(input_data: InputData):

    features = [[

        input_data.sepal_length,

        input_data.sepal_width,

        input_data.petal_length,

        input_data.petal_width

    ]]

    prediction = model.predict(features)[0]

    # Log request and prediction
    logging.info(f"Request: {input_data.model_dump()}, Prediction: {prediction}")

    return {"prediction": str(prediction)}

