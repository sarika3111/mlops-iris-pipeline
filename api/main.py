from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("C:/Users/6134155/Assignment-MLOPS/mlops-iris-pipeline/models/model.pkl")  # Replace with your best model path

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Iris Classification API is running successfully!"}

@app.post("/predict")
def predict_species(data: IrisInput):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(features)
    return {"predicted_class": int(prediction[0])}
