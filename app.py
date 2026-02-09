from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

app = FastAPI()

# 1. Define the Input Format
class OpportunityInput(BaseModel):
    amount: float
    stage: str
    lead_source: str
    type: str
    close_days: int

# 2. Define the Output Format (THIS IS WHAT WAS MISSING!)
class PredictionOutput(BaseModel):
    win_probability: float

@app.get("/")
def home():
    return {"status": "running"}

# 3. Add response_model=PredictionOutput here
@app.post("/predict", response_model=PredictionOutput)
def predict(data: OpportunityInput):
    # Create dataframe
    input_df = pd.DataFrame([{
        'amount': data.amount,
        'stage': data.stage,
        'lead_source': data.lead_source,
        'type': data.type,
        'close_days': data.close_days
    }])
    
    # One-hot encode
    input_df = pd.get_dummies(input_df, columns=['stage', 'lead_source', 'type'])
    
    # Add missing columns
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns
    input_df = input_df[features]
    
    # Scale and predict
    input_scaled = scaler.transform(input_df)
    probability = float(model.predict_proba(input_scaled)[0][1])
    
    # Return matched format
    return {"win_probability": round(probability, 4)}