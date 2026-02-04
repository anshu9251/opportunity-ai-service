from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# -----------------------------
# Load frozen ML artifacts
# -----------------------------
model = joblib.load("opportunity_win_lr_model.pkl")
scaler = joblib.load("opportunity_feature_scaler.pkl")
model_features = joblib.load("opportunity_model_features.pkl")

app = FastAPI(title="Opportunity Win Probability API")

# -----------------------------
# Input schema (Salesforce payload)
# -----------------------------
class OpportunityInput(BaseModel):
    amount: float
    stage: str
    lead_source: str
    industry: str
    close_days: int
    account_size: str

# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "OK"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
@app.post("/predict")
def predict_win_probability(data: OpportunityInput):

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # One-hot encode
        input_encoded = pd.get_dummies(input_df)

        # Align columns with training
        input_encoded = input_encoded.reindex(
            columns=model_features,
            fill_value=0
        )

        # Scale numeric features SAFELY
        numeric_cols = ["amount", "close_days"]
        input_encoded[numeric_cols] = scaler.transform(
            input_encoded[numeric_cols].values
        )

        # Predict probability
        win_prob = model.predict_proba(input_encoded)[0][1]

        return {
            "win_probability": round(float(win_prob), 4)
        }

    except Exception as e:
        return {
            "error": str(e)
        }
