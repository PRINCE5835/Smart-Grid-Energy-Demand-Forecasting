import pickle
import os
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'features.pkl')

app = FastAPI(title="Smart Grid Energy Demand Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(FEATURES_PATH, 'rb') as f:
    feature_columns = pickle.load(f)

print(f"[INFO] Model loaded. Features: {feature_columns}")

class PredictResponse(BaseModel):
    datetime: str
    predicted_load_mw: float
    features_used: dict

@app.get("/")
def root():
    return {"message": "Smart Grid Energy Demand Forecasting API", "status": "running"}

@app.get("/predict", response_model=PredictResponse)
def predict(dt: str = Query(..., description="Datetime in format YYYY-MM-DD HH:MM:SS")):
    try:
        dt_parsed = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": "Invalid datetime format. Use YYYY-MM-DD HH:MM:SS"}

    hour = dt_parsed.hour
    day_of_week = dt_parsed.weekday()
    month = dt_parsed.month

    input_df = pd.DataFrame([[hour, day_of_week, month]], columns=feature_columns)
    prediction = model.predict(input_df)[0]

    return PredictResponse(
        datetime=dt,
        predicted_load_mw=round(prediction, 2),
        features_used={"hour": hour, "day_of_week": day_of_week, "month": month}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
