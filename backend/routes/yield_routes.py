from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
import os

router = APIRouter(tags=["Yield Prediction"])

# Load trained ML model
model = joblib.load("yield_model.pkl")

CROP_MAPPING = {
    "wheat": 0,
    "rice": 1,
    "maize": 2
}

# Weather API config
WEATHER_API_KEY = "8877041cd9715b9fc45e993e8b0e984c"

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

class YieldResponse(BaseModel):
    predicted_yield: float
    crop: str
    city: str
    temperature: float
    rainfall: float
    fertilizer: float

@router.post(
    "/predict-yield",
    response_model=YieldResponse,
    summary="Predict crop yield using weather data",
    description="""
    Predicts crop yield by combining user inputs with live weather data.

    Inputs:
    - Rainfall (mm)
    - Fertilizer usage (kg per acre)
    - City name

    Weather:
    - Temperature is fetched automatically using OpenWeatherMap API.
    """
)
def predict_yield(
    crop: str = Query(
        ...,
        description="Crop type (wheat, rice, maize)",
        examples={"default": {"value": "wheat"}}
    ),
    rainfall: float = Query(
        ...,
        description="Total rainfall during the crop season (in mm)",
        examples={"normal": {"value": 120}}
    ),
    fertilizer: float = Query(
        ...,
        description="Amount of fertilizer applied (kg per acre)",
        examples={"standard": {"value": 50}}
    ),
    city: str = Query(
        ...,
        description="City name for fetching live weather",
        examples={"example": {"value": "Delhi"}}
    )
):
    
    crop = crop.lower()

    if crop not in CROP_MAPPING:
        raise HTTPException(
            status_code=400,
            detail="Invalid crop. Choose from wheat, rice, or maize."
        )

    crop_code = CROP_MAPPING[crop]


    # 1️⃣ Fetch live weather
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    weather = requests.get(WEATHER_URL, params=params).json()

    if "main" not in weather:
        raise HTTPException(status_code=400, detail="Weather data not available")

    temperature = weather["main"]["temp"]

    # 2️⃣ ML prediction
    df = pd.DataFrame(
    [[rainfall, temperature, fertilizer, crop_code]],
    columns=["rainfall", "temperature", "fertilizer", "crop"])
    prediction = model.predict(df)[0]

    return {
        "predicted_yield": round(prediction, 2),
        "crop": crop,
        "city": city,
        "temperature": temperature,
        "rainfall": rainfall,
        "fertilizer": fertilizer
    }
