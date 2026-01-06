from fastapi import APIRouter
import joblib
import pandas as pd
import requests
import os

router = APIRouter()

model = joblib.load("irrigation_model.pkl")

WEATHER_API_KEY = "8877041cd9715b9fc45e993e8b0e984c"

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

@router.post("/predict-irrigation")
def predict_irrigation(soil_moisture: float, city: str):
    
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    weather = requests.get(WEATHER_URL, params=params).json()

    temperature = weather["main"]["temp"]
    humidity = weather["main"]["humidity"]
    condition = weather["weather"][0]["description"]

    df = pd.DataFrame([[soil_moisture, temperature]],
                      columns=["soil_moisture", "temperature"])
    water = model.predict(df)[0]

    reasons = []

    if "rain" in condition.lower():
        return {
            "recommended_water_mm": 0,
            "reason": "Rain detected in weather forecast",
            "weather": condition
        }

    if temperature <= 18:
        water *= 0.7
        reasons.append("Low temperature")

    if humidity >= 60:
        water *= 0.8
        reasons.append("High humidity")

    if temperature >= 35 and humidity < 40:
        water *= 1.2
        reasons.append("Hot and dry conditions")

    water = round(water, 2)

    return {
        "recommended_water_mm": water,
        "city": city,
        "temperature": temperature,
        "humidity": humidity,
        "weather": condition,
        "reason": ", ".join(reasons) if reasons else "Optimal conditions"
    }
