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
    "maize": 2,
    "sugarcane": 3,
    "cotton": 4,
    "potato": 5,
    "onion": 6,
    "tomato": 7,
    "mustard": 8,
    "groundnut": 9,
    "soybean": 10,
    "sunflower": 11,
    "chickpea": 12,
    "pigeonpea": 13,
    "lentil": 14,
    "sorghum": 15,
    "pearlmillet": 16,
    "barley": 17,
    "jute": 18,
    "tea": 19,
    "coffee": 20,
    "banana": 21,
    "mango": 22,
    "orange": 23,
    "apple": 24,
    "grapes": 25,
    "papaya": 26,
    "pomegranate": 27,
    "guava": 28,
    "watermelon": 29,
    "muskmelon": 30,
    "pineapple": 31,
    "coconut": 32,
    "cashew": 33,
    "arecanut": 34,
    "rubber": 35,
    "cocoa": 36,
    "brinjal": 37,
    "cauliflower": 38,
    "cabbage": 39,
    "carrot": 40,
    "radish": 41,
    "peas": 42,
    "beans": 43,
    "okra": 44,
    "spinach": 45,
    "capsicum": 46,
    "cucumber": 47,
    "pumpkin": 48,
    "bottle_gourd": 49,
    "bitter_gourd": 50,
    "ridge_gourd": 51,
    "garlic": 52,
    "ginger": 53,
    "turmeric": 54,
    "chilli": 55,
    "coriander": 56,
    "cumin": 57,
    "fennel": 58,
    "fenugreek": 59,
    "cardamom": 60,
    "clove": 61,
    "black_pepper": 62,
    "cinnamon": 63,
    "bay_leaf": 64,
    "safflower": 65,
    "sesame": 66,
    "castor": 67,
    "linseed": 68,
    "niger": 69,
    "moong": 70,
    "urad": 71,
    "masoor": 72,
    "rajma": 73,
    "cowpea": 74,
    "mothbean": 75,
    "horsegram": 76,
    "kidneybean": 77,
    "tobacco": 78,
    "bajra": 79,
    "jowar": 80,
    "ragi": 81,
    "foxtail_millet": 82,
    "little_millet": 83,
    "kodo_millet": 84,
    "barnyard_millet": 85,
    "proso_millet": 86,
    "amaranth": 87,
    "sugarbeet": 88,
    "sweet_potato": 89,
    "tapioca": 90,
    "jackfruit": 91,
    "litchi": 92,
    "custard_apple": 93,
    "fig": 94,
    "date_palm": 95,
    "jamun": 96,
    "kiwi": 97,
    "strawberry": 98,
    "dragon_fruit": 99
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
            detail="Invalid crop name"
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
