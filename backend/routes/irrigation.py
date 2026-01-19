from fastapi import APIRouter
import joblib
import pandas as pd
import requests
import os

router = APIRouter()

# Load trained ML model
model = joblib.load("irrigation_model.pkl")

CROP_WATER_FACTOR = {crop: 1.0 for crop in [
    "wheat","rice","maize","sugarcane","cotton","potato","onion","tomato","mustard",
    "groundnut","soybean","sunflower","chickpea","pigeonpea","lentil","sorghum",
    "pearlmillet","barley","jute","tea","coffee","banana","mango","orange","apple",
    "grapes","papaya","pomegranate","guava","watermelon","muskmelon","pineapple",
    "coconut","cashew","arecanut","rubber","cocoa","brinjal","cauliflower","cabbage",
    "carrot","radish","peas","beans","okra","spinach","capsicum","cucumber","pumpkin",
    "bottle_gourd","bitter_gourd","ridge_gourd","garlic","ginger","turmeric","chilli",
    "coriander","cumin","fennel","fenugreek","cardamom","clove","black_pepper",
    "cinnamon","bay_leaf","safflower","sesame","castor","linseed","niger","moong",
    "urad","masoor","rajma","cowpea","mothbean","horsegram","kidneybean","tobacco",
    "bajra","jowar","ragi","foxtail_millet","little_millet","kodo_millet","barnyard_millet",
    "proso_millet","amaranth","sugarbeet","sweet_potato","tapioca","jackfruit","litchi",
    "custard_apple","fig","date_palm","jamun","kiwi","strawberry","dragon_fruit"
]}

# Higher water requirement crops
CROP_WATER_FACTOR["rice"] = 1.3
CROP_WATER_FACTOR["sugarcane"] = 1.4
CROP_WATER_FACTOR["banana"] = 1.2
CROP_WATER_FACTOR["jute"] = 1.2

# Lower water requirement crops
CROP_WATER_FACTOR["chickpea"] = 0.85
CROP_WATER_FACTOR["lentil"] = 0.85
CROP_WATER_FACTOR["pearlmillet"] = 0.85
CROP_WATER_FACTOR["bajra"] = 0.85

# Weather API config
WEATHER_API_KEY = "8877041cd9715b9fc45e993e8b0e984c"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

@router.post("/predict-irrigation")
def predict_irrigation(
    soil_moisture: float,
    crop: str,
    city: str
    ):

    crop = crop.lower()
    if crop not in CROP_WATER_FACTOR:
        return {"error": "Invalid crop name"}

    # 1️⃣ Fetch live weather
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    weather = requests.get(WEATHER_URL, params=params).json()

    if "main" not in weather:
        return {"error": "Weather data not available", "api_response": weather}

    temperature = weather["main"]["temp"]
    humidity = weather["main"]["humidity"]
    condition = weather["weather"][0]["description"]

    # 2️⃣ ML prediction
    df = pd.DataFrame([[soil_moisture, temperature]],
                      columns=["soil_moisture", "temperature"])
    water = model.predict(df)[0]
    water *= CROP_WATER_FACTOR.get(crop, 1.0)

    # 3️⃣ Rule-based adjustments
    reasons = []

    if "rain" in condition.lower():
        return {
            "recommended_water_mm": 0,
            "crop": crop,
            "city": city,
            "temperature": temperature,
            "humidity": humidity,
            "weather": condition,
            "reason": "Rain detected"
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
        "crop": crop,
        "city": city,
        "temperature": temperature,
        "humidity": humidity,
        "weather": condition,
        "reason": ", ".join(reasons) if reasons else "Optimal conditions"
    }
