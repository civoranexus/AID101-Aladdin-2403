import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

data = pd.read_csv("datasets/crop_yield.csv")

X = data[["rainfall", "temperature", "fertilizer", "crop"]]
y = data["yield"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "yield_model.pkl")
print("Yield model retrained successfully")
