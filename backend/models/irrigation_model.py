import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

DATA_PATH = os.path.join("datasets", "irrigation.csv")
MODEL_PATH = "irrigation_model.pkl"

data = pd.read_csv(DATA_PATH)

X = data[["soil_moisture", "temperature"]]
y = data["water"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print("✅ Irrigation model trained successfully")
print(f"📄 Model saved as: {MODEL_PATH}")
print(f"✅ Rows used: {len(data)}")
