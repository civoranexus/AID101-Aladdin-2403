import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("datasets/irrigation.csv")
X = data[["soil_moisture", "temperature"]]
y = data["water"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "irrigation_model.pkl")
