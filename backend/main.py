from fastapi import FastAPI
from routes import yield_routes, irrigation, weather

app = FastAPI()

app.include_router(yield_routes.router)
app.include_router(irrigation.router)
app.include_router(weather.router)
