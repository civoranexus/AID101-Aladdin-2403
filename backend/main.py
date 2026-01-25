from fastapi import FastAPI
from routes import yield_routes, irrigation, weather
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AgriAssist AI Backend Running ✅"}

app.include_router(yield_routes.router)
app.include_router(irrigation.router)
app.include_router(weather.router)