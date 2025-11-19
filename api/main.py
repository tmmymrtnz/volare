from __future__ import annotations

from datetime import date
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from volare_model.serving import LocalModelService

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model_pipeline.joblib"

app = FastAPI(title="Flight Fare Estimator", version="0.1.0")
service = LocalModelService(MODEL_PATH)


class PredictionRequest(BaseModel):
    Date_of_journey: date = Field(..., description="Fecha del vuelo en formato YYYY-MM-DD")
    Journey_day: str
    Airline: str
    Flight_code: str | None = None
    Class: str
    Source: str
    Departure: str
    Total_stops: str
    Arrival: str
    Destination: str
    Duration_in_hours: float
    Days_left: int

    def to_payload(self) -> dict:
        data = self.model_dump()
        data["Date_of_journey"] = self.Date_of_journey.isoformat()
        return data


class PredictionResponse(BaseModel):
    fare: float
    model_version: str
    latency_ms: float


@app.get("/health")
def health_check():
    status = "ok"
    message = None
    try:
        _ = service.load()
    except FileNotFoundError as err:
        status = "error"
        message = str(err)
    return {
        "status": status,
        "model_path": str(MODEL_PATH),
        "message": message,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start = perf_counter()
    try:
        fare = service.predict(request.to_payload())
    except FileNotFoundError as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
    latency_ms = (perf_counter() - start) * 1000
    return PredictionResponse(
        fare=fare,
        model_version=service.model_path.name,
        latency_ms=latency_ms,
    )

