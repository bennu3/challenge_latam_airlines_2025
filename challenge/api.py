import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import fastapi
import pandas as pd
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from challenge.model import DelayModel

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"


class DelayService:
    """
    Encapsulates training and inference logic for the delay model.
    Ensures the model is trained at most once and reused for every request.
    """

    def __init__(self, dataset_path: Path):
        self._dataset_path = dataset_path
        self._model = DelayModel()
        self._is_trained = False
        self.logger = logging.getLogger(__name__)

    def _load_dataset(self) -> pd.DataFrame:
        dtype = {"Vlo-I": "string", "Vlo-O": "string"}
        return pd.read_csv(self._dataset_path, dtype=dtype, low_memory=False)

    def ensure_model(self) -> DelayModel:
        if not self._is_trained:
            try:
                data = self._load_dataset()
                features, target = self._model.preprocess(data, target_column="delay")
                self._model.fit(features, target)
                self._is_trained = True
                self.logger.info("Delay model entrenado")
            except:
                self.logger.exception("Error al entrenar el modelo Delay")
                raise
        return self._model

    def predict(self, inference_data: pd.DataFrame) -> List[int]:
        model = self.ensure_model()
        features = model.preprocess(inference_data)
        return model.predict(features)


def _get_delay_service(app: fastapi.FastAPI) -> DelayService:
    service = getattr(app.state, "delay_service", None)
    if service is None:
        service = DelayService(DATASET_PATH)
        app.state.delay_service = service
    return service


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """
    Lifespan hook that preloads the model once when the app boots.
    Tests that skip the lifecycle still rely on DelayService.ensure_model().
    """
    service = DelayService(DATASET_PATH)
    try:
        service.ensure_model()
    except Exception as exc:
        print(f"Failed to preload delay model: {exc}")
    app.state.delay_service = service
    yield


app = fastapi.FastAPI(lifespan=lifespan)


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("MES")
    def validate_mes(cls, number):
        if number < 1 or number > 12:
            raise ValueError("MES debe ser entre 1 y 12")
        return number

    @validator("TIPOVUELO")
    def validate_tipovuelo(cls, type):
        if type not in ["N", "I"]:
            raise ValueError("TIPOVUELO debe ser N o I")
        return type

    @validator("OPERA")
    def validate_opera(cls, operator):
        valid_airlines = [
            "Aerolineas Argentinas",
            "Aeromexico",
            "Air Canada",
            "Air France",
            "Alitalia",
            "American Airlines",
            "Austral",
            "Avianca",
            "British Airways",
            "Copa Air",
            "Delta Air",
            "Gol Trans",
            "Grupo LATAM",
            "Iberia",
            "JetSmart SPA",
            "K.L.M.",
            "Lacsa",
            "Latin American Wings",
            "Oceanair Linhas Aereas",
            "Plus Ultra Lineas Aereas",
            "Qantas Airways",
            "Sky Airline",
            "United Airlines",
        ]
        if operator not in valid_airlines:
            raise ValueError(
                f"OPERA debe ser alguno de los siguientes {','.join(valid_airlines)}"
            )
        return operator


class FlightRequest(BaseModel):
    flights: List[Flight]

    @validator("flights")
    def validate_flights(cls, flights):
        if not flights:
            raise ValueError("Debe proporcionar al menos un vuelo")
        return flights


class PredictionResponse(BaseModel):
    predict: List[int]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    Ensure FastAPI surfaces validation errors as HTTP 400 to match the spec/tests.
    """
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.post("/predict", status_code=200)
async def post_predict(
    request: FlightRequest, http_request: Request
) -> PredictionResponse:
    """
    Predecir delays para una lista de vuelos.

    Args:
        request: FlightRequest con lista de vuelos

    Returns:
        PredictionResponse con lista de predicciones
    """

    try:
        delay_service = _get_delay_service(http_request.app)
        flights_data = [flight.dict() for flight in request.flights]
        df = pd.DataFrame(flights_data)
        predictions = delay_service.predict(df)

        return PredictionResponse(predict=predictions)
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))
