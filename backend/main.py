"""Serve the notebook's frozen pipeline and exported historical summaries."""
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

ARTIFACTS = Path(os.getenv('ARTIFACT_DIR', Path(__file__).parent / 'artifacts'))
bundle = {}
load_error = None


@asynccontextmanager
async def lifespan(app):
    """Load trusted, locally exported artifacts once; never fit any estimator."""
    global load_error
    try:
        bundle['model'] = joblib.load(ARTIFACTS / 'traffic_model.joblib')
        for name in ['overview', 'analytics', 'metrics']:
            bundle[name] = json.loads((ARTIFACTS / f'{name}.json').read_text(encoding='utf-8'))
        load_error = None
    except Exception as error:
        bundle.clear()
        load_error = f'Model artifacts unavailable. Execute the notebook first. {type(error).__name__}'
    yield


app = FastAPI(title='TrafficRush API', version='1.0.0', lifespan=lifespan)
origins = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173,https://doctordictator.github.io')
app.add_middleware(CORSMiddleware, allow_origins=origins.split(','),
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


def require_bundle():
    if not bundle:
        raise HTTPException(503, detail=load_error or 'Model is not loaded.')
    return bundle


class TrafficInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    date_time: datetime
    is_holiday: bool = False
    temp_c: float = Field(ge=-60, le=60)
    rain_1h: float = Field(ge=0, le=1000)
    snow_1h: float = Field(ge=0, le=1000)
    clouds_all: float = Field(ge=0, le=100)
    weather_main: str = Field(min_length=1, max_length=40)

    @field_validator('date_time')
    @classmethod
    def local_hour(cls, value):
        # The source provides local wall-clock hours; don't silently shift zones.
        if value.tzinfo is not None or value.minute or value.second or value.microsecond:
            raise ValueError('Use a local date/time at the start of an hour, without a timezone offset.')
        if not 2012 <= value.year <= 2100:
            raise ValueError('Choose a year from 2012 to 2100.')
        return value


@app.get('/api/health')
def health():
    return {'status': 'ready' if bundle else 'unavailable', 'model_loaded': bool(bundle), 'detail': load_error}


@app.get('/api/overview')
def overview():
    return require_bundle()['overview']


@app.get('/api/analytics')
def analytics():
    return require_bundle()['analytics']


@app.get('/api/model/metrics')
def metrics():
    return require_bundle()['metrics']


@app.post('/api/predict')
def predict(inputs: TrafficInput):
    data = require_bundle()
    if inputs.weather_main not in data['metrics']['weather_categories']:
        raise HTTPException(422, detail='Choose one of the dataset weather categories.')
    # Derive only calendar fields here. Learned imputation, encoding and scaling
    # remain inside the exact pipeline fitted and exported by the notebook.
    row = dict(temp_c=inputs.temp_c, rain_1h=inputs.rain_1h, snow_1h=inputs.snow_1h,
        clouds_all=inputs.clouds_all, weather_main=inputs.weather_main,
        hour=inputs.date_time.hour, day_of_week=inputs.date_time.weekday(),
        month=inputs.date_time.month, is_holiday=int(inputs.is_holiday))
    frame = pd.DataFrame([row])
    model = data['model']
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, list(model.classes_).index(1)])
    return {'prediction': prediction, 'category': 'High Traffic' if prediction else 'Normal Traffic',
        'high_probability': probability, 'normal_probability': 1-probability,
        'volume_threshold': data['metrics']['volume_threshold'], 'decision_threshold': .5,
        'model': data['metrics']['model_name'], 'inputs': inputs.model_dump(mode='json'),
        'derived': {key: row[key] for key in ['hour', 'day_of_week', 'month']},
        'scope': 'Historical volume estimate; not live traffic, congestion or crash probability.',
        'extrapolation': inputs.date_time.year > 2018}
