from contextlib import asynccontextmanager
from pathlib import Path
import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

ARTIFACTS = Path(os.getenv('ARTIFACT_DIR', Path(__file__).parent / 'artifacts'))
bundle = {}
load_error = None

@asynccontextmanager
async def lifespan(app):
    global load_error
    bundle.clear()
    try:
        loaded = {name: json.loads((ARTIFACTS / f'{name}.json').read_text(encoding='utf-8')) for name in ['info','metrics','analytics']}
        loaded['model'] = joblib.load(ARTIFACTS / 'nearmiss.joblib')
        if list(loaded['model'].feature_names_in_) != loaded['info']['features']:
            raise ValueError('Feature schema mismatch')
        bundle.update(loaded)
        load_error = None
    except Exception as exc:
        load_error = f'Model unavailable ({type(exc).__name__}). Execute notebooks/NearMiss.ipynb.'
    yield

app = FastAPI(title='NearMiss API', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in os.getenv('CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173').split(',') if x.strip()], allow_methods=['GET','POST'], allow_headers=['Content-Type'])

@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Omit raw inputs/context: nonfinite numbers in them cannot be JSON serialized.
    errors = [{key: error[key] for key in ['loc','msg','type']} for error in exc.errors()]
    return JSONResponse(status_code=422, content={'detail': errors})

def ready():
    if not bundle: raise HTTPException(503, detail=load_error or 'Model unavailable')
    return bundle

class Pair(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    follower_speed: float = Field(ge=.5, le=55)
    leader_speed: float = Field(ge=0, le=55)
    follower_accel: float = Field(ge=-12, le=8)
    leader_accel: float = Field(ge=-12, le=8)
    spacing_m: float = Field(gt=0, le=2000)
    leader_length: float = Field(ge=1, le=25)

    @model_validator(mode='after')
    def separated_bodies(self):
        if self.spacing_m <= self.leader_length:
            raise ValueError('Front spacing must exceed leader length (positive bumper gap).')
        return self

class Item(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=50)
    inputs: Pair

class Batch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    pairs: list[Item] = Field(min_length=1, max_length=30)

    @model_validator(mode='after')
    def unique_ids(self):
        if len({p.id for p in self.pairs}) != len(self.pairs): raise ValueError('Pair IDs must be unique.')
        return self

def support(row, info):
    return [key for key, value in row.items() if not info['ranges'][key][0] <= value <= info['ranges'][key][1]]

def infer(rows):
    data = ready()
    frame = pd.DataFrame(rows, columns=data['info']['features'])
    probabilities = data['model'].predict_proba(frame)[:,list(data['model'].classes_).index(1)]
    return [dict(probability=float(p), prediction=int(p >= .5), model=data['info']['name'], version=data['info']['version'], scope='Future close-following event; not crash probability') for p in probabilities]

@app.get('/api/health')
def health(): return dict(status='ready' if bundle else 'unavailable', model_loaded=bool(bundle), detail=load_error)

@app.get('/api/model/info')
def info(): return ready()['info']

@app.get('/api/model/metrics')
def metrics(): return ready()['metrics']

@app.get('/api/analytics')
def analytics(): return ready()['analytics']

@app.post('/api/predict')
def predict(pair: Pair):
    row = pair.model_dump()
    invalid = support(row, ready()['info'])
    if invalid: raise HTTPException(422, detail=f'Outside training support: {", ".join(invalid)}')
    return infer([row])[0]

@app.post('/api/predict-batch')
def predict_batch(batch: Batch):
    data = ready()
    results = []; valid = []
    for item in batch.pairs:
        row = item.inputs.model_dump()
        invalid = support(row, data['info'])
        if invalid: results.append(dict(id=item.id, eligible=False, reason=f'Outside training support: {", ".join(invalid)}'))
        else: valid.append((item.id, row))
    if valid:
        for (pair_id, _), prediction in zip(valid, infer([row for _, row in valid])):
            results.append(dict(id=pair_id, eligible=True, **prediction))
    return dict(predictions=results)
