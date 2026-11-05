from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

ARTIFACTS=Path(os.getenv('ARTIFACT_DIR',Path(__file__).parent/'artifacts'))
TASK='lane-change-behavior'
bundle={}
load_error=None

@asynccontextmanager
async def lifespan(app):
    global load_error
    bundle.clear()
    try:
        loaded={name:json.loads((ARTIFACTS/f'{name}.json').read_text(encoding='utf-8')) for name in ['info','metrics','analytics','replay']}
        if any(value.get('task')!=TASK for value in loaded.values()): raise ValueError('Incompatible task artifacts')
        loaded['model']=joblib.load(ARTIFACTS/'behavior.joblib')
        if list(loaded['model'].feature_names_in_)!=loaded['info']['features']: raise ValueError('Feature schema mismatch')
        bundle.update(loaded);load_error=None
    except Exception as exc:
        load_error=f'Behavior model unavailable ({type(exc).__name__}). Execute notebooks/NearMiss.ipynb.'
    yield

app=FastAPI(title='NearMiss Lane-Change Behavior API',version='2.0.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=[v.strip() for v in os.getenv('CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173').split(',') if v.strip()],allow_methods=['GET','POST'],allow_headers=['Content-Type'])

@app.exception_handler(RequestValidationError)
async def validation_error(request,exc):
    return JSONResponse(status_code=422,content={'detail':[{key:e[key] for key in ['loc','msg','type']} for e in exc.errors()]})

def ready():
    if not bundle: raise HTTPException(503,detail=load_error or 'Model unavailable')
    return bundle

class Observation(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    speed: float=Field(ge=0,le=55)
    acceleration: float=Field(ge=-12,le=8)
    lateral_speed: float=Field(ge=0,le=15)
    front_gap: float=Field(ge=0,le=150)
    closing_speed: float=Field(ge=-55,le=55)
    has_leader: Literal[0,1]

    @model_validator(mode='after')
    def absence_encoding(self):
        if not self.has_leader and (self.front_gap!=150 or self.closing_speed!=0): raise ValueError('Without an observed leader, encode front_gap=150 and closing_speed=0.')
        return self

class Item(BaseModel):
    model_config=ConfigDict(extra='forbid')
    id: str=Field(min_length=1,max_length=50)
    inputs: Observation

class Batch(BaseModel):
    model_config=ConfigDict(extra='forbid')
    observations: list[Item]=Field(min_length=1,max_length=30)

    @model_validator(mode='after')
    def unique_ids(self):
        if len({item.id for item in self.observations})!=len(self.observations): raise ValueError('Observation IDs must be unique.')
        return self

def outside_support(row,info):
    return [key for key,value in row.items() if not info['ranges'][key][0]<=value<=info['ranges'][key][1]]

def infer(rows):
    data=ready(); frame=pd.DataFrame(rows,columns=data['info']['features'])
    probabilities=data['model'].predict_proba(frame)[:,list(data['model'].classes_).index(1)]
    threshold=data['info']['decision_threshold']
    return [dict(task=TASK,version=data['info']['version'],probability=float(p),prediction=int(p>=threshold),label='Lane change' if p>=threshold else 'Stay in lane',model=data['info']['name'],decision_threshold=threshold,horizon_seconds=2,scope='Estimated observed lane-crossing behavior; not accident probability or a lane recommendation') for p in probabilities]

@app.get('/api/health')
def health(): return dict(status='ready' if bundle else 'unavailable',model_loaded=bool(bundle),task=TASK,detail=load_error)

@app.get('/api/model/info')
def info(): return ready()['info']

@app.get('/api/model/metrics')
def metrics(): return ready()['metrics']

@app.get('/api/analytics')
def analytics(): return ready()['analytics']

@app.get('/api/replay')
def replay(): return ready()['replay']

@app.post('/api/predict')
def predict(observation:Observation):
    row=observation.model_dump(); invalid=outside_support(row,ready()['info'])
    if invalid: raise HTTPException(422,detail=f'Outside training support: {", ".join(invalid)}')
    return infer([row])[0]

@app.post('/api/predict-batch')
def predict_batch(batch:Batch):
    data=ready(); valid=[]; results=[]
    for item in batch.observations:
        row=item.inputs.model_dump(); invalid=outside_support(row,data['info'])
        if invalid: results.append(dict(id=item.id,eligible=False,reason=f'Outside training support: {", ".join(invalid)}'))
        else: valid.append((item.id,row))
    if valid:
        results.extend(dict(id=pair_id,eligible=True,**prediction) for (pair_id,_),prediction in zip(valid,infer([row for _,row in valid])))
    return dict(task=TASK,predictions=results)
