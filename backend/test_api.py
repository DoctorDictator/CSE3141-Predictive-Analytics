import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend import main

@pytest.fixture
def client():
    with TestClient(main.app) as c: yield c

@pytest.fixture
def example(): return json.loads((main.ARTIFACTS/'example.json').read_text())

def test_readiness_and_task(client):
    assert client.get('/api/health').json()['model_loaded'] is True
    assert client.get('/api/health').json()['task']=='lane-change-behavior'
    for path in ['/api/model/info','/api/model/metrics','/api/analytics','/api/replay']:
        response=client.get(path);assert response.status_code==200;assert response.json()['task']==main.TASK
    assert client.get('/api/lane-change/metrics').status_code==404

def test_actual_pipeline_probability_and_threshold(client,example):
    info=client.get('/api/model/info').json(); model=joblib.load(main.ARTIFACTS/'behavior.joblib')
    response=client.post('/api/predict',json=example);assert response.status_code==200
    p=model.predict_proba(pd.DataFrame([example],columns=info['features']))[0,1]
    assert response.json()['probability']==pytest.approx(p)
    assert response.json()['prediction']==int(p>=info['decision_threshold'])
    assert response.json()['label'] in ['Stay in lane','Lane change']
    for check in info['checks']:
        p=model.predict_proba(pd.DataFrame([check['inputs']],columns=info['features']))[0,1]
        assert p==pytest.approx(check['probability'])

def test_batch_ids_and_support(client,example):
    payload={'observations':[{'id':'v1','inputs':example},{'id':'v2','inputs':{**example,'speed':50}}]}
    r=client.post('/api/predict-batch',json=payload);assert r.status_code==200
    predictions={p['id']:p for p in r.json()['predictions']}
    assert predictions['v1']['eligible'] is True
    assert predictions['v2']['eligible'] is False and 'probability' not in predictions['v2']

@pytest.mark.parametrize('field,value',[('speed',-1),('acceleration',100),('lateral_speed',-1),('front_gap',151),('has_leader',2),('unknown',1)])
def test_invalid_features(client,example,field,value):
    assert client.post('/api/predict',json={**example,field:value}).status_code==422

@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_values(client,example,value):
    r=client.post('/api/predict',content=json.dumps({**example,'speed':value}),headers={'Content-Type':'application/json'})
    assert r.status_code==422;assert r.json()['detail'][0]['loc']==['body','speed']

def test_batch_limits_legacy_body_and_absence_encoding(client,example):
    item={'id':'same','inputs':example}
    for observations in [[],[item]*31,[item,item]]: assert client.post('/api/predict-batch',json={'observations':observations}).status_code==422
    assert client.post('/api/predict-batch',json={'pairs':[item]}).status_code==422
    assert client.post('/api/predict',json={**example,'has_leader':0,'front_gap':20}).status_code==422

def test_unavailable_model_has_no_substitute_predictions(monkeypatch,tmp_path):
    monkeypatch.setattr(main,'ARTIFACTS',tmp_path)
    with TestClient(main.app) as c:
        assert c.get('/api/health').json()['model_loaded'] is False
        assert c.get('/api/model/info').status_code==503

def test_executed_notebook_artifacts_and_replay_labels():
    nb=json.loads(Path('notebooks/NearMiss.ipynb').read_text(encoding='utf-8'))
    for cell in nb['cells']:
        assert 'viva' not in ''.join(cell['source']).lower()
        if cell['cell_type']=='code':
            assert cell['execution_count'] is not None
            assert not any(o['output_type']=='error' for o in cell['outputs'])
    info=json.loads((main.ARTIFACTS/'info.json').read_text())
    assert not set(info['features'])&{'target','future_lane','vehicle_id','global_time','lane_id'}
    metrics=json.loads((main.ARTIFACTS/'metrics.json').read_text())
    assert {r['name'] for r in metrics['comparison']}=={'Logistic Regression','LDA','QDA'}
    assert sum(map(sum,metrics['test']['confusion_matrix']))==metrics['split_sizes']['test']
    assert metrics['baseline']['f1']==0 and metrics['baseline']['recall']==0
    replay=json.loads((main.ARTIFACTS/'replay.json').read_text())
    for e in replay['examples']:
        lanes=[next(c['lane'] for c in s['cars'] if c['id']==e['vehicle_id']) for s in e['snapshots']]
        assert e['actual']==int(any(abs(b-a)==1 for a,b in zip(lanes,lanes[1:])))
        assert e['predicted']==int(e['probability']>=info['decision_threshold'])
    assert {(e['actual'],e['predicted']) for e in replay['examples']}=={(0,0),(0,1),(1,0),(1,1)}

def test_label_logic_uses_actual_crossings_and_complete_windows():
    nb=json.loads(Path('notebooks/NearMiss.ipynb').read_text(encoding='utf-8'))
    cell=next(c for c in nb['cells'] if c['cell_type']=='code' and 'future_group=df.groupby' in ''.join(c['source']))
    rows=[]
    for vehicle in range(1,7):
        for frame in range(25 if vehicle!=6 else 20):
            if vehicle==3 and frame==10:continue
            lane=1
            if vehicle in [1,4] and frame>=10: lane=2
            if vehicle==4 and frame>=15: lane=1
            if vehicle==5 and frame>=10: lane=3
            rows.append(dict(vehicle_id=vehicle,global_time=frame*100,lane_id=lane,frame_id=frame,speed=10,acceleration=0,lateral_speed=.1,front_gap=20,closing_speed=0,has_leader=1))
    frame=pd.DataFrame(rows).sort_values(['vehicle_id','global_time']).reset_index(drop=True)
    features=['speed','acceleration','lateral_speed','front_gap','closing_speed','has_leader']
    ns={'df':frame.copy(),'observed':frame.copy(),'valid_leader':pd.Series(True,index=frame.index),'features':features,'pd':pd,'np':np}
    exec(''.join(cell['source']),ns)
    eligible=ns['eligible']
    assert set(eligible.vehicle_id)=={1,2,4}
    assert eligible.groupby('vehicle_id').target.first().to_dict()=={1:1,2:0,4:1}
    assert eligible.groupby('vehicle_id').size().to_dict()=={1:5,2:5,4:5}

def test_cors(client):
    r=client.options('/api/predict',headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'POST'})
    assert r.headers['access-control-allow-origin']=='http://localhost:5173'
