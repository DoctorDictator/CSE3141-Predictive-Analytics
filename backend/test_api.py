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
def example():
    return json.loads((main.ARTIFACTS/'example.json').read_text())

def test_research_and_health(client):
    assert client.get('/api/health').json()['model_loaded'] is True
    for path in ['/api/model/info','/api/model/metrics','/api/analytics']:
        assert client.get(path).status_code == 200
    info=client.get('/api/model/info').json()
    assert info['horizon_seconds']==2
    assert info['headway_threshold_seconds']==1.5

def test_probability_equals_exported_pipeline(client,example):
    result=client.post('/api/predict',json=example)
    assert result.status_code==200
    model=joblib.load(main.ARTIFACTS/'nearmiss.joblib')
    prob=model.predict_proba(pd.DataFrame([example],columns=model.feature_names_in_))[0,1]
    assert result.json()['probability']==pytest.approx(prob)
    assert 0<=prob<=1
    assert 'not crash' in result.json()['scope']

def test_batch_preserves_ids_and_per_item_support(client,example):
    bad={**example,'follower_speed':50}
    result=client.post('/api/predict-batch',json={'pairs':[{'id':'v2','inputs':example},{'id':'v3','inputs':bad}]}).json()
    items={r['id']:r for r in result['predictions']}
    assert items['v2']['eligible'] is True
    assert items['v3']['eligible'] is False
    assert 'probability' not in items['v3']

@pytest.mark.parametrize('field,value',[('follower_speed',-1),('leader_accel',100),('leader_length',0),('spacing_m',0),('unknown',1)])
def test_invalid_inputs(client,example,field,value):
    assert client.post('/api/predict',json={**example,field:value}).status_code==422

def test_overlap_and_outside_training_support(client,example):
    assert client.post('/api/predict',json={**example,'spacing_m':example['leader_length']}).status_code==422
    assert client.post('/api/predict',json={**example,'follower_speed':50}).status_code==422

@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_input_is_readable_validation_error(client,example,value):
    result=client.post('/api/predict',content=json.dumps({**example,'follower_speed':value}),headers={'Content-Type':'application/json'})
    assert result.status_code==422
    assert result.json()['detail'][0]['loc']==['body','follower_speed']

def test_batch_size_and_duplicate_ids(client,example):
    item={'id':'same','inputs':example}
    for pairs in [[],[item]*31,[item,item]]:
        assert client.post('/api/predict-batch',json={'pairs':pairs}).status_code==422

def test_cors(client):
    result=client.options('/api/predict',headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'POST'})
    assert result.headers['access-control-allow-origin']=='http://localhost:5173'

def test_unavailable_artifacts_never_fabricate_model(monkeypatch,tmp_path):
    monkeypatch.setattr(main,'ARTIFACTS',tmp_path)
    with TestClient(main.app) as c:
        assert c.get('/api/health').json()['model_loaded'] is False
        assert c.get('/api/model/info').status_code==503

def test_notebook_executed_and_only_current_features():
    nb=json.loads(Path('notebooks/NearMiss.ipynb').read_text(encoding='utf-8'))
    code=[c for c in nb['cells'] if c['cell_type']=='code']
    assert all(c['execution_count'] is not None for c in code)
    assert not any(o['output_type']=='error' for c in code for o in c['outputs'])
    info=json.loads((main.ARTIFACTS/'info.json').read_text())
    assert not set(info['features']) & {'target','future_min','vehicle_id','global_time','lane_id'}
    metrics=json.loads((main.ARTIFACTS/'metrics.json').read_text())
    assert {r['name'] for r in metrics['comparison']}=={'Logistic Regression','LDA','QDA'}
    assert sum(sum(row) for row in metrics['test']['confusion_matrix'])==metrics['split_sizes']['test']

def test_future_labels_use_complete_subsequent_windows():
    # Execute the notebook's actual labeling cell on controlled trajectories.
    nb=json.loads(Path('notebooks/NearMiss.ipynb').read_text(encoding='utf-8'))
    cell=next(c for c in nb['cells'] if c['cell_type']=='code' and 'future_headways=[]' in ''.join(c['source']))
    rows=[]
    for v in [1,2,3,4,5]:
        for frame in range(25):
            if v==3 and frame==10: continue  # incomplete window
            rows.append(dict(vehicle_id=v,global_time=frame*100,preceding=100+(frame>=12 if v==4 else 0),lane_id=1,v_vel=0 if v==5 and frame==10 else 10,headway=1 if v==1 and frame==20 else 2))
    ns={'pairs':pd.DataFrame(rows),'pd':pd,'np':np}
    exec(''.join(cell['source']),ns)
    eligible=ns['eligible']
    assert eligible[eligible.vehicle_id==1].iloc[0].target==1
    assert eligible[eligible.vehicle_id==2].iloc[0].target==0
    assert not eligible.vehicle_id.isin([3,4,5]).any()
    # Last 20 observations are not zero-labeled; they are censored.
    assert eligible.groupby('vehicle_id').size().to_dict()=={1:5,2:5}
