"""Exercise notebook preparation on tiny recordings, without the real archive."""
import ast
import csv
import gzip
import io
import json
import math
import time
import zipfile
from pathlib import Path

import ijson
import numpy as np
import pytest


@pytest.fixture
def preparation(tmp_path):
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/NearMiss.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"] if "def add_recording(" in "".join(cell["source"]))
    definitions = ast.Module(body=[node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    info = json.loads((Path(__file__).parent / "artifacts/info.json").read_text())
    env = dict(np=np, math=math, time=time, ijson=ijson, gzip=gzip, json=json,
               CACHE=tmp_path / "samples.csv.gz", FEATURES=info["features"])
    exec(compile(definitions, "notebook-preparation", "exec"), env)
    return env


def track(identity="ego", distance=0, lane=None, road=None, times=None, heading=math.pi / 2):
    t = np.round(np.arange(0, 4.01, .1), 6) if times is None else np.asarray(times)
    n = len(t)
    along = 20 * t + distance
    return dict(UUID=identity, length=4.0, width=1.8, objType="car", time=t.tolist(),
        x_vec=(along * math.cos(heading)).tolist(), y_vec=(along * math.sin(heading)).tolist(),
        vx_vec=[20.0]*n, vy_vec=[.002]*n, ax_vec=[0.0]*n, psi_vec=[heading]*n,
        lane_id_vec=lane if lane is not None else [1]*n,
        road_id_vec=road if road is not None else [1]*n,
        object_relation_dict_list=[{} for _ in t])


def process(env, objects):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("record/dynamicWorld.json", json.dumps(dict(objects=objects)))
    output=io.StringIO(); writer=csv.writer(output)
    columns=env["FEATURES"]+["target","recording","vehicle_id","second","lane_id","road_id","x","y","length","width","vehicle_is_truck"]
    writer.writerow(columns)
    with zipfile.ZipFile(buffer) as archive:
        env["add_recording"]("record/dynamicWorld.json", "test-record", writer, archive, [])
    return list(csv.DictReader(io.StringIO(output.getvalue())))


def test_lateral_velocity_and_rotated_front_gap(preparation):
    ego=track(); front=track("front", distance=30)
    ego["object_relation_dict_list"]=[dict(front_ego="front") for _ in ego["time"]]
    rows=[r for r in process(preparation,[ego,front]) if r["vehicle_id"]=="ego"]
    assert rows
    assert float(rows[0]["lateral_speed"])==pytest.approx(.002)
    assert float(rows[0]["front_gap"])==pytest.approx(26.0, abs=1e-4)
    assert float(rows[0]["has_leader"])==1
    assert float(rows[0]["left_rear_present"])==0
    assert float(rows[0]["left_rear_gap"])==150


@pytest.mark.parametrize("bad", ["road", "divider", "nonadjacent", "framegap"])
def test_incomplete_or_invalid_future_windows_are_censored(preparation, bad):
    ego=track()
    if bad=="road": ego["road_id_vec"][10:]=[2]*31
    elif bad=="divider": ego["lane_id_vec"][10]=0
    elif bad=="nonadjacent": ego["lane_id_vec"][10:]=[3]*31
    else:
        for key,value in list(ego.items()):
            if isinstance(value,list) and len(value)==41: ego[key]=value[:10]+value[12:]
    rows=process(preparation,[ego])
    assert all(int(r["second"])!=0 for r in rows)


def test_labels_follow_actual_crossings_and_require_complete_two_seconds(preparation):
    ego=track();ego["lane_id_vec"][15:]=[2]*26
    rows=process(preparation,[ego])
    assert {int(r["second"]):int(r["target"]) for r in rows}=={0:1,1:1,2:0}
