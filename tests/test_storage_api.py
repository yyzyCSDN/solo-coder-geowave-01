import numpy as np
from fastapi.testclient import TestClient
from geowave.storage.sqlite import Database
from geowave.workflow.artifacts import ArtifactStore
from geowave.service.app import create_app

def test_database_audit(tmp_path):
    db=Database(tmp_path/'db.sqlite');db.append_audit('x',{'a':1});db.append_audit('y',{'b':2});assert db.audit_valid()

def test_artifact_store(tmp_path):
    s=ArtifactStore(tmp_path/'art');a=s.put_bytes(b'abc','application/octet-stream',{'x':1});assert s.get(a.digest)==b'abc'

def test_http_end_to_end(tmp_path):
    client=TestClient(create_app(str(tmp_path/'state')))
    samples=(np.sin(np.linspace(0,10,128))).tolist()
    payload={'dataset_id':'d1','name':'demo','traces':[{'header':{'trace_id':'t1','shot_id':'s1','receiver_id':'r1','source_x':0,'source_y':0,'receiver_x':100,'receiver_y':0,'dt_s':.004,'start_s':0},'samples':samples},{'header':{'trace_id':'t2','shot_id':'s1','receiver_id':'r2','source_x':0,'source_y':0,'receiver_x':200,'receiver_y':0,'dt_s':.004,'start_s':0},'samples':samples}]}
    r=client.post('/v1/datasets',json=payload);assert r.status_code==200
    q=client.get('/v1/datasets/d1/qc');assert q.status_code==200
    run=client.post('/v1/datasets/d1/run',json={'name':'basic','steps':[{'operation':'remove_mean','parameters':{}},{'operation':'rms_agc','parameters':{'window_s':.05}}]});assert run.status_code==200
    assert client.get('/ready').json()['status']=='ready'
