import numpy as np
from geowave.model.horizon import Horizon,HorizonPick
from geowave.interpretation.gridding import grid_horizon
from geowave.interpretation.thickness import isochron
from geowave.interpretation.tracker import track_peak
from geowave.well.checkshot import Checkshot
from geowave.well.depth_time import build_depth_time
from geowave.well.synthetic import LogCurve,acoustic_impedance
from geowave.workflow.recipe import Recipe,StepSpec
from geowave.workflow.operations import build_default_executor
from geowave.workflow.executor import dataset_digest
from geowave.workflow.snapshot import snapshot_for,diff_snapshots

def test_horizon_grid():
    h=Horizon('H',[HorizonPick(0,0,.5),HorizonPick(100,0,.6),HorizonPick(0,100,.55),HorizonPick(100,100,.65)])
    g=grid_horizon(h,np.array([0,50,100]),np.array([0,50,100]))
    assert g.shape==(3,3)

def test_isochron():
    top=np.array([[.5,.6]]);bot=np.array([[.7,.9]])
    assert np.allclose(isochron(top,bot),[[.2,.3]])

def test_tracker():
    a=np.zeros((8,40))
    for i in range(8):a[i,10+i//2]=1
    points=track_peak(a,3,11,3)
    assert len(points)==8

def test_well_depth_time():
    m=build_depth_time([Checkshot(0,0),Checkshot(1000,1.0),Checkshot(2000,1.8)])
    assert 1400<m.depth_at_time(1.4)<1600

def test_acoustic_impedance():
    depth=np.arange(5);sonic=LogCurve(depth,np.full(5,80.0),'sonic');rho=LogCurve(depth,np.full(5,2.3),'rho')
    ai=acoustic_impedance(sonic,rho)
    assert np.all(ai.values>0)

def test_workflow_reproducible(dataset):
    recipe=Recipe('p',(StepSpec('remove_mean',{}),StepSpec('rms_agc',{'window_s':.1})))
    ex=build_default_executor();a=ex.execute(dataset,recipe,'a');b=ex.execute(dataset,recipe,'b')
    assert a.output_digest==b.output_digest
    snap=snapshot_for(a.input_digest,recipe,'1.0',{'x':1});snap2=snapshot_for(a.input_digest,recipe,'1.0',{'x':2})
    assert 'parameters' in diff_snapshots(snap,snap2)
    assert dataset_digest(dataset)==a.input_digest
