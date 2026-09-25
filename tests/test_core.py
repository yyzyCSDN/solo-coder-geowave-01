import numpy as np
from geowave.model.traces import TraceHeader,Trace,TraceSet
from geowave.model.geometry import GeometryIndex
from geowave.model.velocity import VelocityFunction,VelocityPick,VelocityModel1D
from geowave.model.wavelet import ricker
from geowave.util.coordinates import LocalFrame
from geowave.util.units import convert

def test_trace_geometry(dataset):
    tr=dataset.traces[0]
    assert tr.header.midpoint==(0.0,0.0)
    assert tr.header.offset_m==0
    geom=GeometryIndex(dataset,25,25)
    assert sum(geom.fold_map().values())==len(dataset.traces)

def test_velocity_interpolation():
    f=VelocityFunction(0,0,[VelocityPick(0,1500),VelocityPick(1,2500)])
    assert f.at(.5)==2000

def test_velocity_model_time():
    m=VelocityModel1D(np.array([0,1000]),np.array([2000,2000]))
    assert abs(m.one_way_time(1000)-.5)<.01

def test_ricker_and_frame():
    w=ricker(25,.002)
    assert abs(np.max(np.abs(w.samples))-1)<1e-9
    frame=LocalFrame(100,200,90)
    local=frame.to_local(110,200)
    assert abs(local[0]-10)<1e-6
    world=frame.to_world(*local)
    assert np.allclose(world,(110,200))

def test_units():
    assert abs(convert(1,'km','m')-1000)<1e-9
