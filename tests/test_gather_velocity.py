import numpy as np
from geowave.gather.cmp import cmp_gathers,Gather
from geowave.gather.nmo import nmo_correct
from geowave.gather.stack import linear_stack
from geowave.gather.radon import linear_radon
from geowave.velocity.semblance import semblance_scan,pick_maxima
from geowave.velocity.dix import dix_interval
from geowave.velocity.interpolate import idw_velocity
from geowave.model.velocity import VelocityFunction,VelocityPick

def test_cmp_nmo_stack(dataset):
    g=max(cmp_gathers(dataset,50,50),key=lambda x:len(x.traces))
    f=VelocityFunction(g.x or 0,g.y or 0,[VelocityPick(0,1800),VelocityPick(2,2300)])
    corrected=Gather(g.key,[nmo_correct(t,f) for t in g.traces],g.x,g.y)
    stack=linear_stack(corrected)
    assert stack.samples.size>100

def test_semblance(dataset):
    g=max(cmp_gathers(dataset,100,100),key=lambda x:len(x.traces))
    panel=semblance_scan(g,np.array([1500,1800,2200,2600]))
    assert panel.values.shape[0]==4
    assert len(pick_maxima(panel,.01))>0

def test_dix():
    out=dix_interval([VelocityPick(.5,1800),VelocityPick(1,2200)])
    assert len(out)==2 and all(x.velocity_mps>0 for x in out)

def test_idw():
    a=VelocityFunction(0,0,[VelocityPick(0,1500),VelocityPick(1,2000)])
    b=VelocityFunction(100,0,[VelocityPick(0,1700),VelocityPick(1,2400)])
    c=idw_velocity([a,b],50,0,[0,.5,1])
    assert 1500<c.at(.5)<2400

def test_radon(dataset):
    g=max(cmp_gathers(dataset,100,100),key=lambda x:len(x.traces))
    panel=linear_radon(g,np.linspace(-.001,.001,11))
    assert panel.amplitudes.shape[0]==11
