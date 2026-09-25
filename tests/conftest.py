import numpy as np
import pytest
from geowave.model.traces import TraceHeader, Trace, TraceSet

@pytest.fixture
def dataset():
    traces=[]
    dt=.004
    t=np.arange(512)*dt
    for shot in range(3):
        sx=shot*50.0
        for r in range(6):
            rx=r*25.0
            off=abs(rx-sx)
            event=np.exp(-((t-(.45+off/2200))/.02)**2)
            wave=np.sin(2*np.pi*30*t)*event
            noise=.02*np.sin(2*np.pi*6*t+0.3*r)
            h=TraceHeader(f't{shot}-{r}',f's{shot}',f'r{r}',sx,0,rx,0,dt,0)
            traces.append(Trace(h,wave+noise))
    return TraceSet(traces,'fixture')
