import numpy as np
from geowave.io.tracepack import write_tracepack,read_tracepack
from geowave.signal.detrend import remove_mean
from geowave.signal.filters import bandpass
from geowave.signal.agc import rms_agc
from geowave.signal.despike import repair_spikes
from geowave.signal.resample import resample_trace
from geowave.signal.statics import apply_static,estimate_shift
from geowave.signal.q_compensation import inverse_q_trace

def test_tracepack_roundtrip(dataset,tmp_path):
    p=tmp_path/'a.gwtp';write_tracepack(p,dataset);got=read_tracepack(p)
    assert len(got.traces)==len(dataset.traces)
    assert np.allclose(got.traces[2].samples,dataset.traces[2].samples,atol=1e-6)

def test_signal_chain(dataset):
    t=remove_mean(dataset.traces[1]);t=bandpass(t,10,80);t=rms_agc(t,.1)
    assert t.samples.shape==dataset.traces[1].samples.shape
    assert np.isfinite(t.samples).all()

def test_despike(dataset):
    tr=dataset.traces[0];x=tr.samples.copy();x[100]=100
    bad=tr.copy_with(x);good=repair_spikes(bad,9,5)
    assert abs(good.samples[100])<10

def test_resample(dataset):
    tr=resample_trace(dataset.traces[0],.002)
    assert tr.samples.size>dataset.traces[0].samples.size
    assert tr.header.dt_s==.002

def test_static_estimation(dataset):
    ref=dataset.traces[0]
    shifted=apply_static(ref,.012)
    est=estimate_shift(ref,shifted,.05)
    assert abs(abs(est.shift_s)-.012)<=ref.header.dt_s

def test_q_compensation(dataset):
    out=inverse_q_trace(dataset.traces[0],80,gain_limit=3)
    assert np.isfinite(out.samples).all()
