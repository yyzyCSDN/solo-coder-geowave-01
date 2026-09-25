from __future__ import annotations
import numpy as np
from scipy.signal import hilbert
from geowave.model.traces import Trace

def analytic_signal(trace: Trace) -> np.ndarray:
    return hilbert(trace.samples)

def envelope(trace: Trace) -> np.ndarray:
    return np.abs(analytic_signal(trace))

def instantaneous_phase(trace: Trace) -> np.ndarray:
    return np.unwrap(np.angle(analytic_signal(trace)))

def instantaneous_frequency(trace: Trace) -> np.ndarray:
    p = instantaneous_phase(trace)
    f = np.gradient(p) / (2 * np.pi * trace.header.dt_s)
    if f.size > 2:
        f[0] = f[1]
        f[-1] = f[-2]
    return f

def sweetness(trace: Trace, frequency_floor: float=1.0) -> np.ndarray:
    env = envelope(trace)
    freq = np.maximum(np.abs(instantaneous_frequency(trace)), frequency_floor)
    return env / np.sqrt(freq)
