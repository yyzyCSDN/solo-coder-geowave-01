from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from geowave.model.traces import Trace

def estimate_swell(trace: Trace, cutoff_hz: float=2.0) -> np.ndarray:
    sigma = 1.0 / max(2 * np.pi * cutoff_hz * trace.header.dt_s, 1e-06)
    sigma = min(max(sigma, 1.0), max(1.0, trace.samples.size / 8))
    return gaussian_filter1d(trace.samples, sigma=sigma, mode='reflect')

def remove_swell(trace: Trace, cutoff_hz: float=2.0) -> Trace:
    noise = estimate_swell(trace, cutoff_hz)
    return trace.copy_with(trace.samples - noise, swell=f'removed<{cutoff_hz:g}Hz')

def swell_ratio(trace: Trace, cutoff_hz: float=2.0) -> float:
    noise = estimate_swell(trace, cutoff_hz)
    num = float(np.sqrt(np.mean(noise * noise)))
    den = float(np.sqrt(np.mean(trace.samples * trace.samples))) + 1e-12
    return num / den
