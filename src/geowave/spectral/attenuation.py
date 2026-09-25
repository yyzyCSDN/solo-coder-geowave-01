from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import Trace
from geowave.spectral.fft_tools import spectrum

@dataclass(frozen=True)
class AttenuationEstimate:
    slope_per_hz: float
    q_estimate: float
    fit_r2: float

def spectral_decay(trace: Trace, low_hz: float, high_hz: float) -> tuple[np.ndarray, np.ndarray]:
    spec = spectrum(trace)
    mask = (spec.frequencies_hz >= low_hz) & (spec.frequencies_hz <= high_hz) & (spec.amplitude > 0)
    return (spec.frequencies_hz[mask], np.log(spec.amplitude[mask]))

def estimate_attenuation(shallow: Trace, deep: Trace, delta_time_s: float, low_hz: float, high_hz: float) -> AttenuationEstimate:
    fs, ls = spectral_decay(shallow, low_hz, high_hz)
    fd, ld = spectral_decay(deep, low_hz, high_hz)
    if fs.size < 3 or fd.size < 3:
        raise ValueError('insufficient spectral bandwidth')
    fmin = max(float(fs.min()), float(fd.min()))
    fmax = min(float(fs.max()), float(fd.max()))
    f = np.linspace(fmin, fmax, min(len(fs), len(fd)))
    y = np.interp(f, fd, ld) - np.interp(f, fs, ls)
    X = np.column_stack([f, np.ones_like(f)])
    slope, intercept = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ np.array([slope, intercept])
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst else 1.0
    q = float('inf') if slope >= 0 or delta_time_s <= 0 else float(-np.pi * delta_time_s / slope)
    return AttenuationEstimate(float(slope), q, r2)
