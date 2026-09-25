from __future__ import annotations
import numpy as np
from geowave.model.traces import Trace

def constant_q_gain(frequencies_hz: np.ndarray, time_s: float, q: float, gain_limit: float=20.0) -> np.ndarray:
    if q <= 0:
        raise ValueError('Q must be positive')
    f = np.asarray(frequencies_hz, float)
    gain = np.exp(np.pi * np.maximum(f, 0.0) * max(time_s, 0.0) / q)
    return np.minimum(gain, gain_limit)

def inverse_q_trace(trace: Trace, q: float, reference_time_s: float | None=None, gain_limit: float=20.0) -> Trace:
    x = trace.samples
    if not x.size:
        return trace.copy_with(x, q_comp='empty')
    ref = reference_time_s
    if ref is None:
        ref = trace.header.start_s + trace.duration_s / 2
    f = np.fft.rfftfreq(x.size, trace.header.dt_s)
    z = np.fft.rfft(x)
    gain = constant_q_gain(f, ref, q, gain_limit)
    y = np.fft.irfft(z * gain, n=x.size)
    return trace.copy_with(y, q_comp=f'Q={q:g}')

def estimate_q_from_spectral_ratio(freq_hz: np.ndarray, amp_shallow: np.ndarray, amp_deep: np.ndarray, delta_time_s: float) -> float:
    f = np.asarray(freq_hz, float)
    a = np.asarray(amp_shallow, float)
    b = np.asarray(amp_deep, float)
    mask = (f > 0) & (a > 0) & (b > 0)
    if mask.sum() < 3 or delta_time_s <= 0:
        raise ValueError('insufficient spectral samples')
    y = np.log(b[mask] / a[mask])
    X = np.column_stack([f[mask], np.ones(mask.sum())])
    slope, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    if slope >= 0:
        return float('inf')
    return float(-np.pi * delta_time_s / slope)
