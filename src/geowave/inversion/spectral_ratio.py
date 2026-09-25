from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SpectralRatioResult:
    frequencies_hz: np.ndarray
    log_ratio: np.ndarray
    slope: float
    intercept: float
    r2: float

def spectral_ratio(reference: np.ndarray, target: np.ndarray, dt_s: float, low_hz: float, high_hz: float) -> SpectralRatioResult:
    a = np.asarray(reference, float)
    b = np.asarray(target, float)
    n = min(len(a), len(b))
    A = np.abs(np.fft.rfft(a[:n] * np.hanning(n)))
    B = np.abs(np.fft.rfft(b[:n] * np.hanning(n)))
    f = np.fft.rfftfreq(n, dt_s)
    mask = (f >= low_hz) & (f <= high_hz) & (A > 0) & (B > 0)
    ff = f[mask]
    ratio = np.log(B[mask] / A[mask])
    if len(ff) < 3:
        raise ValueError('insufficient spectral samples')
    X = np.column_stack([ff, np.ones_like(ff)])
    slope, intercept = np.linalg.lstsq(X, ratio, rcond=None)[0]
    pred = X @ np.array([slope, intercept])
    sse = float(np.sum((ratio - pred) ** 2))
    sst = float(np.sum((ratio - np.mean(ratio)) ** 2))
    return SpectralRatioResult(ff, ratio, float(slope), float(intercept), 1 - sse / sst if sst else 1.0)
