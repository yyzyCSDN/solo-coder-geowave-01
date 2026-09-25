from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import correlate

@dataclass(frozen=True)
class TieResult:
    shift_s: float
    correlation: float
    scale: float

def tie_synthetic(seismic: np.ndarray, synthetic: np.ndarray, dt_s: float, max_shift_s: float=0.1) -> TieResult:
    a = np.asarray(seismic, float)
    b = np.asarray(synthetic, float)
    n = min(a.size, b.size)
    a = a[:n] - np.mean(a[:n])
    b = b[:n] - np.mean(b[:n])
    c = correlate(a, b, mode='full')
    lags = np.arange(-n + 1, n)
    m = np.abs(lags) <= round(max_shift_s / dt_s)
    idx = np.argmax(c[m])
    lag = lags[m][idx]
    shifted = np.roll(b, lag)
    den = float(np.dot(shifted, shifted))
    scale = float(np.dot(a, shifted) / den) if den else 1.0
    corr = float(c[m][idx] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return TieResult(float(lag * dt_s), corr, scale)
