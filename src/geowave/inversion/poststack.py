from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve

@dataclass(frozen=True)
class PoststackInversionResult:
    reflectivity: np.ndarray
    impedance: np.ndarray
    synthetic: np.ndarray
    residual: np.ndarray

def model_based_inversion(trace: np.ndarray, wavelet: np.ndarray, initial_impedance: float, iterations: int=80, step: float=0.2, smoothness: float=0.02) -> PoststackInversionResult:
    x = np.asarray(trace, float)
    w = np.asarray(wavelet, float)
    r = np.zeros_like(x)
    for _ in range(iterations):
        pred = fftconvolve(r, w, mode='same')
        residual = pred - x
        grad = fftconvolve(residual, w[::-1], mode='same')
        lap = np.zeros_like(r)
        lap[1:-1] = 2 * r[1:-1] - r[:-2] - r[2:]
        r -= step * (grad + smoothness * lap)
        r = np.clip(r, -0.95, 0.95)
    z = np.empty_like(r)
    z[0] = initial_impedance
    for i in range(1, len(z)):
        z[i] = z[i - 1] * (1 + r[i]) / max(1 - r[i], 1e-06)
    synthetic = fftconvolve(r, w, mode='same')
    return PoststackInversionResult(r, z, synthetic, x - synthetic)
