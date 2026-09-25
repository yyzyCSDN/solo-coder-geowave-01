from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve

def reflectivity_from_impedance(impedance: np.ndarray) -> np.ndarray:
    z = np.asarray(impedance, float)
    if np.any(z <= 0):
        raise ValueError('impedance positive')
    r = np.zeros_like(z)
    r[1:] = (z[1:] - z[:-1]) / (z[1:] + z[:-1])
    return r

def recursive_impedance(reflectivity: np.ndarray, z0: float) -> np.ndarray:
    r = np.asarray(reflectivity, float)
    z = np.empty_like(r)
    z[0] = z0
    for i in range(1, len(r)):
        q = float(np.clip(r[i], -0.999, 0.999))
        z[i] = z[i - 1] * (1 + q) / (1 - q)
    return z

def synthetic_trace(impedance: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    return fftconvolve(reflectivity_from_impedance(impedance), np.asarray(wavelet, float), mode='same')

def sparse_spike_inversion(trace: np.ndarray, wavelet: np.ndarray, iterations: int=100, lam: float=0.02) -> np.ndarray:
    x = np.asarray(trace, float)
    w = np.asarray(wavelet, float)
    r = np.zeros_like(x)
    step = 0.5 / (np.sum(w * w) + 1e-09)
    for _ in range(iterations):
        pred = fftconvolve(r, w, mode='same')
        grad = fftconvolve(pred - x, w[::-1], mode='same')
        u = r - step * grad
        r = np.sign(u) * np.maximum(np.abs(u) - lam * step, 0)
    return r
