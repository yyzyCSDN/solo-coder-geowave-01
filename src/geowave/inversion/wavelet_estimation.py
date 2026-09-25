from __future__ import annotations
import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve

def estimate_wavelet(trace: np.ndarray, reflectivity: np.ndarray, length: int=41, regularization: float=0.001) -> np.ndarray:
    t = np.asarray(trace, float)
    r = np.asarray(reflectivity, float)
    if t.size != r.size:
        raise ValueError('length mismatch')
    n = t.size
    half = length // 2
    A = np.zeros((n, length))
    for j in range(length):
        shift = j - half
        if shift >= 0:
            A[shift:, j] = r[:n - shift]
        else:
            A[:shift, j] = r[-shift:]
    lhs = A.T @ A + regularization * np.eye(length)
    rhs = A.T @ t
    return np.linalg.solve(lhs, rhs)

def phase_match(observed: np.ndarray, synthetic: np.ndarray) -> float:
    o = np.fft.rfft(observed)
    s = np.fft.rfft(synthetic)
    cross = o * np.conj(s)
    phase = np.angle(np.sum(cross))
    return float(np.rad2deg(phase))
