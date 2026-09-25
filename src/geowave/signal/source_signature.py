from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve

@dataclass(frozen=True)
class SignatureEstimate:
    wavelet: np.ndarray
    delay_samples: int
    match_correlation: float

def estimate_signature(reference: np.ndarray, recorded: np.ndarray, length: int=81, regularization: float=0.001) -> SignatureEstimate:
    r = np.asarray(reference, float)
    y = np.asarray(recorded, float)
    if r.size != y.size:
        raise ValueError('input length mismatch')
    half = length // 2
    A = np.zeros((r.size, length))
    for j in range(length):
        shift = j - half
        if shift >= 0:
            A[shift:, j] = r[:r.size - shift]
        else:
            A[:shift, j] = r[-shift:]
    lhs = A.T @ A + regularization * np.eye(length)
    wavelet = np.linalg.solve(lhs, A.T @ y)
    pred = A @ wavelet
    corr = np.corrcoef(pred, y)[0, 1] if np.std(pred) and np.std(y) else 0.0
    delay = int(np.argmax(np.abs(wavelet)) - half)
    return SignatureEstimate(wavelet, delay, float(corr))

def remove_signature(recorded: np.ndarray, signature: np.ndarray, water_level: float=0.01) -> np.ndarray:
    x = np.asarray(recorded, float)
    w = np.asarray(signature, float)
    n = x.size
    W = np.fft.rfft(w, n=n)
    X = np.fft.rfft(x)
    floor = water_level * np.max(np.abs(W) ** 2)
    inv = np.conj(W) / (np.abs(W) ** 2 + floor)
    return np.fft.irfft(X * inv, n=n)
