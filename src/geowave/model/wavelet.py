from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Wavelet:
    dt_s: float
    samples: np.ndarray
    name: str = 'wavelet'

    def normalized(self) -> 'Wavelet':
        a = np.asarray(self.samples, float)
        m = float(np.max(np.abs(a))) if a.size else 1.0
        return Wavelet(self.dt_s, a / (m or 1.0), self.name)

def ricker(f_hz: float, dt_s: float, length_s: float=0.128) -> Wavelet:
    if f_hz <= 0 or dt_s <= 0:
        raise ValueError('positive frequency and dt required')
    half = int(round(length_s / (2 * dt_s)))
    t = np.arange(-half, half + 1) * dt_s
    p = (math.pi * f_hz * t) ** 2
    a = (1 - 2 * p) * np.exp(-p)
    return Wavelet(dt_s, a, f'ricker-{f_hz:g}Hz')
