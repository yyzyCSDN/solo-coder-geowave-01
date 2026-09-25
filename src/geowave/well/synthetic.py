from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve
from geowave.model.wavelet import Wavelet

@dataclass(frozen=True)
class LogCurve:
    depth_m: np.ndarray
    values: np.ndarray
    name: str

    def __post_init__(self):
        object.__setattr__(self, 'depth_m', np.asarray(self.depth_m, float))
        object.__setattr__(self, 'values', np.asarray(self.values, float))
        if self.depth_m.shape != self.values.shape:
            raise ValueError('log shape')

def acoustic_impedance(sonic_us_per_ft: LogCurve, density_gcc: LogCurve) -> LogCurve:
    if sonic_us_per_ft.depth_m.shape != density_gcc.depth_m.shape:
        raise ValueError('log sampling mismatch')
    velocity_mps = 304800.0 / sonic_us_per_ft.values
    density_kgm3 = density_gcc.values * 1000
    return LogCurve(sonic_us_per_ft.depth_m, velocity_mps * density_kgm3, 'AI')

def synthetic_from_impedance(ai: LogCurve, wavelet: Wavelet, dt_s: float, time_depth: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    depth_t, time_t = time_depth
    times = np.arange(float(time_t.min()), float(time_t.max()), dt_s)
    z = np.interp(times, time_t, ai.values)
    r = np.zeros_like(z)
    r[1:] = (z[1:] - z[:-1]) / (z[1:] + z[:-1] + 1e-12)
    return (times, fftconvolve(r, wavelet.samples, mode='same'))
