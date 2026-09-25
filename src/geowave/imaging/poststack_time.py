from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TimeMigrationResult:
    migrated: np.ndarray
    x_m: np.ndarray
    time_s: np.ndarray

def stolt_like_shift(section: np.ndarray, dx_m: float, dt_s: float, velocity_mps: float) -> TimeMigrationResult:
    a = np.asarray(section, float)
    if a.ndim != 2:
        raise ValueError('section must be traces x samples')
    nx, nt = a.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx_m)
    omega = 2 * np.pi * np.fft.rfftfreq(nt, dt_s)
    spec = np.fft.fft(np.fft.rfft(a, axis=1), axis=0)
    out = np.zeros_like(spec)
    for ix, k in enumerate(kx):
        mapped = np.sqrt(np.maximum(omega * omega - (velocity_mps * k / 2) ** 2, 0.0))
        real = np.interp(mapped, omega, spec[ix].real, left=0, right=0)
        imag = np.interp(mapped, omega, spec[ix].imag, left=0, right=0)
        out[ix] = real + 1j * imag
    migrated = np.fft.irfft(np.fft.ifft(out, axis=0), n=nt, axis=1).real
    return TimeMigrationResult(migrated, np.arange(nx) * dx_m, np.arange(nt) * dt_s)
