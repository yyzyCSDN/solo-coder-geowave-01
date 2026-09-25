from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FKPanel:
    frequencies_hz: np.ndarray
    wavenumbers_per_m: np.ndarray
    amplitude: np.ndarray

def fk_transform(data: np.ndarray, dt_s: float, dx_m: float) -> FKPanel:
    a = np.asarray(data, float)
    if a.ndim != 2:
        raise ValueError('data must be traces x samples')
    spec = np.fft.fftshift(np.fft.fft2(a), axes=0)
    f = np.fft.rfftfreq(a.shape[1], dt_s)
    k = np.fft.fftshift(np.fft.fftfreq(a.shape[0], dx_m))
    half = spec[:, :f.size]
    return FKPanel(f, k, np.abs(half))

def velocity_fan_mask(panel: FKPanel, min_velocity: float, max_velocity: float) -> np.ndarray:
    F, K = np.meshgrid(panel.frequencies_hz, panel.wavenumbers_per_m)
    vel = np.full_like(F, np.inf, dtype=float)
    m = np.abs(K) > 1e-12
    vel[m] = np.abs(F[m] / K[m])
    return (vel >= min_velocity) & (vel <= max_velocity)

def apply_fk_mask(data: np.ndarray, dt_s: float, dx_m: float, mask: np.ndarray) -> np.ndarray:
    a = np.asarray(data, float)
    spec = np.fft.fftshift(np.fft.fft2(a), axes=0)
    full = np.zeros_like(spec, dtype=bool)
    nf = a.shape[1] // 2 + 1
    full[:, :nf] = mask
    for j in range(1, a.shape[1] // 2):
        full[:, -j] = mask[:, j]
    filtered = np.fft.ifft2(np.fft.ifftshift(spec * full, axes=0)).real
    return filtered
