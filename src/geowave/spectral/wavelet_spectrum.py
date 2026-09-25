from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.wavelet import Wavelet

@dataclass(frozen=True)
class WaveletSpectrum:
    peak_hz: float
    centroid_hz: float
    low_hz: float
    high_hz: float
    phase_at_peak_deg: float

def characterize_wavelet(wavelet: Wavelet, energy_fraction: float=0.9) -> WaveletSpectrum:
    z = np.fft.rfft(wavelet.samples)
    f = np.fft.rfftfreq(len(wavelet.samples), wavelet.dt_s)
    p = np.abs(z) ** 2
    if not p.size or p.sum() == 0:
        return WaveletSpectrum(0, 0, 0, 0, 0)
    peak_idx = int(np.argmax(p))
    centroid = float(np.sum(f * p) / np.sum(p))
    cs = np.cumsum(p)
    tail = (1 - energy_fraction) / 2
    lo = int(np.searchsorted(cs, cs[-1] * tail))
    hi = int(np.searchsorted(cs, cs[-1] * (1 - tail)))
    return WaveletSpectrum(float(f[peak_idx]), centroid, float(f[min(lo, len(f) - 1)]), float(f[min(hi, len(f) - 1)]), float(np.rad2deg(np.angle(z[peak_idx]))))

def match_wavelet_phase(reference: Wavelet, candidate: Wavelet) -> float:
    n = max(len(reference.samples), len(candidate.samples))
    a = np.fft.rfft(reference.samples, n=n)
    b = np.fft.rfft(candidate.samples, n=n)
    cross = a * np.conj(b)
    return float(np.rad2deg(np.angle(np.sum(cross))))
