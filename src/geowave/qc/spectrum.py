from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import TraceSet
from geowave.spectral.fft_tools import spectrum

@dataclass(frozen=True)
class SpectralQC:
    low_hz: float
    high_hz: float
    centroid_hz: float
    bandwidth_hz: float

def aggregate_spectrum(dataset: TraceSet) -> tuple[np.ndarray, np.ndarray]:
    if not dataset.traces:
        return (np.array([]), np.array([]))
    specs = [spectrum(t) for t in dataset.traces]
    n = min((s.amplitude.size for s in specs))
    f = specs[0].frequencies_hz[:n]
    amp = np.mean([s.amplitude[:n] for s in specs], axis=0)
    return (f, amp)

def spectral_qc(dataset: TraceSet, energy_fraction: float=0.9) -> SpectralQC:
    f, a = aggregate_spectrum(dataset)
    if not f.size:
        return SpectralQC(0, 0, 0, 0)
    p = a * a
    cs = np.cumsum(p)
    total = cs[-1] or 1
    lo_i = np.searchsorted(cs, total * (1 - energy_fraction) / 2)
    hi_i = np.searchsorted(cs, total * (1 - (1 - energy_fraction) / 2))
    cent = float(np.sum(f * p) / total)
    bw = float(np.sqrt(np.sum((f - cent) ** 2 * p) / total))
    return SpectralQC(float(f[min(lo_i, len(f) - 1)]), float(f[min(hi_i, len(f) - 1)]), cent, bw)
