from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import stft
from geowave.model.traces import Trace

@dataclass(frozen=True)
class TimeFrequencyPanel:
    times_s: np.ndarray
    frequencies_hz: np.ndarray
    magnitude: np.ndarray

def spectrogram(trace: Trace, window_s: float=0.128, overlap: float=0.75) -> TimeFrequencyPanel:
    n = max(8, int(round(window_s / trace.header.dt_s)))
    nover = min(n - 1, int(round(n * overlap)))
    f, t, z = stft(trace.samples, fs=1 / trace.header.dt_s, nperseg=n, noverlap=nover, boundary=None)
    return TimeFrequencyPanel(trace.header.start_s + t, f, np.abs(z))

def band_energy(panel: TimeFrequencyPanel, low: float, high: float) -> np.ndarray:
    mask = (panel.frequencies_hz >= low) & (panel.frequencies_hz <= high)
    if not np.any(mask):
        return np.zeros(panel.times_s.size)
    return np.trapz(panel.magnitude[mask, :] ** 2, panel.frequencies_hz[mask], axis=0)
