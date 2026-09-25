from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import Trace

@dataclass(frozen=True)
class Spectrum:
    frequencies_hz: np.ndarray
    amplitude: np.ndarray
    phase_rad: np.ndarray

def spectrum(trace: Trace, window: bool=True) -> Spectrum:
    x = trace.samples.astype(float)
    if window and x.size:
        x = x * np.hanning(x.size)
    z = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, trace.header.dt_s)
    return Spectrum(f, np.abs(z), np.angle(z))

def dominant_frequency(trace: Trace, min_hz: float=1.0) -> float:
    s = spectrum(trace)
    mask = s.frequencies_hz >= min_hz
    if not np.any(mask):
        return 0.0
    return float(s.frequencies_hz[mask][np.argmax(s.amplitude[mask])])

def phase_rotate(trace: Trace, degrees: float) -> Trace:
    z = np.fft.rfft(trace.samples)
    theta = np.deg2rad(degrees)
    z *= np.exp(1j * theta)
    y = np.fft.irfft(z, n=trace.samples.size)
    return trace.copy_with(y, phase_rotation=f'{degrees:g}')
