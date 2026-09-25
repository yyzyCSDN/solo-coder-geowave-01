from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import Trace

@dataclass(frozen=True)
class GhostModel:
    delay_s: float
    reflection_coefficient: float

def apply_ghost(trace: Trace, model: GhostModel) -> Trace:
    delay_samples = model.delay_s / trace.header.dt_s
    index = np.arange(trace.samples.size)
    delayed = np.interp(index - delay_samples, index, trace.samples, left=0.0, right=0.0)
    y = trace.samples + model.reflection_coefficient * delayed
    return trace.copy_with(y, ghost=f'{model.delay_s:g}:{model.reflection_coefficient:g}')

def deghost(trace: Trace, model: GhostModel, water_level: float=0.02) -> Trace:
    n = trace.samples.size
    f = np.fft.rfftfreq(n, trace.header.dt_s)
    response = 1.0 + model.reflection_coefficient * np.exp(-2j * np.pi * f * model.delay_s)
    power = np.abs(response) ** 2
    inverse = np.conj(response) / (power + water_level * max(float(np.max(power)), 1e-12))
    y = np.fft.irfft(np.fft.rfft(trace.samples) * inverse, n=n)
    return trace.copy_with(y, deghost=f'{model.delay_s:g}')

def ghost_notch_frequencies(model: GhostModel, max_hz: float) -> list[float]:
    if model.delay_s <= 0:
        return []
    fundamental = 1.0 / (2.0 * model.delay_s)
    out = []
    k = 1
    while k * fundamental <= max_hz:
        out.append(k * fundamental)
        k += 1
    return out
