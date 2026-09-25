from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import Trace

@dataclass(frozen=True)
class PhaseConsistency:
    mean_resultant: float
    circular_std_deg: float
    trace_count: int

def phase_consistency(traces: list[Trace], frequency_hz: float, window: tuple[float, float] | None=None) -> PhaseConsistency:
    phases = []
    for tr in traces:
        x = tr.samples
        if window is not None:
            i0 = max(0, int((window[0] - tr.header.start_s) / tr.header.dt_s))
            i1 = min(len(x), int((window[1] - tr.header.start_s) / tr.header.dt_s))
            x = x[i0:i1]
        if len(x) < 4:
            continue
        z = np.fft.rfft(x * np.hanning(len(x)))
        f = np.fft.rfftfreq(len(x), tr.header.dt_s)
        idx = int(np.argmin(np.abs(f - frequency_hz)))
        phases.append(float(np.angle(z[idx])))
    if not phases:
        return PhaseConsistency(0.0, 180.0, 0)
    vector = np.mean(np.exp(1j * np.asarray(phases)))
    R = abs(vector)
    std = np.degrees(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12)))))
    return PhaseConsistency(float(R), float(std), len(phases))
