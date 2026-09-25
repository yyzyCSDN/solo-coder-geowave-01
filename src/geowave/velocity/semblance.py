from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from geowave.gather.cmp import Gather

@dataclass(frozen=True)
class SemblancePanel:
    times_s: np.ndarray
    velocities_mps: np.ndarray
    values: np.ndarray

def semblance_scan(gather: Gather, velocities_mps: np.ndarray, window_samples: int=5) -> SemblancePanel:
    if not gather.traces:
        raise ValueError('empty gather')
    dt = gather.traces[0].header.dt_s
    n = min((t.samples.size for t in gather.traces))
    t0 = gather.traces[0].header.start_s
    times = t0 + np.arange(n) * dt
    out = np.zeros((len(velocities_mps), n))
    fold = len(gather.traces)
    for iv, v in enumerate(velocities_mps):
        aligned = np.zeros((fold, n))
        for it, tr in enumerate(gather.traces):
            tn = np.sqrt(np.maximum(times, 0) ** 2 + (tr.header.offset_m / v) ** 2)
            src = (tn - t0) / dt
            aligned[it] = np.interp(src, np.arange(tr.samples.size), tr.samples, left=0, right=0)
        num = np.sum(aligned, axis=0) ** 2
        den = fold * np.sum(aligned * aligned, axis=0) + 1e-12
        sem = num / den
        if window_samples > 1:
            sem = np.convolve(sem, np.ones(window_samples) / window_samples, mode='same')
        out[iv] = sem
    return SemblancePanel(times, np.asarray(velocities_mps, float), out)

def pick_maxima(panel: SemblancePanel, min_value: float=0.15) -> list[tuple[float, float, float]]:
    picks = []
    for i, t in enumerate(panel.times_s):
        j = int(np.argmax(panel.values[:, i]))
        s = float(panel.values[j, i])
        if s >= min_value:
            picks.append((float(t), float(panel.velocities_mps[j]), s))
    return picks
