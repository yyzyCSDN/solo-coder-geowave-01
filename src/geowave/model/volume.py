from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class SeismicVolume:
    inline: np.ndarray
    crossline: np.ndarray
    samples: np.ndarray
    data: np.ndarray
    domain: str = 'time'

    def __post_init__(self):
        self.inline = np.asarray(self.inline)
        self.crossline = np.asarray(self.crossline)
        self.samples = np.asarray(self.samples, float)
        self.data = np.asarray(self.data, float)
        expected = (len(self.inline), len(self.crossline), len(self.samples))
        if self.data.shape != expected:
            raise ValueError(f'volume shape {self.data.shape} does not match {expected}')
        if self.domain not in {'time', 'depth'}:
            raise ValueError('domain must be time or depth')

    def inline_slice(self, value: int) -> np.ndarray:
        idx = int(np.argmin(np.abs(self.inline - value)))
        return self.data[idx]

    def crossline_slice(self, value: int) -> np.ndarray:
        idx = int(np.argmin(np.abs(self.crossline - value)))
        return self.data[:, idx, :]

    def sample_slice(self, coordinate: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(self.samples - coordinate)))
        return self.data[:, :, idx]

    def percentile_clip(self, low: float=1.0, high: float=99.0) -> 'SeismicVolume':
        lo, hi = np.nanpercentile(self.data, [low, high])
        clipped = np.clip(self.data, lo, hi)
        return SeismicVolume(self.inline.copy(), self.crossline.copy(), self.samples.copy(), clipped, self.domain)

def volume_stats(volume: SeismicVolume) -> dict[str, float]:
    a = volume.data
    finite = np.isfinite(a)
    if not finite.any():
        return {'mean': 0.0, 'std': 0.0, 'rms': 0.0, 'finite_fraction': 0.0}
    values = a[finite]
    return {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'rms': float(np.sqrt(np.mean(values * values))), 'finite_fraction': float(np.mean(finite))}
