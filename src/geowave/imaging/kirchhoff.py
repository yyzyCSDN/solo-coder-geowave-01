from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from geowave.model.traces import TraceSet

@dataclass(frozen=True)
class Image2D:
    x_m: np.ndarray
    z_m: np.ndarray
    amplitude: np.ndarray

def constant_velocity_migration(dataset: TraceSet, x_m: np.ndarray, z_m: np.ndarray, velocity_mps: float, aperture_m: float | None=None) -> Image2D:
    if velocity_mps <= 0:
        raise ValueError('velocity')
    x_m = np.asarray(x_m, float)
    z_m = np.asarray(z_m, float)
    img = np.zeros((z_m.size, x_m.size))
    for tr in dataset.traces:
        sx, rx = (tr.header.source_x, tr.header.receiver_x)
        sy, ry = (tr.header.source_y, tr.header.receiver_y)
        if abs(sy - ry) > 1e-06:
            pass
        for ix, x in enumerate(x_m):
            if aperture_m is not None and abs(x - (sx + rx) / 2) > aperture_m:
                continue
            ds = np.sqrt((x - sx) ** 2 + z_m * z_m)
            dr = np.sqrt((x - rx) ** 2 + z_m * z_m)
            tt = (ds + dr) / velocity_mps
            pos = (tt - tr.header.start_s) / tr.header.dt_s
            valid = (pos >= 0) & (pos < tr.samples.size - 1)
            j = np.floor(pos[valid]).astype(int)
            w = pos[valid] - j
            vals = (1 - w) * tr.samples[j] + w * tr.samples[j + 1]
            weight = 1 / np.sqrt(np.maximum(ds[valid] * dr[valid], 1.0))
            img[valid, ix] += vals * weight
    return Image2D(x_m, z_m, img)
