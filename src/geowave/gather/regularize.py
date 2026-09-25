from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.gather.cmp import Gather
from geowave.model.traces import Trace, TraceHeader

@dataclass(frozen=True)
class RegularizationResult:
    gather: Gather
    interpolated_offsets: tuple[float, ...]

def regularize_offsets(gather: Gather, target_offsets_m: np.ndarray) -> RegularizationResult:
    if not gather.traces:
        return RegularizationResult(Gather(gather.key, [], gather.x, gather.y), ())
    target = np.asarray(target_offsets_m, float)
    existing = np.asarray([t.header.offset_m for t in gather.traces], float)
    n = min((t.samples.size for t in gather.traces))
    data = np.vstack([t.samples[:n] for t in gather.traces])
    order = np.argsort(existing)
    existing = existing[order]
    data = data[order]
    out = []
    interpolated = []
    base = gather.traces[0].header
    for k, off in enumerate(target):
        samples = np.asarray([np.interp(off, existing, data[:, j]) for j in range(n)])
        sx = (gather.x or 0.0) - off / 2
        rx = (gather.x or 0.0) + off / 2
        header = TraceHeader(f'{gather.key}:reg:{k}', gather.key, f'reg-{k}', sx, base.source_y, rx, base.receiver_y, base.dt_s, base.start_s)
        out.append(Trace(header, samples, {'regularized': 'true'}))
        if not np.any(np.isclose(existing, off)):
            interpolated.append(float(off))
    return RegularizationResult(Gather(gather.key + ':regular', out, gather.x, gather.y), tuple(interpolated))
