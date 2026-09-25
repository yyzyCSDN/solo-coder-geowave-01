from __future__ import annotations
import numpy as np
from geowave.model.traces import Trace, TraceHeader
from .cmp import Gather

def linear_stack(gather: Gather, normalize: bool=True) -> Trace:
    if not gather.traces:
        raise ValueError('empty gather')
    dt = gather.traces[0].header.dt_s
    n = min((t.samples.size for t in gather.traces))
    if any((abs(t.header.dt_s - dt) > 1e-12 for t in gather.traces)):
        raise ValueError('sampling mismatch')
    data = np.vstack([t.samples[:n] for t in gather.traces])
    y = np.sum(data, axis=0)
    if normalize:
        y /= len(gather.traces)
    h0 = gather.traces[0].header
    h = TraceHeader(f'stack:{gather.key}', gather.key, 'stack', gather.x or h0.source_x, gather.y or h0.source_y, gather.x or h0.receiver_x, gather.y or h0.receiver_y, dt, h0.start_s)
    return Trace(h, y, {'fold': str(len(gather.traces))})

def trimmed_stack(gather: Gather, trim_fraction: float=0.1) -> Trace:
    if not 0 <= trim_fraction < 0.5:
        raise ValueError('trim fraction')
    n = min((t.samples.size for t in gather.traces))
    data = np.sort(np.vstack([t.samples[:n] for t in gather.traces]), axis=0)
    k = int(len(gather.traces) * trim_fraction)
    core = data[k:len(gather.traces) - k or None]
    y = np.mean(core, axis=0)
    base = linear_stack(gather)
    return base.copy_with(y, stack=f'trimmed:{trim_fraction:g}')
