from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from geowave.model.geometry import GeometryIndex
from geowave.model.traces import Trace, TraceSet

@dataclass
class Gather:
    key: str
    traces: list[Trace]
    x: float | None = None
    y: float | None = None

def cmp_gathers(dataset: TraceSet, bin_x_m: float=25, bin_y_m: float=25) -> list[Gather]:
    geom = GeometryIndex(dataset, bin_x_m, bin_y_m)
    groups = defaultdict(list)
    centers = {}
    for tr in dataset.traces:
        b = geom.bin_for(*tr.header.midpoint)
        k = (b.ix, b.iy)
        groups[k].append(tr)
        centers[k] = (b.center_x, b.center_y)
    out = []
    for k, traces in groups.items():
        x, y = centers[k]
        out.append(Gather(f'cmp:{k[0]}:{k[1]}', sorted(traces, key=lambda t: t.header.offset_m), x, y))
    return sorted(out, key=lambda g: g.key)

def shot_gathers(dataset: TraceSet) -> list[Gather]:
    return [Gather(f'shot:{k}', sorted(v, key=lambda t: t.header.receiver_id)) for k, v in sorted(dataset.by_shot().items())]

def receiver_gathers(dataset: TraceSet) -> list[Gather]:
    return [Gather(f'receiver:{k}', sorted(v, key=lambda t: t.header.shot_id)) for k, v in sorted(dataset.by_receiver().items())]
