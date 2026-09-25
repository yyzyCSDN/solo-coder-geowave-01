from __future__ import annotations
from collections import defaultdict
from geowave.model.traces import TraceSet
from geowave.gather.cmp import Gather

def offset_gathers(dataset: TraceSet, edges_m: list[float]) -> list[Gather]:
    if len(edges_m) < 2:
        raise ValueError('need at least two offset edges')
    groups: dict[int, list] = defaultdict(list)
    for tr in dataset.traces:
        off = tr.header.offset_m
        for i, (a, b) in enumerate(zip(edges_m, edges_m[1:])):
            if a <= off < b or (i == len(edges_m) - 2 and off == b):
                groups[i].append(tr)
                break
    out = []
    for i in range(len(edges_m) - 1):
        traces = sorted(groups.get(i, []), key=lambda t: t.header.midpoint[0])
        out.append(Gather(f'offset:{edges_m[i]:g}-{edges_m[i + 1]:g}', traces))
    return out

def offset_coverage(dataset: TraceSet, edges_m: list[float]) -> dict[str, int]:
    return {g.key: len(g.traces) for g in offset_gathers(dataset, edges_m)}

def missing_offset_bins(dataset: TraceSet, edges_m: list[float]) -> list[str]:
    coverage = offset_coverage(dataset, edges_m)
    return [key for key, count in coverage.items() if count == 0]
