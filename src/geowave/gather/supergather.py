from __future__ import annotations
from dataclasses import dataclass
from geowave.gather.cmp import Gather

@dataclass(frozen=True)
class SuperGather:
    center_key: str
    member_keys: tuple[str, ...]
    gather: Gather

def merge_neighboring_gathers(gathers: list[Gather], radius: int=1) -> list[SuperGather]:
    ordered = sorted(gathers, key=lambda g: g.key)
    out = []
    for i, center in enumerate(ordered):
        lo = max(0, i - radius)
        hi = min(len(ordered), i + radius + 1)
        neighbors = ordered[lo:hi]
        traces = []
        for g in neighbors:
            traces.extend(g.traces)
        merged = Gather('super:' + center.key, sorted(traces, key=lambda t: t.header.offset_m), center.x, center.y)
        out.append(SuperGather(center.key, tuple((g.key for g in neighbors)), merged))
    return out

def offset_distribution(gather: Gather, bins: list[float]) -> list[int]:
    counts = [0 for _ in range(max(0, len(bins) - 1))]
    for tr in gather.traces:
        off = tr.header.offset_m
        for i, (a, b) in enumerate(zip(bins, bins[1:])):
            if a <= off < b or (i == len(counts) - 1 and off == b):
                counts[i] += 1
                break
    return counts
