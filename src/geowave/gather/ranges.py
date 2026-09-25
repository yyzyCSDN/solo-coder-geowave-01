from __future__ import annotations
from dataclasses import dataclass
from geowave.gather.cmp import Gather

@dataclass(frozen=True)
class OffsetRanges:
    near: Gather
    mid: Gather
    far: Gather

def split_near_mid_far(gather: Gather, near_max_m: float, far_min_m: float) -> OffsetRanges:
    if near_max_m < 0 or far_min_m <= near_max_m:
        raise ValueError('invalid offset thresholds')
    near = []
    mid = []
    far = []
    for tr in gather.traces:
        off = tr.header.offset_m
        if off <= near_max_m:
            near.append(tr)
        elif off >= far_min_m:
            far.append(tr)
        else:
            mid.append(tr)
    return OffsetRanges(Gather(gather.key + ':near', near, gather.x, gather.y), Gather(gather.key + ':mid', mid, gather.x, gather.y), Gather(gather.key + ':far', far, gather.x, gather.y))

def range_balance(ranges: OffsetRanges) -> float:
    counts = [len(ranges.near.traces), len(ranges.mid.traces), len(ranges.far.traces)]
    total = sum(counts)
    if total == 0:
        return 0.0
    return min(counts) / (total / 3.0) if min(counts) else 0.0
