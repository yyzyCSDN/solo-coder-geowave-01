from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import math
from .traces import TraceSet

@dataclass(frozen=True)
class SurveyBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

@dataclass(frozen=True)
class MidpointBin:
    ix: int
    iy: int
    center_x: float
    center_y: float

class GeometryIndex:

    def __init__(self, dataset: TraceSet, bin_x_m: float=25.0, bin_y_m: float=25.0):
        if bin_x_m <= 0 or bin_y_m <= 0:
            raise ValueError('bin sizes must be positive')
        self.dataset = dataset
        self.bin_x_m = bin_x_m
        self.bin_y_m = bin_y_m
        xs = []
        ys = []
        for tr in dataset.traces:
            xs.extend([tr.header.source_x, tr.header.receiver_x])
            ys.extend([tr.header.source_y, tr.header.receiver_y])
        self.bounds = SurveyBounds(min(xs, default=0), max(xs, default=0), min(ys, default=0), max(ys, default=0))

    def bin_for(self, x: float, y: float) -> MidpointBin:
        ix = math.floor((x - self.bounds.min_x) / self.bin_x_m)
        iy = math.floor((y - self.bounds.min_y) / self.bin_y_m)
        cx = self.bounds.min_x + (ix + 0.5) * self.bin_x_m
        cy = self.bounds.min_y + (iy + 0.5) * self.bin_y_m
        return MidpointBin(ix, iy, cx, cy)

    def fold_map(self) -> dict[tuple[int, int], int]:
        fold = defaultdict(int)
        for tr in self.dataset.traces:
            b = self.bin_for(*tr.header.midpoint)
            fold[b.ix, b.iy] += 1
        return dict(fold)

    def azimuths_deg(self) -> list[float]:
        out = []
        for tr in self.dataset.traces:
            dx = tr.header.receiver_x - tr.header.source_x
            dy = tr.header.receiver_y - tr.header.source_y
            out.append((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)
        return out
