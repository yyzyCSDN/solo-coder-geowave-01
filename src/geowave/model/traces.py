from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class TraceHeader:
    trace_id: str
    shot_id: str
    receiver_id: str
    source_x: float
    source_y: float
    receiver_x: float
    receiver_y: float
    dt_s: float
    start_s: float = 0.0
    line: str = ''
    inline: int | None = None
    crossline: int | None = None
    elevation_m: float | None = None

    @property
    def offset_m(self) -> float:
        return float(np.hypot(self.receiver_x - self.source_x, self.receiver_y - self.source_y))

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.source_x + self.receiver_x) / 2.0, (self.source_y + self.receiver_y) / 2.0)

@dataclass
class Trace:
    header: TraceHeader
    samples: np.ndarray
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.samples = np.asarray(self.samples, dtype=float)
        if self.samples.ndim != 1:
            raise ValueError('trace samples must be one-dimensional')
        if self.header.dt_s <= 0:
            raise ValueError('sample interval must be positive')

    @property
    def times(self) -> np.ndarray:
        return self.header.start_s + np.arange(self.samples.size) * self.header.dt_s

    @property
    def duration_s(self) -> float:
        return self.header.dt_s * max(0, self.samples.size - 1)

    def copy_with(self, samples: Iterable[float], **tags: str) -> 'Trace':
        merged = dict(self.tags)
        merged.update(tags)
        return Trace(self.header, np.asarray(list(samples), dtype=float), merged)

@dataclass
class TraceSet:
    traces: list[Trace]
    name: str = 'dataset'

    def validate_sampling(self, tol: float=1e-12) -> float:
        if not self.traces:
            return 0.0
        dt = self.traces[0].header.dt_s
        for tr in self.traces[1:]:
            if abs(tr.header.dt_s - dt) > tol:
                raise ValueError('mixed sample intervals')
        return dt

    def by_shot(self) -> dict[str, list[Trace]]:
        out: dict[str, list[Trace]] = {}
        for tr in self.traces:
            out.setdefault(tr.header.shot_id, []).append(tr)
        return out

    def by_receiver(self) -> dict[str, list[Trace]]:
        out: dict[str, list[Trace]] = {}
        for tr in self.traces:
            out.setdefault(tr.header.receiver_id, []).append(tr)
        return out

    def trace_ids(self) -> set[str]:
        return {t.header.trace_id for t in self.traces}
