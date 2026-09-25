from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class SourcePoint:
    source_id: str
    x: float
    y: float
    elevation_m: float = 0.0

@dataclass(frozen=True)
class ReceiverPoint:
    receiver_id: str
    x: float
    y: float
    elevation_m: float = 0.0

@dataclass(frozen=True)
class AcquisitionPair:
    source: SourcePoint
    receiver: ReceiverPoint

    @property
    def offset_m(self) -> float:
        return math.hypot(self.receiver.x - self.source.x, self.receiver.y - self.source.y)

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.source.x + self.receiver.x) / 2, (self.source.y + self.receiver.y) / 2)

def unique_points(pairs: list[AcquisitionPair]) -> tuple[dict[str, SourcePoint], dict[str, ReceiverPoint]]:
    sources = {}
    receivers = {}
    for pair in pairs:
        previous_source = sources.get(pair.source.source_id)
        if previous_source is not None and previous_source != pair.source:
            raise ValueError(f'source {pair.source.source_id} changed coordinates')
        previous_receiver = receivers.get(pair.receiver.receiver_id)
        if previous_receiver is not None and previous_receiver != pair.receiver:
            raise ValueError(f'receiver {pair.receiver.receiver_id} changed coordinates')
        sources[pair.source.source_id] = pair.source
        receivers[pair.receiver.receiver_id] = pair.receiver
    return (sources, receivers)

def maximum_offset(pairs: list[AcquisitionPair]) -> float:
    return max((p.offset_m for p in pairs), default=0.0)

def midpoint_cloud(pairs: list[AcquisitionPair]) -> list[tuple[float, float]]:
    return [p.midpoint for p in pairs]
