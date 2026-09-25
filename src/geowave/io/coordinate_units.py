from __future__ import annotations
from dataclasses import dataclass
from geowave.model.traces import TraceHeader
from geowave.util.units import convert

@dataclass(frozen=True)
class CoordinateConvention:
    distance_unit: str
    elevation_unit: str
    coordinate_scale: float = 1.0

def normalize_header(header: TraceHeader, convention: CoordinateConvention) -> TraceHeader:
    sx = convert(header.source_x * convention.coordinate_scale, convention.distance_unit, 'm')
    sy = convert(header.source_y * convention.coordinate_scale, convention.distance_unit, 'm')
    rx = convert(header.receiver_x * convention.coordinate_scale, convention.distance_unit, 'm')
    ry = convert(header.receiver_y * convention.coordinate_scale, convention.distance_unit, 'm')
    elevation = header.elevation_m
    if elevation is not None:
        elevation = convert(elevation, convention.elevation_unit, 'm')
    return TraceHeader(trace_id=header.trace_id, shot_id=header.shot_id, receiver_id=header.receiver_id, source_x=sx, source_y=sy, receiver_x=rx, receiver_y=ry, dt_s=header.dt_s, start_s=header.start_s, line=header.line, inline=header.inline, crossline=header.crossline, elevation_m=elevation)

def coordinate_extent(headers: list[TraceHeader]) -> tuple[float, float, float, float]:
    if not headers:
        return (0.0, 0.0, 0.0, 0.0)
    xs = []
    ys = []
    for h in headers:
        xs.extend([h.source_x, h.receiver_x])
        ys.extend([h.source_y, h.receiver_y])
    return (min(xs), max(xs), min(ys), max(ys))
