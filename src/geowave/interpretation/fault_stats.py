from __future__ import annotations
from dataclasses import dataclass
import math
from geowave.interpretation.faults import FaultSegment

@dataclass(frozen=True)
class FaultNetworkStats:
    segment_count: int
    total_length_m: float
    mean_azimuth_deg: float
    intersection_count: int

def segment_azimuth(segment: FaultSegment) -> float:
    return (math.degrees(math.atan2(segment.x1 - segment.x0, segment.y1 - segment.y0)) + 360) % 180

def network_stats(segments: list[FaultSegment]) -> FaultNetworkStats:
    lengths = [math.hypot(s.x1 - s.x0, s.y1 - s.y0) for s in segments]
    az = [segment_azimuth(s) for s in segments]
    vectors = [complex(math.cos(math.radians(a * 2)), math.sin(math.radians(a * 2))) for a in az]
    mean_vec = sum(vectors, 0j) / len(vectors) if vectors else 0j
    mean_az = math.degrees(math.atan2(mean_vec.imag, mean_vec.real)) / 2 % 180 if vectors else 0.0
    intersections = 0
    from geowave.interpretation.faults import segment_intersection
    for i, a in enumerate(segments):
        for b in segments[i + 1:]:
            if segment_intersection(a, b) is not None:
                intersections += 1
    return FaultNetworkStats(len(segments), sum(lengths), mean_az, intersections)

def connected_components(segments: list[FaultSegment], tolerance_m: float=1.0) -> list[list[int]]:
    graph = {i: set() for i in range(len(segments))}

    def near(a, b):
        pts_a = ((a.x0, a.y0), (a.x1, a.y1))
        pts_b = ((b.x0, b.y0), (b.x1, b.y1))
        return min((math.hypot(x1 - x2, y1 - y2) for x1, y1 in pts_a for x2, y2 in pts_b)) <= tolerance_m
    for i, a in enumerate(segments):
        for j in range(i + 1, len(segments)):
            if near(a, segments[j]):
                graph[i].add(j)
                graph[j].add(i)
    components = []
    seen = set()
    for root in graph:
        if root in seen:
            continue
        stack = [root]
        comp = []
        seen.add(root)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components
