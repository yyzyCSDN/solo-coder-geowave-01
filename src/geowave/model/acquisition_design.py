from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class LineDesign:
    source_spacing_m: float
    receiver_spacing_m: float
    receivers_per_shot: int
    maximum_offset_m: float
    line_length_m: float

@dataclass(frozen=True)
class DesignEstimate:
    source_count: int
    receiver_count: int
    nominal_fold: float
    midpoint_spacing_m: float
    maximum_incidence_angle_deg: float
    active_spread_m: float

def estimate_line_design(design: LineDesign, target_depth_m: float) -> DesignEstimate:
    if design.source_spacing_m <= 0 or design.receiver_spacing_m <= 0:
        raise ValueError('source and receiver spacing must be positive')
    if design.receivers_per_shot < 1:
        raise ValueError('receivers_per_shot must be positive')
    if design.maximum_offset_m <= 0 or design.line_length_m <= 0:
        raise ValueError('maximum offset and line length must be positive')
    source_count = int(math.floor(design.line_length_m / design.source_spacing_m)) + 1
    receiver_count = int(math.floor(design.line_length_m / design.receiver_spacing_m)) + 1
    midpoint_spacing = math.gcd(int(round(design.source_spacing_m * 1000)), int(round(design.receiver_spacing_m * 1000))) / 2000.0
    active_spread = min(design.maximum_offset_m * 2, design.receivers_per_shot * design.receiver_spacing_m)
    nominal_fold = active_spread / max(2 * design.source_spacing_m, 1e-12)
    angle = math.degrees(math.atan2(design.maximum_offset_m / 2, max(target_depth_m, 1e-09)))
    return DesignEstimate(source_count, receiver_count, nominal_fold, midpoint_spacing, angle, active_spread)

def offset_classes(design: LineDesign, classes: int=5) -> list[tuple[float, float]]:
    if classes < 1:
        raise ValueError('classes must be positive')
    width = design.maximum_offset_m / classes
    return [(i * width, (i + 1) * width) for i in range(classes)]

def fold_profile(design: LineDesign, midpoint_positions_m: list[float]) -> list[float]:
    estimate = estimate_line_design(design, target_depth_m=max(design.maximum_offset_m, 1.0))
    half_spread = estimate.active_spread_m / 2
    profile = []
    for midpoint in midpoint_positions_m:
        edge_distance = min(midpoint, max(0.0, design.line_length_m - midpoint))
        edge_factor = min(1.0, edge_distance / max(half_spread, 1e-12))
        profile.append(estimate.nominal_fold * edge_factor)
    return profile

def design_warnings(design: LineDesign, target_depth_m: float, dominant_frequency_hz: float, velocity_mps: float) -> list[str]:
    warnings = []
    estimate = estimate_line_design(design, target_depth_m)
    if estimate.maximum_incidence_angle_deg > 45:
        warnings.append('far offsets exceed 45 degree approximate incidence angle')
    wavelength = velocity_mps / dominant_frequency_hz if dominant_frequency_hz > 0 else float('inf')
    if design.receiver_spacing_m > wavelength / 2:
        warnings.append('receiver spacing risks spatial aliasing at dominant frequency')
    if design.source_spacing_m > wavelength:
        warnings.append('source spacing is coarse relative to dominant wavelength')
    if estimate.nominal_fold < 6:
        warnings.append('nominal fold is low for robust stacking')
    return warnings
