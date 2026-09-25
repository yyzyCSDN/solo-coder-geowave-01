from __future__ import annotations
from dataclasses import dataclass
from geowave.model.acquisition_design import LineDesign, estimate_line_design, design_warnings

@dataclass(frozen=True)
class DesignQC:
    nominal_fold: float
    source_count: int
    receiver_count: int
    midpoint_spacing_m: float
    max_incidence_angle_deg: float
    warning_count: int
    acceptable: bool

def evaluate_design(design: LineDesign, target_depth_m: float, dominant_frequency_hz: float, velocity_mps: float, minimum_fold: float=6.0) -> DesignQC:
    estimate = estimate_line_design(design, target_depth_m)
    warnings = design_warnings(design, target_depth_m, dominant_frequency_hz, velocity_mps)
    acceptable = estimate.nominal_fold >= minimum_fold and estimate.maximum_incidence_angle_deg <= 60 and (len(warnings) <= 2)
    return DesignQC(nominal_fold=estimate.nominal_fold, source_count=estimate.source_count, receiver_count=estimate.receiver_count, midpoint_spacing_m=estimate.midpoint_spacing_m, max_incidence_angle_deg=estimate.maximum_incidence_angle_deg, warning_count=len(warnings), acceptable=acceptable)

def compare_designs(left: LineDesign, right: LineDesign, target_depth_m: float, dominant_frequency_hz: float, velocity_mps: float) -> dict[str, object]:
    a = evaluate_design(left, target_depth_m, dominant_frequency_hz, velocity_mps)
    b = evaluate_design(right, target_depth_m, dominant_frequency_hz, velocity_mps)
    return {'fold_delta': b.nominal_fold - a.nominal_fold, 'source_count_delta': b.source_count - a.source_count, 'receiver_count_delta': b.receiver_count - a.receiver_count, 'midpoint_spacing_delta_m': b.midpoint_spacing_m - a.midpoint_spacing_m, 'incidence_angle_delta_deg': b.max_incidence_angle_deg - a.max_incidence_angle_deg, 'warning_delta': b.warning_count - a.warning_count, 'acceptance_changed': a.acceptable != b.acceptable}
