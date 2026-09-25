from __future__ import annotations
import json
from geowave.qc.pipeline import quality_grade

def build_text_report(dataset_name: str, qc_report: dict[str, object], extra: dict[str, object] | None=None) -> str:
    grade, reasons = quality_grade(qc_report)
    lines = [f'GeoWave QC Report: {dataset_name}', f'Grade: {grade}']
    if reasons:
        lines.append('Reasons:')
        lines.extend((f'- {r}' for r in reasons))
    geom = qc_report.get('geometry', {})
    lines.append(f"Trace count: {geom.get('trace_count', 0)}")
    lines.append(f"Shot count: {geom.get('shot_count', 0)}")
    lines.append(f"Receiver count: {geom.get('receiver_count', 0)}")
    spec = qc_report.get('spectrum', {})
    lines.append(f"Spectral centroid: {spec.get('centroid_hz', 0):.2f} Hz")
    if extra:
        lines.append('Additional metrics:')
        for key in sorted(extra):
            lines.append(f'- {key}: {extra[key]}')
    return '\n'.join(lines)

def build_json_report(dataset_name: str, qc_report: dict[str, object], extra: dict[str, object] | None=None) -> str:
    grade, reasons = quality_grade(qc_report)
    payload = {'dataset': dataset_name, 'grade': grade, 'reasons': reasons, 'qc': qc_report, 'extra': extra or {}}
    return json.dumps(payload, sort_keys=True, indent=2, default=str)
