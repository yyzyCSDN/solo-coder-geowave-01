from __future__ import annotations
import html
from geowave.qc.pipeline import quality_grade

def render_qc_html(dataset_name: str, report: dict[str, object]) -> str:
    grade, reasons = quality_grade(report)
    rows = []
    geometry = report.get('geometry', {})
    for label, key in (('Traces', 'trace_count'), ('Shots', 'shot_count'), ('Receivers', 'receiver_count'), ('Minimum offset', 'min_offset_m'), ('Maximum offset', 'max_offset_m')):
        rows.append(f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(geometry.get(key, '')))}</td></tr>")
    reason_items = ''.join((f'<li>{html.escape(r)}</li>' for r in reasons)) or '<li>None</li>'
    return f"<!doctype html><html><head><meta charset='utf-8'><title>GeoWave QC</title></head><body><h1>{html.escape(dataset_name)}</h1><h2>Grade {html.escape(grade)}</h2><ul>{reason_items}</ul><table>{''.join(rows)}</table></body></html>"
