from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json

@dataclass(frozen=True)
class SurveyManifest:
    survey_id: str
    coordinate_reference: str
    vertical_datum: str
    sample_interval_s: float
    units: dict[str, str]
    source_files: tuple[str, ...]
    acquisition_date: str | None = None

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.survey_id.strip():
            issues.append('survey_id is empty')
        if self.sample_interval_s <= 0:
            issues.append('sample interval must be positive')
        if not self.coordinate_reference.strip():
            issues.append('coordinate reference missing')
        if self.units.get('distance') not in {'m', 'ft'}:
            issues.append('distance units must be m or ft')
        if not self.source_files:
            issues.append('no source files declared')
        return issues

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'), default=list).encode()
        return hashlib.sha256(raw).hexdigest()

def compare_manifests(a: SurveyManifest, b: SurveyManifest) -> dict[str, tuple[object, object]]:
    left = asdict(a)
    right = asdict(b)
    return {k: (left[k], right[k]) for k in left if left[k] != right[k]}
