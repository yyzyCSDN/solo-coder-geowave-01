from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
from geowave.workflow.recipe import Recipe

@dataclass(frozen=True)
class AnalysisSnapshot:
    dataset_digest: str
    recipe_digest: str
    software_version: str
    parameters: dict[str, object]
    calibration_refs: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload['calibration_refs'] = list(self.calibration_refs)
        raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str).encode()
        return hashlib.sha256(raw).hexdigest()

def snapshot_for(dataset_digest: str, recipe: Recipe, software_version: str, parameters: dict[str, object], calibration_refs: list[str] | None=None) -> AnalysisSnapshot:
    return AnalysisSnapshot(dataset_digest, recipe.fingerprint(), software_version, dict(parameters), tuple(calibration_refs or ()))

def diff_snapshots(a: AnalysisSnapshot, b: AnalysisSnapshot) -> dict[str, object]:
    out: dict[str, object] = {}
    if a.dataset_digest != b.dataset_digest:
        out['dataset'] = (a.dataset_digest, b.dataset_digest)
    if a.recipe_digest != b.recipe_digest:
        out['recipe'] = (a.recipe_digest, b.recipe_digest)
    if a.software_version != b.software_version:
        out['software'] = (a.software_version, b.software_version)
    keys = set(a.parameters) | set(b.parameters)
    changes = {k: (a.parameters.get(k), b.parameters.get(k)) for k in keys if a.parameters.get(k) != b.parameters.get(k)}
    if changes:
        out['parameters'] = changes
    if a.calibration_refs != b.calibration_refs:
        out['calibrations'] = (a.calibration_refs, b.calibration_refs)
    return out
