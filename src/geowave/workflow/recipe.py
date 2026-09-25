from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json

@dataclass(frozen=True)
class StepSpec:
    operation: str
    parameters: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class Recipe:
    name: str
    steps: tuple[StepSpec, ...]
    version: int = 1

    def fingerprint(self) -> str:
        payload = {'name': self.name, 'version': self.version, 'steps': [{'operation': s.operation, 'parameters': s.parameters} for s in self.steps]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def validate_recipe(recipe: Recipe, known_operations: set[str]) -> list[str]:
    issues = []
    for i, s in enumerate(recipe.steps):
        if s.operation not in known_operations:
            issues.append(f'step {i}: unknown operation {s.operation}')
        if not isinstance(s.parameters, dict):
            issues.append(f'step {i}: parameters must be object')
    return issues
