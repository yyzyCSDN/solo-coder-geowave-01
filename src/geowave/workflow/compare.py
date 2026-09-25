from __future__ import annotations
from dataclasses import dataclass
from geowave.workflow.recipe import Recipe

@dataclass(frozen=True)
class RecipeChange:
    kind: str
    index: int
    before: object | None
    after: object | None

def compare_recipes(a: Recipe, b: Recipe) -> list[RecipeChange]:
    out = []
    n = max(len(a.steps), len(b.steps))
    for i in range(n):
        left = a.steps[i] if i < len(a.steps) else None
        right = b.steps[i] if i < len(b.steps) else None
        if left is None:
            out.append(RecipeChange('added', i, None, right))
        elif right is None:
            out.append(RecipeChange('removed', i, left, None))
        elif left.operation != right.operation:
            out.append(RecipeChange('operation', i, left.operation, right.operation))
        elif left.parameters != right.parameters:
            out.append(RecipeChange('parameters', i, left.parameters, right.parameters))
    if a.version != b.version:
        out.append(RecipeChange('version', -1, a.version, b.version))
    return out

def potentially_breaking(changes: list[RecipeChange]) -> bool:
    return any((c.kind in {'removed', 'operation'} for c in changes))
