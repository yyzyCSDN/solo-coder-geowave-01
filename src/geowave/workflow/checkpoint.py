from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import time

@dataclass(frozen=True)
class Checkpoint:
    run_id: str
    step_index: int
    input_digest: str
    output_digest: str
    recipe_digest: str
    created_at: float

    def token(self) -> str:
        body = json.dumps({'run_id': self.run_id, 'step_index': self.step_index, 'input_digest': self.input_digest, 'output_digest': self.output_digest, 'recipe_digest': self.recipe_digest, 'created_at': self.created_at}, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(body.encode()).hexdigest()

def new_checkpoint(run_id: str, step_index: int, input_digest: str, output_digest: str, recipe_digest: str) -> Checkpoint:
    return Checkpoint(run_id, step_index, input_digest, output_digest, recipe_digest, time.time())

def can_resume(checkpoint: Checkpoint, current_recipe_digest: str, available_input_digest: str) -> tuple[bool, str]:
    if checkpoint.recipe_digest != current_recipe_digest:
        return (False, 'recipe changed')
    if checkpoint.output_digest != available_input_digest:
        return (False, 'checkpoint output unavailable or changed')
    return (True, 'ok')
