from __future__ import annotations
from dataclasses import dataclass
from geowave.workflow.executor import Executor, RunResult, dataset_digest
from geowave.workflow.recipe import Recipe
from geowave.model.traces import TraceSet

@dataclass(frozen=True)
class ReplayCheck:
    expected_input: str
    actual_input: str
    expected_output: str | None
    actual_output: str | None
    reproducible: bool
    reason: str

def replay(executor: Executor, dataset: TraceSet, recipe: Recipe, run_id: str, expected_input: str, expected_output: str | None=None) -> tuple[RunResult | None, ReplayCheck]:
    actual_input = dataset_digest(dataset)
    if actual_input != expected_input:
        return (None, ReplayCheck(expected_input, actual_input, expected_output, None, False, 'input digest changed'))
    result = executor.execute(dataset, recipe, run_id)
    if expected_output is None:
        return (result, ReplayCheck(expected_input, actual_input, None, result.output_digest, True, 'no prior output to compare'))
    ok = result.output_digest == expected_output
    return (result, ReplayCheck(expected_input, actual_input, expected_output, result.output_digest, ok, 'match' if ok else 'output digest changed'))
