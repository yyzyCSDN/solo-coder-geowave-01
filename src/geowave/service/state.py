from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tempfile
from geowave.model.traces import TraceSet
from geowave.workflow.artifacts import ArtifactStore
from geowave.storage.sqlite import Database
from geowave.workflow.operations import build_default_executor

@dataclass
class ServiceState:
    datasets: dict[str, TraceSet]
    artifacts: ArtifactStore
    db: Database
    executor: object

def create_state(root: str | Path | None=None) -> ServiceState:
    base = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix='geowave-'))
    base.mkdir(parents=True, exist_ok=True)
    return ServiceState({}, ArtifactStore(base / 'artifacts'), Database(base / 'geowave.sqlite'), build_default_executor())
