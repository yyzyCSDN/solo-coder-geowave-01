from __future__ import annotations
from pydantic import BaseModel, Field

class TraceHeaderIn(BaseModel):
    trace_id: str
    shot_id: str
    receiver_id: str
    source_x: float
    source_y: float
    receiver_x: float
    receiver_y: float
    dt_s: float = Field(gt=0)
    start_s: float = 0.0

class TraceIn(BaseModel):
    header: TraceHeaderIn
    samples: list[float]

class DatasetIn(BaseModel):
    dataset_id: str
    name: str
    traces: list[TraceIn]

class StepIn(BaseModel):
    operation: str
    parameters: dict[str, object] = {}

class RecipeIn(BaseModel):
    name: str
    version: int = 1
    steps: list[StepIn]

class VelocityPickIn(BaseModel):
    time_s: float
    velocity_mps: float = Field(gt=0)
    confidence: float = Field(default=1.0, ge=0, le=1)

class VelocityFunctionIn(BaseModel):
    x: float
    y: float
    picks: list[VelocityPickIn]
