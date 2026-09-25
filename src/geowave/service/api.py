from __future__ import annotations
import io
import json
import uuid
import numpy as np
from dataclasses import asdict
from fastapi import APIRouter, HTTPException
from geowave.model.traces import TraceHeader, Trace, TraceSet
from geowave.model.velocity import VelocityFunction, VelocityPick
from geowave.workflow.recipe import Recipe, StepSpec, validate_recipe
from geowave.workflow.executor import dataset_digest
from geowave.qc.pipeline import dataset_qc, quality_grade
from geowave.gather.cmp import cmp_gathers
from geowave.velocity.semblance import semblance_scan, pick_maxima
from geowave.service.schemas import DatasetIn, RecipeIn, VelocityFunctionIn
from geowave.service.state import ServiceState

def build_router(state: ServiceState) -> APIRouter:
    r = APIRouter()

    @r.get('/live')
    def live():
        return {'status': 'live'}

    @r.get('/ready')
    def ready():
        return {'status': 'ready', 'audit_valid': state.db.audit_valid()}

    @r.post('/v1/datasets')
    def create_dataset(payload: DatasetIn):
        traces = []
        for item in payload.traces:
            h = TraceHeader(**item.header.model_dump())
            traces.append(Trace(h, np.asarray(item.samples, float)))
        ds = TraceSet(traces, payload.name)
        ds.validate_sampling()
        digest = dataset_digest(ds)
        state.datasets[payload.dataset_id] = ds
        state.db.put_dataset(payload.dataset_id, payload.name, digest, {'trace_count': len(traces)})
        state.db.append_audit('dataset.created', {'dataset_id': payload.dataset_id, 'digest': digest})
        return {'dataset_id': payload.dataset_id, 'digest': digest, 'trace_count': len(traces)}

    @r.get('/v1/datasets/{dataset_id}/qc')
    def qc(dataset_id: str):
        ds = state.datasets.get(dataset_id)
        if ds is None:
            raise HTTPException(404, 'dataset not found')
        report = dataset_qc(ds)
        grade, reasons = quality_grade(report)
        return {'grade': grade, 'reasons': reasons, 'report': report}

    @r.get('/v1/datasets/{dataset_id}/cmp')
    def cmp_summary(dataset_id: str, bin_m: float=25.0):
        ds = state.datasets.get(dataset_id)
        if ds is None:
            raise HTTPException(404, 'dataset not found')
        gathers = cmp_gathers(ds, bin_m, bin_m)
        return [{'key': g.key, 'x': g.x, 'y': g.y, 'fold': len(g.traces), 'offsets': [t.header.offset_m for t in g.traces]} for g in gathers]

    @r.post('/v1/datasets/{dataset_id}/run')
    def run(dataset_id: str, payload: RecipeIn):
        ds = state.datasets.get(dataset_id)
        if ds is None:
            raise HTTPException(404, 'dataset not found')
        recipe = Recipe(payload.name, tuple((StepSpec(s.operation, dict(s.parameters)) for s in payload.steps)), payload.version)
        issues = validate_recipe(recipe, set(state.executor.operations))
        if issues:
            raise HTTPException(422, issues)
        run_id = uuid.uuid4().hex
        state.db.put_run(run_id, dataset_id, recipe.fingerprint(), 'running')
        result = state.executor.execute(ds, recipe, run_id)
        state.db.put_run(run_id, dataset_id, recipe.fingerprint(), 'complete', result.output_digest)
        state.db.append_audit('run.complete', {'run_id': run_id, 'output': result.output_digest, 'steps': len(recipe.steps)})
        return {'run_id': run_id, 'input_digest': result.input_digest, 'output_digest': result.output_digest, 'provenance_root': result.provenance.root()}

    @r.post('/v1/datasets/{dataset_id}/semblance')
    def semblance(dataset_id: str, velocity_min: float=1500, velocity_max: float=4500, velocity_step: float=100, bin_m: float=25):
        ds = state.datasets.get(dataset_id)
        if ds is None:
            raise HTTPException(404, 'dataset not found')
        gathers = cmp_gathers(ds, bin_m, bin_m)
        if not gathers:
            return []
        velocities = np.arange(velocity_min, velocity_max + velocity_step / 2, velocity_step)
        out = []
        for g in gathers:
            panel = semblance_scan(g, velocities)
            picks = pick_maxima(panel, 0.2)
            out.append({'gather': g.key, 'picks': picks[::max(1, len(picks) // 25 or 1)]})
        return out

    @r.post('/v1/velocity/validate')
    def validate_velocity(payload: VelocityFunctionIn):
        from geowave.qc.velocity import validate_velocity
        func = VelocityFunction(payload.x, payload.y, [VelocityPick(p.time_s, p.velocity_mps, p.confidence) for p in payload.picks])
        return [asdict(x) for x in validate_velocity(func)]

    @r.get('/v1/system/audit')
    def audit():
        return {'valid': state.db.audit_valid()}

    @r.get('/v1/system/operations')
    def operations():
        return {'operations': sorted(state.executor.operations)}
    return r
