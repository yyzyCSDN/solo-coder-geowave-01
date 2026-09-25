from __future__ import annotations
from dataclasses import asdict
import numpy as np
from geowave.service.state import ServiceState
from geowave.qc.pipeline import dataset_qc, quality_grade
from geowave.qc.acquisition import acquisition_qc
from geowave.qc.fold import fold_summary
from geowave.qc.spectrum import spectral_qc

def dataset_overview(state: ServiceState, dataset_id: str) -> dict[str, object]:
    dataset = state.datasets.get(dataset_id)
    if dataset is None:
        raise KeyError(dataset_id)
    report = dataset_qc(dataset)
    grade, reasons = quality_grade(report)
    acquisition = acquisition_qc(dataset)
    fold = fold_summary(dataset)
    spectrum = spectral_qc(dataset)
    durations = [tr.duration_s for tr in dataset.traces]
    offsets = [tr.header.offset_m for tr in dataset.traces]
    return {'dataset_id': dataset_id, 'name': dataset.name, 'trace_count': len(dataset.traces), 'grade': grade, 'reasons': reasons, 'acquisition': asdict(acquisition), 'fold': asdict(fold), 'spectrum': asdict(spectrum), 'duration_median_s': float(np.median(durations)) if durations else 0.0, 'offset_median_m': float(np.median(offsets)) if offsets else 0.0}

def list_dataset_overviews(state: ServiceState) -> list[dict[str, object]]:
    return [dataset_overview(state, dataset_id) for dataset_id in sorted(state.datasets)]
