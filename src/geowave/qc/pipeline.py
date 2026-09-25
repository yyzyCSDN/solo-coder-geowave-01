from __future__ import annotations
from dataclasses import asdict
from geowave.model.traces import TraceSet
from geowave.qc.geometry import geometry_qc, azimuth_gaps
from geowave.qc.fold import fold_summary
from geowave.qc.spectrum import spectral_qc
from geowave.qc.trace_stats import trace_stats, outlier_trace_ids
from geowave.qc.noise import spike_indices

def dataset_qc(dataset: TraceSet, bin_m: float=25.0) -> dict[str, object]:
    stats = [trace_stats(t) for t in dataset.traces]
    spike_counts = {t.header.trace_id: len(spike_indices(t)) for t in dataset.traces}
    geom = geometry_qc(dataset, bin_m)
    fold = fold_summary(dataset, bin_m)
    spec = spectral_qc(dataset)
    return {'geometry': asdict(geom), 'fold': asdict(fold), 'spectrum': asdict(spec), 'rms_outliers': outlier_trace_ids(stats), 'spike_counts': spike_counts, 'azimuth_gaps': azimuth_gaps(dataset)}

def quality_grade(report: dict[str, object]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    geom = report['geometry']
    fold = report['fold']
    if geom['trace_count'] == 0:
        return ('F', ['dataset has no traces'])
    if fold['mean_fold'] < 2:
        reasons.append('very low average fold')
    if fold['holes'] > 0:
        reasons.append('expected bins contain holes')
    if report['rms_outliers']:
        reasons.append('amplitude outliers present')
    if sum(report['spike_counts'].values()) > geom['trace_count']:
        reasons.append('frequent spike noise')
    grade = 'A' if not reasons else 'B' if len(reasons) == 1 else 'C' if len(reasons) == 2 else 'D'
    return (grade, reasons)
