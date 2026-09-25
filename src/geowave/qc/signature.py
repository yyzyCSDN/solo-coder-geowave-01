from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.signal.source_signature import SignatureEstimate

@dataclass(frozen=True)
class SignatureQC:
    peak_index: int
    sidelobe_ratio: float
    match_correlation: float
    stable: bool

def signature_qc(estimate: SignatureEstimate, sidelobe_limit: float=0.5, correlation_floor: float=0.7) -> SignatureQC:
    w = np.asarray(estimate.wavelet, float)
    if not w.size:
        return SignatureQC(0, 0.0, estimate.match_correlation, False)
    peak_index = int(np.argmax(np.abs(w)))
    peak = abs(float(w[peak_index]))
    mask = np.ones(len(w), dtype=bool)
    lo = max(0, peak_index - 1)
    hi = min(len(w), peak_index + 2)
    mask[lo:hi] = False
    sidelobe = float(np.max(np.abs(w[mask]))) if np.any(mask) else 0.0
    ratio = sidelobe / (peak + 1e-12)
    stable = ratio <= sidelobe_limit and estimate.match_correlation >= correlation_floor
    return SignatureQC(peak_index, ratio, estimate.match_correlation, stable)

def compare_signatures(a: SignatureEstimate, b: SignatureEstimate) -> dict[str, float]:
    n = min(len(a.wavelet), len(b.wavelet))
    x = np.asarray(a.wavelet[:n], float)
    y = np.asarray(b.wavelet[:n], float)
    corr = float(np.corrcoef(x, y)[0, 1]) if n > 1 and np.std(x) and np.std(y) else 0.0
    return {'wavelet_correlation': corr, 'delay_delta_samples': float(b.delay_samples - a.delay_samples), 'match_delta': float(b.match_correlation - a.match_correlation)}
