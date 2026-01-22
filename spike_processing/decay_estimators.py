from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Tuple
import numpy as np

from scipy.optimize import curve_fit
from utils.feature_utils import compute_spike_constants as compute_spike_constants_legacy

def _exp_offset(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    # tau is a *time constant* in seconds (NOT a rate)
    return A * np.exp(-t / tau) + C


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


class DecayEstimator(Protocol):
    name: str
    def estimate(self, window: np.ndarray, peak_idx_in_window: int, fs: float) -> Tuple[float, Dict[str, float]]:
        ...

@dataclass
class LegacyTimeTo1eDecayEstimator:
    """
    Current behavior: time to 1/e of normalized peak with fallback log-linear.
    Uses archive.utils.spike_utils.compute_spike_constants to preserve behavior.
    """
    name: str = "legacy_time_to_1e"

    def estimate(self, window: np.ndarray, peak_idx_in_window: int, fs: float) -> Tuple[float, Dict[str, float]]:
        rise_slope, tau = compute_spike_constants_legacy(window, peak_idx_in_window, fs=fs)
        return float(tau), {}
    
@dataclass
class ExpOffsetDecayEstimator:
    """
    Curve-fit decay to y(t) = A * exp(-t/tau) + C on the post-peak segment.

    Returns:
      tau_seconds, diagnostics
    """
    name: str = "exp_offset"

    # Fit controls
    min_points: int = 8                 # require enough samples to fit 3 params
    tail_fraction_for_C0: float = 0.2   # tail portion used to init offset C0
    max_tau_seconds: float = 30.0       # sanity cap (adjust to your recording)
    maxfev: int = 20000                 # curve_fit iterations

    def estimate(self, window: np.ndarray, peak_idx_in_window: int, fs: float) -> Tuple[float, Dict[str, float]]:
        seg = np.asarray(window, dtype=float)
        diag: Dict[str, float] = {}

        # Basic validation
        if seg.size < self.min_points or not np.isfinite(seg).all() or fs <= 0:
            return np.nan, {"ok": 0.0, "reason": 1.0}

        peak_rel = int(np.clip(int(peak_idx_in_window), 0, seg.size - 1))

        # Use post-peak decay segment
        y = seg[peak_rel:].copy()
        if y.size < self.min_points:
            return np.nan, {"ok": 0.0, "reason": 2.0}

        # Time in seconds starting at the peak
        t = np.arange(y.size, dtype=float) / float(fs)

        # Optional: enforce non-increasing trend a bit (helps fits on noisy traces)
        # (comment this out if you prefer pure raw fitting)
        # y = np.minimum.accumulate(y)

        # Initialize offset C0 from tail (robust to noise)
        tail_n = max(3, int(np.ceil(self.tail_fraction_for_C0 * y.size)))
        C0 = float(np.median(y[-tail_n:]))

        # Initialize amplitude A0 from (peak - offset)
        A0 = float(max(1e-8, y[0] - C0))

        # Initialize tau0 using a crude half-life or slope estimate
        # If we can find where it crosses halfway to C0, use that.
        half_level = C0 + 0.5 * A0
        below_half = np.where(y <= half_level)[0]
        if below_half.size > 0 and below_half[0] > 0:
            t_half = below_half[0] / float(fs)
            tau0 = max(1e-3, t_half / np.log(2.0))
        else:
            # fallback: small fraction of window length
            tau0 = max(1e-3, min(self.max_tau_seconds, (y.size / float(fs)) / 3.0))

        # Bounds:
        # A >= 0 (decay amplitude)
        # tau in (0, max_tau_seconds]
        # C between min and max of the segment (with a bit of slack)
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        slack = 0.1 * (y_max - y_min + 1e-8)

        lower = (0.0, 1e-4, y_min - slack)
        upper = (max(1e-8, y_max - y_min + slack), self.max_tau_seconds, y_max + slack)

        # If there’s basically no decay dynamic, bail early
        if (y_max - y_min) <= 1e-8:
            return np.nan, {"ok": 0.0, "reason": 3.0}

        try:
            popt, pcov = curve_fit(
                _exp_offset,
                t,
                y,
                p0=(A0, tau0, C0),
                bounds=(lower, upper),
                maxfev=self.maxfev,
            )
            A_hat, tau_hat, C_hat = [float(x) for x in popt]
            yhat = _exp_offset(t, A_hat, tau_hat, C_hat)

            diag.update(
                {
                    "ok": 1.0,
                    "A": A_hat,
                    "tau": tau_hat,
                    "C": C_hat,
                    "r2": float(_r2(y, yhat)),
                    "sse": float(np.sum((y - yhat) ** 2)),
                    "n": float(y.size),
                    "t_end": float(t[-1]),
                }
            )

            # Safety: if tau pins to bound or is tiny, mark as suspicious but still return it
            diag["tau_at_upper_bound"] = 1.0 if abs(tau_hat - self.max_tau_seconds) < 1e-6 else 0.0
            diag["tau_too_small"] = 1.0 if tau_hat < (1.0 / fs) else 0.0

            return tau_hat, diag

        except Exception:
            # Fit failure — return NaN with reason code
            return np.nan, {"ok": 0.0, "reason": 4.0}
