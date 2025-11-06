import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import rankdata

# ---------- Utilities ----------
def _finite_or_fill(x):
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).all():
        return x
    # Simple in-place finite fill (linear interpolation)
    t = np.arange(len(x))
    mask = np.isfinite(x)
    if not mask.any():
        return np.zeros_like(x)
    x[~mask] = np.interp(t[~mask], t[mask], x[mask])
    return x

def _robust_std(x, eps=1e-12):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + eps, med

def _safe_rank(values):
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() == 0:
        return np.array([]), mask
    v = values[mask]
    if v.size == 1:
        r = np.array([1.0], dtype=float)
    else:
        r = (rankdata(v, method="average") - 1) / (len(v) - 1)
    out = np.full_like(values, np.nan, dtype=float)
    out[mask] = r
    return out, mask

def _kernel(d, kind="gaussian", bw=3.0):
    d = np.asarray(d, dtype=float)
    if kind == "gaussian":
        return np.exp(-0.5 * (d / bw)**2)
    if kind == "laplace":
        return np.exp(-np.abs(d) / bw)
    if kind == "cauchy":
        return 1.0 / (1.0 + (d / bw)**2)
    # For FULL FIELDS do NOT use exp_causal (directional); default to gaussian
    return np.exp(-0.5 * (d / bw)**2)

def _build_field_safe(T, minima_idx, values, kind="gaussian", bw=3.0, eps=1e-12):
    """Normalized kernel regression; NaN-safe; falls back to global mean when weights ~ 0."""
    minima_idx = np.asarray(minima_idx, dtype=int)
    values = np.asarray(values, dtype=float)
    if minima_idx.size == 0 or values.size == 0:
        base = float(np.nanmean(values)) if values.size else 0.0
        return np.full(T, base, dtype=float)

    # Keep only in-bounds, finite values
    mask = (minima_idx >= 0) & (minima_idx < T) & np.isfinite(values)
    minima_idx = minima_idx[mask]
    values = values[mask]
    if minima_idx.size == 0:
        base = float(np.nanmean(values)) if values.size else 0.0
        return np.full(T, base, dtype=float)

    t = np.arange(T)[:, None]            # (T, 1)
    m = minima_idx[None, :]              # (1, M)
    D = np.abs(t - m).astype(float)      # (T, M)
    W = _kernel(D, kind=kind, bw=bw)     # (T, M)

    num = W @ values                     # (T,)
    den = W.sum(axis=1)                  # (T,)

    out = np.empty(T, dtype=float)
    zero = ~np.isfinite(den) | (den < eps)
    out[~zero] = num[~zero] / den[~zero]
    out[zero] = float(np.nanmean(values))  # fallback to global mean
    return out

def _weighted_stat_at_time(tp, minima_idx, values, kind="gaussian", bw=3.0, restrict="all", eps=1e-12):
    minima_idx = np.asarray(minima_idx, dtype=int)
    values = np.asarray(values, dtype=float)
    if minima_idx.size == 0 or values.size == 0:
        return np.nan
    if restrict == "pre":
        mask = minima_idx < tp
    elif restrict == "post":
        mask = minima_idx > tp
    else:
        mask = np.ones_like(minima_idx, dtype=bool)
    idx = minima_idx[mask]
    vals = values[mask]
    if idx.size == 0:
        return np.nan
    d = np.abs(tp - idx).astype(float)
    w = _kernel(d, kind=kind, bw=bw)
    wsum = np.sum(w)
    if not np.isfinite(wsum) or wsum < eps:
        return float(np.nanmean(vals))
    return float(np.sum(vals * w) / wsum)

# ---------- Main workflow ----------
def compute_valley_features(trace_1d, smooth_sigma=2.0, field_kernel="gaussian", field_bw=5.0,
                            peak_distance=3, eps=1e-12):
    # 1) Smooth & sanitize
    x = gaussian_filter1d(_finite_or_fill(trace_1d), sigma=smooth_sigma)

    # 2) Extrema
    med = np.median(x)
    rstd, med = _robust_std(x, eps=eps)
    peaks, _ = find_peaks(x, distance=max(2, peak_distance))  # keep it permissive
    minima, _ = find_peaks(-x)

    # ---- Diagnostics ----
    print(f"[diag] T={len(x)} peaks={len(peaks)} minima={len(minima)} "
          f"x[min,mean,max]=({np.min(x):.4f},{np.mean(x):.4f},{np.max(x):.4f}) rstd={rstd:.4g}")
    if len(minima) == 0:
        print("[diag] No minima found. Increase smoothing or use argrelextrema(-x, np.greater, order=1).")

    # 3) Depth S and Rank R
    S = -(x[minima] - med) / rstd if len(minima) else np.array([])
    # Keep only finite depths
    fin_mask = np.isfinite(S)
    minima = minima[fin_mask]
    S = S[fin_mask]

    R, rank_mask = _safe_rank(S)  # R is aligned to S; NaNs where S was non-finite
    minima = minima[rank_mask]
    S = S[rank_mask]
    R = R[rank_mask]

    print(f"[diag] valid minima after finite/rank checks: {len(minima)}; "
          f"S[min,mean,max]=({np.nanmin(S) if S.size else np.nan},"
          f"{np.nanmean(S) if S.size else np.nan},"
          f"{np.nanmax(S) if S.size else np.nan}); "
          f"R[min,mean,max]=({np.nanmin(R) if R.size else np.nan},"
          f"{np.nanmean(R) if R.size else np.nan},"
          f"{np.nanmax(R) if R.size else np.nan})")

    # 4) Build V(t), D(t) with a SYMMETRIC kernel (do NOT use exp_causal for full fields)
    T = len(x)
    V = _build_field_safe(T, minima, R, kind=field_kernel, bw=field_bw, eps=eps)
    D = _build_field_safe(T, minima, S, kind=field_kernel, bw=field_bw, eps=eps)

    print(f"[diag] V[min,mean,max]=({np.min(V):.4f},{np.mean(V):.4f},{np.max(V):.4f}) "
          f"D[min,mean,max]=({np.min(D):.4f},{np.mean(D):.4f},{np.max(D):.4f})")

    # 5) Per-peak features (use symmetric kernel for both pre/post unless you specifically want causal)
    rows = []
    for tp in np.atleast_1d(peaks):
        rows.append(dict(
            peak_index=int(tp),
            V_tp=float(V[tp]),
            V_pre=_weighted_stat_at_time(tp, minima, R, kind=field_kernel, bw=field_bw, restrict="pre", eps=eps),
            V_post=_weighted_stat_at_time(tp, minima, R, kind=field_kernel, bw=field_bw, restrict="post", eps=eps),
            D_tp=float(D[tp]),
            D_pre=_weighted_stat_at_time(tp, minima, S, kind=field_kernel, bw=field_bw, restrict="pre", eps=eps),
            D_post=_weighted_stat_at_time(tp, minima, S, kind=field_kernel, bw=field_bw, restrict="post", eps=eps),
        ))
    features = pd.DataFrame(rows).sort_values("peak_index").reset_index(drop=True)
    return V, D, features, peaks, minima

# ---------- Example usage ----------
if __name__ == "__main__":
    # Replace with your own load call

    import argparse
    import numpy as np
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--sigma", type=float, default=2)
    args = parser.parse_args()

    trace = np.load(args.trace)[args.row]
    

    V, D, feats, peaks, mins = compute_valley_features(
        trace,
        smooth_sigma=2.0,
        field_kernel="gaussian",  # 'gaussian' or 'laplace' or 'cauchy'
        field_bw=5.0,             # try 3, 5, 7, 10...
        peak_distance=3
    )
    print(feats.head())
