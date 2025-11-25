"""Update ROI features while preserving spike data and labels.

This script recomputes ROI-level features from existing trace data while preserving:
- Existing ROI labels (0/1)
- All spike data (features, windows, labels)
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import argparse
import numpy as np
from scipy.stats import skew
from scipy.signal import find_peaks, peak_prominences


def left_based_prominence(spike_prob: np.ndarray) -> tuple:
    """Compute mean and skew of left-based prominences."""
    peaks, _ = find_peaks(spike_prob)
    if len(peaks) == 0:
        return (0.0, 0.0, False)
    proms, left_bases, _ = peak_prominences(spike_prob, peaks)
    peak_vals = spike_prob[peaks]
    left_vals = spike_prob[left_bases]
    left_base_prominences = peak_vals - left_vals
    prom_mean = float(np.mean(left_base_prominences))
    prom_skew = float(skew(left_base_prominences)) if len(left_base_prominences) > 0 else 0.0
    return (prom_mean, prom_skew, True)
def derivative_skewness(smoothed_scaled_f: np.ndarray) -> tuple:
    """Compute skewness of the derivative."""
    derivative = np.diff(smoothed_scaled_f)
    if len(derivative) == 0:
        return (0.0, False, derivative)
    if np.any(np.isnan(derivative)) or np.any(np.isinf(derivative)):
        return (0.0, False, derivative)
    return (float(skew(derivative)), True, derivative)


def derivative_asymmetry(smoothed_scaled_f: np.ndarray) -> tuple:
    """
    Energy asymmetry between positive and negative derivatives.

    Real ROIs with upward transients tend to have pos_energy > neg_energy.
    Noise-like ROIs tend to have pos_energy ≈ neg_energy → asymmetry ≈ 1.
    """
    d = np.diff(smoothed_scaled_f)
    if d.size == 0:
        return (0.0, False)
    if np.any(~np.isfinite(d)):
        return (0.0, False)

    pos = np.abs(d[d > 0]).sum()
    neg = np.abs(d[d < 0]).sum()
    if pos == 0 and neg == 0:
        return (0.0, False)

    asym = pos / (neg + 1e-9)
    return (float(asym), True)


def rolling_variance_of_variance(
    x: np.ndarray,
    window: int = 30,
) -> tuple:
    """
    Variance of a rolling variance over the trace.

    Real ROIs: baseline + bursts → rolling variance changes over time → higher var_of_var.
    Noise ROIs: more stationary → lower var_of_var.
    """
    x = np.asarray(x, float)
    n = x.size
    if n < window * 2:
        # Too short for a meaningful rolling variance
        return (0.0, False)

    w = np.ones(window, float) / window
    mean = np.convolve(x, w, mode="valid")
    mean_sq = np.convolve(x * x, w, mode="valid")
    roll_var = np.maximum(mean_sq - mean * mean, 0.0)

    var_of_var = float(np.var(roll_var))
    if not np.isfinite(var_of_var):
        return (0.0, False)
    return (var_of_var, True)


def autocorr_decay(smoothed_scaled_f: np.ndarray, lag1: int = 1, lag2: int = 5) -> tuple:
    """
    Simple autocorrelation decay metric: rho(lag1) - rho(lag2).

    Real ROIs with slow kinetics keep correlation over multiple lags.
    Noise ROIs decorrelate quickly → smaller difference.
    """
    x = np.asarray(smoothed_scaled_f, float)
    n = x.size
    max_lag = max(lag1, lag2)
    if n <= max_lag:
        return (0.0, False)

    x = x - np.nanmean(x)
    if not np.all(np.isfinite(x)):
        return (0.0, False)

    def _rho(L):
        a = x[:-L]
        b = x[L:]
        num = np.dot(a, b)
        den = np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-12)
        if den == 0:
            return 0.0
        return float(num / den)

    rho1 = _rho(lag1)
    rho2 = _rho(lag2)
    ac_decay = rho1 - rho2
    if not np.isfinite(ac_decay):
        return (0.0, False)
    return (ac_decay, True)


def snr_estimate(smoothed_f_trace: np.ndarray) -> tuple:
    """
    Robust SNR estimate using MAD for noise and percentile spread for signal.

    Good ROIs: snr >> 1
    Bad/noise ROIs: snr ≈ 1
    """
    x = np.asarray(smoothed_f_trace, float)
    if x.size == 0:
        return (0.0, False)
    if not np.all(np.isfinite(x)):
        return (0.0, False)

    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    noise_level = 1.4826 * mad  # approx sigma for Gaussian

    # robust signal spread
    p95 = np.nanpercentile(x, 95)
    p20 = np.nanpercentile(x, 20)
    signal_level = p95 - p20

    if noise_level <= 0:
        return (0.0, False)

    snr = signal_level / (noise_level + 1e-9)
    if not np.isfinite(snr):
        return (0.0, False)
    return (float(snr), True)


def peak_density_and_prominence(smoothed_spike_prob: np.ndarray) -> tuple:
    """
    Trace-level peak density and median prominence of spike-probability peaks.

    Noise ROIs: lots of tiny peaks → high density, tiny median prominence.
    Good ROIs: fewer, stronger peaks.
    """
    x = np.asarray(smoothed_spike_prob, float)
    if x.size == 0:
        return (0.0, 0.0, False)

    # Simple, robust prominence threshold based on dynamic range
    dyn_range = float(np.nanmax(x) - np.nanmin(x))
    if not np.isfinite(dyn_range) or dyn_range <= 0:
        return (0.0, 0.0, False)

    prom_thresh = 0.05 * dyn_range
    peaks, _ = find_peaks(x, prominence=prom_thresh)

    if len(peaks) == 0:
        return (0.0, 0.0, False)

    proms, _, _ = peak_prominences(x, peaks)
    peak_density = len(peaks) / float(x.size)
    median_prom = float(np.median(proms))
    return (peak_density, median_prom, True)


def compute_roi_features(smoothed_f_trace: np.ndarray, smoothed_spike_prob: np.ndarray) -> tuple:
    """
    Extract ROI-level features from traces.

    Returns
    -------
    features : dict
        Scalar features per ROI (one row per ROI for your classifier).
    validity : dict
        Flags indicating which feature groups were computed cleanly.
    """
    # Derivative-based metrics
    deriv_skew, valid_deriv_skew, derivative = derivative_skewness(smoothed_f_trace)
    deriv_asym, valid_deriv_asym = derivative_asymmetry(smoothed_f_trace)

    # Existing left-based prominence metrics on spike-prob trace
    spike_prom_mean, spike_prom_skew, valid_prom = left_based_prominence(smoothed_spike_prob)

    # Rolling variance-of-variance on F trace
    var_of_var, valid_vov = rolling_variance_of_variance(smoothed_f_trace, window=30)

    # Autocorrelation decay
    ac_decay, valid_ac = autocorr_decay(smoothed_f_trace, lag1=1, lag2=5)

    # SNR estimate
    snr, valid_snr = snr_estimate(smoothed_f_trace)

    # Peak density & median prominence on spike-prob trace
    peak_density, median_spike_prom, valid_peak = peak_density_and_prominence(smoothed_spike_prob)

    # Simple range of F trace (already had this)
    trace_range = float(np.nanmax(smoothed_f_trace) - np.nanmin(smoothed_f_trace))

    features = {
        # Derivative-based
        "derivative_skew": float(deriv_skew),
        "derivative_asymmetry": float(deriv_asym),

        # Spike-prob prominence shape
        "spike_prom_mean": float(spike_prom_mean),
        "spike_prom_skew": float(spike_prom_skew),

        # Trace-level dynamics
        "range_trace": trace_range,
        "var_of_var": float(var_of_var),
        "ac_decay": float(ac_decay),

        # Global SNR-ish feature
        "snr_estimate": float(snr),

        # Spike-prob peak statistics
        "peak_density": float(peak_density),
        "median_spike_prom": float(median_spike_prom),
    }

    validity = {
        "valid_deriv_skew": bool(valid_deriv_skew),
        "valid_deriv_asym": bool(valid_deriv_asym),
        "valid_prom": bool(valid_prom),
        "valid_vov": bool(valid_vov),
        "valid_ac": bool(valid_ac),
        "valid_snr": bool(valid_snr),
        "valid_peak": bool(valid_peak),
    }

    return features, validity

def normalize_label_format(label_value):
    """
    Convert old label format (int) to new format (dict).
    
    Args:
        label_value: Either int (-1/0/1) or dict with 'value' and 'source' keys
    
    Returns:
        dict: {'value': -1/0/1, 'source': 'manual'/'classifier'/'unlabeled'}
    """
    # If already a dict, validate and return
    if isinstance(label_value, dict):
        if 'value' in label_value and 'source' in label_value:
            return label_value
        # Malformed dict - treat as unlabeled
        return {'value': -1, 'source': 'unlabeled'}
    
    # Convert old int format to new dict format
    # Since manual classifications were lost, treat all existing labels as auto-generated
    if label_value == -1:
        return {'value': -1, 'source': 'unlabeled'}
    elif label_value in [0, 1]:
        # Existing labeled ROIs are assumed to be auto-generated (from validity checks)
        return {'value': int(label_value), 'source': 'auto'}
    else:
        # Invalid value - treat as unlabeled
        return {'value': -1, 'source': 'unlabeled'}

def update_roi_features(roi_dict: dict) -> dict:
    """Update ROI features while preserving spike data and labels."""
    updated_dict = {}
    
    n_rois_processed = 0
    n_labels_preserved = 0
    n_spikes_preserved = 0
    
    for roi_key, roi_data in roi_dict.items():
        # Get existing traces
        smoothed_traces = roi_data.get('smoothed_traces', [])
        if len(smoothed_traces) < 2:
            print(f"Warning: ROI {roi_key} missing smoothed traces, skipping")
            continue
        
        smoothed_f_trace = np.asarray(smoothed_traces[0])
        smoothed_spike_prob = np.asarray(smoothed_traces[1])
        
        # Recompute ROI features
        features, validity = compute_roi_features(smoothed_f_trace, smoothed_spike_prob)
        
        # Normalize label to new dict format
        existing_label = roi_data.get('label', -1)
        label_dict = normalize_label_format(existing_label)
        
        # Only count as preserved if it was manually labeled
        if label_dict['value'] in [0, 1] and label_dict['source'] == 'manual':
            n_labels_preserved += 1
        
        # Preserve all spike data
        spikes = roi_data.get('spikes', {})
        if spikes:
            n_spikes_preserved += len(spikes)
        
        # Build updated ROI dict
        updated_dict[roi_key] = {
            'smoothed_traces': roi_data.get('smoothed_traces'),
            'raw_traces': roi_data.get('raw_traces'),
            'features': features,  # Updated
            'label': label_dict,  # New dict format
            'spikes': spikes  # Preserved entirely
        }
        
        n_rois_processed += 1
    
    print(f"\n✅ Updated {n_rois_processed} ROIs")
    print(f"  - Preserved {n_labels_preserved} manual ROI labels")
    print(f"  - Preserved {n_spikes_preserved} spikes")
    
    return updated_dict


def get_label_value(label):
    """Extract numeric label value from either dict or int format."""
    if isinstance(label, dict):
        return label.get('value', -1)
    return label


def get_label_source(label):
    """Extract label source from either dict or int format."""
    if isinstance(label, dict):
        return label.get('source', 'unknown')
    # Old format - assume manual if labeled, unlabeled if -1
    if label in [0, 1]:
        return 'manual'
    return 'unlabeled'


def main():
    """Update ROI features while preserving spike data and labels."""
    parser = argparse.ArgumentParser(description='Update ROI features while preserving spike/label data')
    parser.add_argument('--input_file', type=str, 
                       default='training_data/roi_filtering/all_roi_features.npy',
                       help='ROI data file to update')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file (default: overwrites input file)')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Create backup before overwriting')
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file) if args.output_file else input_path
    
    # Load existing ROI data
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    print(f"Loading ROI data from {input_path}...")
    roi_dict = np.load(input_path, allow_pickle=True).item()
    print(f"Found {len(roi_dict)} ROIs")
    
    # Detailed label breakdown (handle both old and new formats)
    n_good_rois = sum(1 for roi in roi_dict.values() if get_label_value(roi.get('label')) == 1)
    n_bad_rois = sum(1 for roi in roi_dict.values() if get_label_value(roi.get('label')) == 0)
    n_unlabeled_rois = sum(1 for roi in roi_dict.values() if get_label_value(roi.get('label')) == -1)
    n_labeled_rois = n_good_rois + n_bad_rois
    
    # Count by source
    n_manual_rois = sum(1 for roi in roi_dict.values() 
                        if get_label_source(roi.get('label')) == 'manual')
    n_classifier_rois = sum(1 for roi in roi_dict.values() 
                            if get_label_source(roi.get('label')) == 'classifier')
    
    print(f"\n📊 ROI Label Breakdown:")
    print(f"  - Good ROIs (label=1):       {n_good_rois:,}")
    print(f"  - Bad ROIs (label=0):        {n_bad_rois:,}")
    print(f"  - Unlabeled ROIs (label=-1): {n_unlabeled_rois:,}")
    print(f"  - Total labeled:             {n_labeled_rois:,}")
    print(f"\n  By Source:")
    print(f"  - Manual labels:             {n_manual_rois:,}")
    print(f"  - Classifier labels:         {n_classifier_rois:,}")
    
    # Count existing labels and spikes
    total_spikes = sum(len(roi.get('spikes', {})) for roi in roi_dict.values())
    n_good_spikes = sum(
        sum(1 for s in roi.get('spikes', {}).values() if get_label_value(s.get('label')) == 1)
        for roi in roi_dict.values()
    )
    n_bad_spikes = sum(
        sum(1 for s in roi.get('spikes', {}).values() if get_label_value(s.get('label')) == 0)
        for roi in roi_dict.values()
    )
    n_unlabeled_spikes = sum(
        sum(1 for s in roi.get('spikes', {}).values() if get_label_value(s.get('label')) == -1)
        for roi in roi_dict.values()
    )
    n_labeled_spikes = n_good_spikes + n_bad_spikes
    
    print(f"\n📊 Spike Label Breakdown:")
    print(f"  - Good spikes (label=1):     {n_good_spikes:,}")
    print(f"  - Bad spikes (label=0):      {n_bad_spikes:,}")
    print(f"  - Unlabeled spikes (label=-1): {n_unlabeled_spikes:,}")
    print(f"  - Total spikes:              {total_spikes:,}")
    print(f"  - Total labeled spikes:      {n_labeled_spikes:,}")
    
    # Create backup if requested
    if args.backup and output_path == input_path:
        import shutil
        from datetime import datetime
        backup_path = input_path.with_suffix(f'.backup_{datetime.now():%Y%m%d_%H%M%S}.npy')
        shutil.copy(input_path, backup_path)
        print(f"\n📦 Created backup: {backup_path}")
    
    # Update features
    print("\nRecomputing ROI features and normalizing label format...")
    updated_dict = update_roi_features(roi_dict)
    
    # Save updated data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, updated_dict)
    print(f"\n💾 Saved updated data to {output_path}")
    
    # Verify preservation with detailed breakdown
    final_n_good_rois = sum(1 for roi in updated_dict.values() if get_label_value(roi.get('label')) == 1)
    final_n_bad_rois = sum(1 for roi in updated_dict.values() if get_label_value(roi.get('label')) == 0)
    final_n_unlabeled_rois = sum(1 for roi in updated_dict.values() if get_label_value(roi.get('label')) == -1)
    final_n_labeled_rois = final_n_good_rois + final_n_bad_rois
    
    final_n_manual_rois = sum(1 for roi in updated_dict.values() 
                              if get_label_source(roi.get('label')) == 'manual')
    final_n_classifier_rois = sum(1 for roi in updated_dict.values() 
                                  if get_label_source(roi.get('label')) == 'classifier')
    
    final_total_spikes = sum(len(roi.get('spikes', {})) for roi in updated_dict.values())
    final_n_good_spikes = sum(
        sum(1 for s in roi.get('spikes', {}).values() if get_label_value(s.get('label')) == 1)
        for roi in updated_dict.values()
    )
    final_n_bad_spikes = sum(
        sum(1 for s in roi.get('spikes', {}).values() if get_label_value(s.get('label')) == 0)
        for roi in updated_dict.values()
    )
    final_n_labeled_spikes = final_n_good_spikes + final_n_bad_spikes
    
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"\nROI Labels:")
    print(f"  Good (1):     {n_good_rois:,} → {final_n_good_rois:,} {'✅' if n_good_rois == final_n_good_rois else '⚠️'}")
    print(f"  Bad (0):      {n_bad_rois:,} → {final_n_bad_rois:,} {'✅' if n_bad_rois == final_n_bad_rois else '⚠️'}")
    print(f"  Unlabeled:    {n_unlabeled_rois:,} → {final_n_unlabeled_rois:,}")
    print(f"  Total labeled: {n_labeled_rois:,} → {final_n_labeled_rois:,} {'✅' if n_labeled_rois == final_n_labeled_rois else '⚠️'}")
    print(f"\n  By Source:")
    print(f"  Manual:       {n_manual_rois:,} → {final_n_manual_rois:,} {'✅' if n_manual_rois == final_n_manual_rois else '⚠️'}")
    print(f"  Classifier:   {n_classifier_rois:,} → {final_n_classifier_rois:,}")
    
    print(f"\nSpike Labels:")
    print(f"  Good (1):     {n_good_spikes:,} → {final_n_good_spikes:,} {'✅' if n_good_spikes == final_n_good_spikes else '⚠️'}")
    print(f"  Bad (0):      {n_bad_spikes:,} → {final_n_bad_spikes:,} {'✅' if n_bad_spikes == final_n_bad_spikes else '⚠️'}")
    print(f"  Total spikes: {total_spikes:,} → {final_total_spikes:,} {'✅' if total_spikes == final_total_spikes else '⚠️'}")
    print(f"  Total labeled: {n_labeled_spikes:,} → {final_n_labeled_spikes:,} {'✅' if n_labeled_spikes == final_n_labeled_spikes else '⚠️'}")
    
    if (final_n_good_rois == n_good_rois and 
        final_n_bad_rois == n_bad_rois and
        final_total_spikes == total_spikes and 
        final_n_labeled_spikes == n_labeled_spikes):
        print("\n✅ All labels and spike data successfully preserved!")
        print("✅ Label format converted to dict structure")
    else:
        print("\n⚠️  Warning: Some data may have been lost. Check the backup file.")


if __name__ == '__main__':
    main()