"""Spike class for detected events."""
from archive.utils.spike_utils import compute_spike_constants
import numpy as np
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter1d
from utils.feature_utils import (
    compute_decay_shape_features,
    compute_additional_decay_features,
    _create_large_window,
    _create_small_window
)
class Spike:
    """Represents a detected spike event."""
    
    def __init__(self,
                 sm_f_idx: int,
                 position_idx: int):
        """
        Initialize Spike.
        
        Parameters:
            frame_index: Frame index of F peak
            cascade_peak_idx: Frame index of cascade peak
            prob_height: Cascade probability at peak
            f_value: Fluorescence value at peak
        """
        self.f_index = None
        self.sm_f_idx = sm_f_idx
        self.prob_height = None
        self.f_value = None
        self.fluorescence_peak = None  # Alias for compatibility
        self.position_idx = position_idx
        # Features for classification (populated later)
        self.prominence = 0.0
        self.baseline_delta = 0.0
        self.window_width = 0.0
        self.window_auc = 0.0
        self.rise_slope = 0.0
        self.decay_tau = 5.0
        
        # Classification result
        self.is_valid = None

        # Feature parameters
        self.prev_position_idx = position_idx - 1 if position_idx > 1 else 0
        self.next_position_idx = position_idx + 1 
        self.f_small_window_sg = None
        self.f_small_window_smooth = None
        self.left_base = None
        self.right_base = None
        self.stats = {}
        
    def __repr__(self):
        return f"Spike(frame={self.frame_index}, prob={self.prob_height:.3f}, F={self.f_value:.2f})"
    
    def create_windows(self, norm_sg_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        large_window_tup = _create_large_window(
            norm_sg_f, self.sm_f_idx, self.left_base, self.right_base)
        small_window_tup = _create_small_window(norm_sg_f, self.sm_f_idx, self.prev_position_idx, self.next_position_idx)
        self.small_window_sg = small_window_tup[0]
        return large_window_tup, small_window_tup

    def get_features(self) -> dict:
        pass
    def get_statistics(self):
        """Compute spike statistics.
        Returns:
            dict: Dictionary of spike statistics
                {'rise_slope': float, 'decay_tau': float,
                'decay_r2': float, 'decay_residual_std': float,
                'decay_curvature': float, 'decay_biphasic_ratio': float,
                'decay_skew': float, 'decay_kurtosis': float,
                'decay_linearity': float}
        """
        rise_slope, decay_tau = compute_spike_constants(
            self.f_small_window_sg, np.argmax(self.f_small_window_sg))
        decay_shape = compute_decay_shape_features(
            self.f_small_window_sg, np.argmax(self.f_small_window_sg))
        decay_shape_features = compute_additional_decay_features(
            self.f_small_window_sg, np.argmax(self.f_small_window_sg))
        half_max_width = _half_max_width(self.f_small_window_sg, np.argmax(self.f_small_window_sg))
        self.stats = {
            'rise_slope': rise_slope,
            'decay_tau': decay_tau,
            **decay_shape,
            **decay_shape_features,
            'half_max_width': half_max_width
        }
        return self.stats
    
def _half_max_width(window: np.ndarray, peak_idx: int) -> float:
    """Compute half-max width of a spike transient."""
    segment = np.asarray(window, dtype=float)
    if segment.size == 0 or not np.isfinite(segment).all():
        return np.nan
    
    peak_value = float(np.nanmax(segment))
    half_max = peak_value / 2.0
    
    # Search left
    left_idx = peak_idx
    while left_idx > 0 and segment[left_idx] >= half_max:
        left_idx -= 1
    left_time = left_idx + (half_max - segment[left_idx]) / (segment[left_idx + 1] - segment[left_idx]) if left_idx < peak_idx else left_idx
    
    # Search right
    right_idx = peak_idx
    while right_idx < segment.size - 1 and segment[right_idx] >= half_max:
        right_idx += 1
    right_time = right_idx - (half_max - segment[right_idx]) / (segment[right_idx - 1] - segment[right_idx]) if right_idx > peak_idx else right_idx
    
    width = right_time - left_time
    return float(width)
def compute_spike_constants(
    window: np.ndarray,
    peak_idx_in_window: int,
    fs: float = 30.0,
    rise_fraction: float = 0.1,
    decay_fraction: float = 0.9,
    ) -> Tuple[float, float]:
    """
    Estimate rise slope and decay time constant for a single spike transient.
    
    The implementation avoids curve-fitting instability by:
    (1) normalizing the transient using the local baseline and peak amplitude,
    (2) fitting a simple linear model to the rising segment, and
    (3) computing the time to decay to 1/e of the peak using linear interpolation.
    
    Args:
        window: Spike probability window containing the spike transient
        peak_idx_in_window: Index of the peak within the window (relative to window start)
        fs: Sampling frequency in Hz (default: 30.0)
        rise_fraction: Fraction of peak amplitude to start rise fitting (default: 0.1)
        decay_fraction: Fraction of peak amplitude to start decay fitting (default: 0.9)
    
    Returns:
        Tuple of (rise_slope, decay_tau):
            - rise_slope: Linear slope of the rising phase (normalized units/second)
            - decay_tau: Decay time constant in seconds (time to reach 1/e of peak)
    """
    segment = np.asarray(window, dtype=float)
    
    # Validate input
    if segment.size < 3 or not np.isfinite(segment).all():
        return np.nan, np.nan
    
    # Normalize the segment
    baseline = float(np.nanmin(segment))
    peak_value = float(np.nanmax(segment))
    amplitude = peak_value - baseline
    
    if not np.isfinite(amplitude) or amplitude <= 1e-8:
        return np.nan, np.nan
    
    normed = (segment - baseline) / amplitude
    peak_rel = int(peak_idx_in_window)
    peak_rel = max(0, min(peak_rel, normed.size - 1))
    
    # ===== Rising slope via linear regression =====
    rise_segment = normed[:peak_rel + 1]
    if rise_segment.size < 2:
        rise_slope = np.nan
    else:
        t_rise = np.arange(rise_segment.size, dtype=float) / float(fs)
        try:
            slope, _ = np.polyfit(t_rise, rise_segment, 1)
        except Exception:
            slope = np.nan
        rise_slope = slope
    
    # ===== Decay constant estimated from time to reach 1/e of peak =====
    decay_segment = normed[peak_rel:]
    if decay_segment.size < 2:
        decay_tau = np.nan
    else:
        t_decay = np.arange(decay_segment.size, dtype=float) / float(fs)
        target = np.exp(-1.0)  # 1/e ≈ 0.368
        below = np.where(decay_segment <= target)[0]
        
        if below.size == 0:
            # Extrapolate using exponential fit if target not reached
            positive = decay_segment > 1e-6
            if np.count_nonzero(positive) >= 2:
                t_fit = t_decay[positive]
                y_fit = np.log(decay_segment[positive])
                try:
                    slope, _ = np.polyfit(t_fit, y_fit, 1)
                except Exception:
                    slope = np.nan
                
                if np.isfinite(slope) and slope < 0:
                    decay_tau = -1.0 / slope
                else:
                    decay_tau = np.nan
            else:
                decay_tau = np.nan
        else:
            # Interpolate to find exact crossing point
            idx = int(below[0])
            if idx == 0:
                decay_tau = 0.0
            else:
                y0 = decay_segment[idx - 1]
                y1 = decay_segment[idx]
                x0 = t_decay[idx - 1]
                x1 = t_decay[idx]
                
                if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
                    decay_tau = t_decay[idx]
                else:
                    frac = (target - y0) / (y1 - y0)
                    frac = np.clip(frac, 0.0, 1.0)
                    decay_tau = x0 + frac * (x1 - x0)
        
        # Fallback if decay_tau is still invalid
        if not np.isfinite(decay_tau):
            if decay_segment.size >= 2:
                decay_tau = t_decay[-1]
            else:
                decay_tau = 0.0
    
    # Final validation
    if not np.isfinite(rise_slope):
        rise_slope = np.nan
    
    return rise_slope, decay_tau

def compute_decay_shape_features(
    window: np.ndarray,
    peak_idx_in_window: int,
    fs: float = 30.0,
) -> dict:
    """
    Compute decay shape features beyond the time constant.
    
    Returns:
        dict with keys:
            - decay_r2: R² of exponential fit (goodness of fit)
            - decay_residual_std: Std of residuals (fit quality)
            - decay_curvature: Second derivative at decay midpoint
            - decay_biphasic: Evidence of two-phase decay
    """
    segment = np.asarray(window, dtype=float)
    baseline = float(np.nanmin(segment))
    peak_value = float(np.nanmax(segment))
    amplitude = peak_value - baseline
    
    if not np.isfinite(amplitude) or amplitude <= 1e-8:
        return {k: np.nan for k in ['decay_r2', 'decay_residual_std', 
                                      'decay_curvature', 'decay_biphasic']}
    
    normed = (segment - baseline) / amplitude
    peak_rel = max(0, min(peak_idx_in_window, normed.size - 1))
    
    # Decay segment
    decay_segment = normed[peak_rel:]
    if decay_segment.size < 3:
        return {k: np.nan for k in ['decay_r2', 'decay_residual_std', 
                                      'decay_curvature', 'decay_biphasic']}
    
    t_decay = np.arange(decay_segment.size, dtype=float) / float(fs)
    
    # 1. Exponential fit quality (R²)
    positive = decay_segment > 1e-6
    if np.count_nonzero(positive) >= 2:
        t_fit = t_decay[positive]
        y_fit = np.log(decay_segment[positive])
        try:
            slope, intercept = np.polyfit(t_fit, y_fit, 1)
            y_pred = slope * t_fit + intercept
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            residual_std = np.std(y_fit - y_pred)
        except:
            r2 = np.nan
            residual_std = np.nan
    else:
        r2 = np.nan
        residual_std = np.nan
    
    # 2. Curvature at decay midpoint (second derivative)
    if decay_segment.size >= 5:
        mid_idx = len(decay_segment) // 2
        # Use finite differences for second derivative
        if mid_idx > 1 and mid_idx < len(decay_segment) - 2:
            second_deriv = (decay_segment[mid_idx + 1] - 2 * decay_segment[mid_idx] + 
                           decay_segment[mid_idx - 1]) * (fs ** 2)
            curvature = float(second_deriv)
        else:
            curvature = np.nan
    else:
        curvature = np.nan
    
    # 3. Biphasic decay detection
    # Fit two exponentials and compare to single exponential
    if decay_segment.size >= 6 and np.count_nonzero(positive) >= 4:
        # Split decay into two halves
        mid = len(decay_segment) // 2
        decay_first = decay_segment[:mid][decay_segment[:mid] > 1e-6]
        decay_second = decay_segment[mid:][decay_segment[mid:] > 1e-6]
        
        if len(decay_first) >= 2 and len(decay_second) >= 2:
            try:
                # Fit each half
                t_first = np.arange(len(decay_first)) / fs
                t_second = np.arange(len(decay_second)) / fs
                
                slope1, _ = np.polyfit(t_first, np.log(decay_first), 1)
                slope2, _ = np.polyfit(t_second, np.log(decay_second), 1)
                
                # Ratio of decay rates (>2 suggests biphasic)
                tau1 = -1.0 / slope1 if slope1 < 0 else np.nan
                tau2 = -1.0 / slope2 if slope2 < 0 else np.nan
                
                if np.isfinite(tau1) and np.isfinite(tau2) and tau2 > 0:
                    biphasic_score = tau2 / tau1  # Slower phase / faster phase
                else:
                    biphasic_score = 1.0
            except:
                biphasic_score = np.nan
        else:
            biphasic_score = np.nan
    else:
        biphasic_score = np.nan
    
    return {
        'decay_r2': float(r2),
        'decay_residual_std': float(residual_std),
        'decay_curvature': float(curvature),
        'decay_biphasic_ratio': float(biphasic_score)
    }

def compute_additional_decay_features(
    window: np.ndarray,
    peak_idx_in_window: int,
) -> dict:
    """More decay shape characteristics."""
    segment = np.asarray(window, dtype=float)
    baseline = float(np.nanmin(segment))
    peak_value = float(np.nanmax(segment))
    amplitude = peak_value - baseline
    
    if not np.isfinite(amplitude) or amplitude <= 1e-8:
        return {k: np.nan for k in ['decay_skew', 'decay_kurtosis', 
                                      'decay_linearity']}
    
    normed = (segment - baseline) / amplitude
    peak_rel = max(0, min(peak_idx_in_window, normed.size - 1))
    decay_segment = normed[peak_rel:]
    
    if decay_segment.size < 3:
        return {k: np.nan for k in ['decay_skew', 'decay_kurtosis', 
                                      'decay_linearity']}
    
    from scipy.stats import skew, kurtosis
    
    # Skewness of decay (symmetry)
    decay_skew = float(skew(decay_segment))
    
    # Kurtosis (tail heaviness)
    decay_kurt = float(kurtosis(decay_segment))
    
    # Linearity in log space (deviation from exponential)
    positive = decay_segment > 1e-6
    if np.count_nonzero(positive) >= 3:
        log_decay = np.log(decay_segment[positive])
        # Fit line and measure deviation
        x = np.arange(len(log_decay))
        try:
            coeffs = np.polyfit(x, log_decay, 1)
            line_fit = np.polyval(coeffs, x)
            deviation = np.std(log_decay - line_fit)
            linearity = 1.0 / (1.0 + deviation)  # 1 = perfect line, 0 = nonlinear
        except:
            linearity = np.nan
    else:
        linearity = np.nan
    
    return {
        'decay_skew': float(decay_skew),
        'decay_kurtosis': float(decay_kurt),
        'decay_linearity': float(linearity)
    }