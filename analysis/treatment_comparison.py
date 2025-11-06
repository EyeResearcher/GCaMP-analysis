"""
Treatment comparison analysis for GCaMP data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import matplotlib.pyplot as plt
import seaborn as sns
from utils.stats_utils import (
    compute_cohen_d,
    compute_hedges_g,
    perform_permutation_test,
    compare_distributions,
    multiple_comparison_correction
)

if TYPE_CHECKING:
    from data_classes import Video

def compare_treatments(
    control_videos: List[Video],
    treatment_videos: List[Video],
    metric: str = 'spike_rate'
) -> pd.DataFrame:
    """
    Compare a metric between control and treatment groups.
    
    Parameters
    ----------
    control_videos : List[Video]
        Control condition videos
    treatment_videos : List[Video]
        Treatment condition videos
    metric : str
        Metric to compare ('spike_rate', 'burst_frequency', 'synchrony')
        
    Returns
    -------
    pd.DataFrame
        Comparison results per neuron/group
    """
    control_values = _extract_metric(control_videos, metric)
    treatment_values = _extract_metric(treatment_videos, metric)
    
    # Compute statistics
    t_stat, p_value = compare_distributions(control_values, treatment_values, test='ttest')
    u_stat, p_value_mw = compare_distributions(control_values, treatment_values, test='mannwhitneyu')
    cohen_d = compute_cohen_d(control_values, treatment_values)
    hedges_g = compute_hedges_g(control_values, treatment_values)
    perm_stat, p_value_perm = perform_permutation_test(control_values, treatment_values)
    
    results = pd.DataFrame([{
        'metric': metric,
        'control_mean': np.mean(control_values),
        'control_std': np.std(control_values),
        'control_n': len(control_values),
        'treatment_mean': np.mean(treatment_values),
        'treatment_std': np.std(treatment_values),
        'treatment_n': len(treatment_values),
        't_statistic': t_stat,
        'p_value_ttest': p_value,
        'u_statistic': u_stat,
        'p_value_mannwhitney': p_value_mw,
        'p_value_permutation': p_value_perm,
        'cohen_d': cohen_d,
        'hedges_g': hedges_g
    }])
    
    return results


def _extract_metric(videos: List[Video], metric: str) -> np.ndarray:
    """
    Extract a specific metric from a list of videos.
    
    Parameters
    ----------
    videos : List[Video]
        Videos to extract metric from
    metric : str
        Metric name
        
    Returns
    -------
    np.ndarray
        Array of metric values
    """
    values = []
    
    for video in videos:
        if metric == 'spike_rate':
            values.extend([n.get_spike_rate() for n in video.neurons])
            
        elif metric == 'burst_frequency':
            from .temporal_analysis import compute_burst_statistics
            for neuron in video.neurons:
                burst_stats = compute_burst_statistics([neuron])
                values.append(burst_stats['burst_frequency'])
                
        elif metric == 'synchrony':
            # Compute mean pairwise synchrony
            spike_trains = [n.binary_spike_train for n in video.neurons]
            if len(spike_trains) >= 2:
                sync_values = []
                for i in range(len(spike_trains)):
                    for j in range(i + 1, len(spike_trains)):
                        intersection = np.sum(spike_trains[i] & spike_trains[j])
                        union = np.sum(spike_trains[i] | spike_trains[j])
                        if union > 0:
                            sync_values.append(intersection / union)
                if sync_values:
                    values.append(np.mean(sync_values))
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return np.array(values)


def analyze_treatment_effects(
    control_videos: List[Video],
    treatment_videos: List[Video],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Analyze treatment effects across multiple metrics.
    
    Parameters
    ----------
    control_videos : List[Video]
        Control condition videos
    treatment_videos : List[Video]
        Treatment condition videos
    metrics : List[str], optional
        List of metrics to analyze
        
    Returns
    -------
    pd.DataFrame
        Combined results for all metrics
    """
    if metrics is None:
        metrics = ['spike_rate', 'burst_frequency', 'synchrony']
    
    results = []
    
    for metric in metrics:
        try:
            metric_results = compare_treatments(control_videos, treatment_videos, metric)
            results.append(metric_results)
        except Exception as e:
            print(f"Warning: Could not analyze metric '{metric}': {e}")
    
    if results:
        combined = pd.concat(results, ignore_index=True)
        
        # Apply multiple comparison correction
        p_values = combined['p_value_ttest'].values
        corrected_p, reject = multiple_comparison_correction(p_values, method='bonferroni')
        combined['p_value_corrected'] = corrected_p
        combined['significant'] = reject
        
        return combined
    
    return pd.DataFrame()


def compute_treatment_statistics(
    video: Video,
    baseline_period: Tuple[int, int],
    response_period: Tuple[int, int]
) -> pd.DataFrame:
    """
    Compute statistics comparing baseline and response periods within a video.
    
    Parameters
    ----------
    video : Video
        Video to analyze
    baseline_period : Tuple[int, int]
        (start_frame, end_frame) for baseline
    response_period : Tuple[int, int]
        (start_frame, end_frame) for response
        
    Returns
    -------
    pd.DataFrame
        Per-neuron statistics
    """
    results = []
    
    for neuron in video.neurons:
        # Extract spike counts in each period
        baseline_spikes = neuron.binary_spike_train[baseline_period[0]:baseline_period[1]]
        response_spikes = neuron.binary_spike_train[response_period[0]:response_period[1]]
        
        baseline_rate = np.sum(baseline_spikes) / (baseline_period[1] - baseline_period[0]) * video.frame_rate
        response_rate = np.sum(response_spikes) / (response_period[1] - response_period[0]) * video.frame_rate
        
        # Extract fluorescence
        baseline_fluor = neuron.roi.fluorescence[baseline_period[0]:baseline_period[1]]
        response_fluor = neuron.roi.fluorescence[response_period[0]:response_period[1]]
        
        # Statistical comparison
        t_stat, p_value = compare_distributions(baseline_fluor, response_fluor, test='ttest')
        effect_size = compute_cohen_d(baseline_fluor, response_fluor)
        
        results.append({
            'neuron_id': neuron.roi.roi_id,
            'baseline_spike_rate': baseline_rate,
            'response_spike_rate': response_rate,
            'spike_rate_change': response_rate - baseline_rate,
            'spike_rate_fold_change': response_rate / (baseline_rate + 1e-10),
            'baseline_fluor_mean': np.mean(baseline_fluor),
            'response_fluor_mean': np.mean(response_fluor),
            'fluor_change': np.mean(response_fluor) - np.mean(baseline_fluor),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohen_d': effect_size
        })
    
    df = pd.DataFrame(results)
    
    # Apply multiple comparison correction
    if len(df) > 0:
        corrected_p, reject = multiple_comparison_correction(df['p_value'].values)
        df['p_value_corrected'] = corrected_p
        df['significant'] = reject
    
    return df


def plot_treatment_comparison(
    control_videos: List[Video],
    treatment_videos: List[Video],
    metric: str = 'spike_rate',
    output_path: Optional[str] = None
):
    """
    Create visualization comparing control and treatment groups.
    
    Parameters
    ----------
    control_videos : List[Video]
        Control condition videos
    treatment_videos : List[Video]
        Treatment condition videos
    metric : str
        Metric to plot
    output_path : str, optional
        Path to save figure
    """
    # Extract data
    control_values = _extract_metric(control_videos, metric)
    treatment_values = _extract_metric(treatment_videos, metric)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Box plot
    ax = axes[0]
    data_to_plot = [control_values, treatment_values]
    ax.boxplot(data_to_plot, labels=['Control', 'Treatment'])
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Group Comparison')
    ax.grid(True, alpha=0.3)
    
    # Violin plot
    ax = axes[1]
    parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Control', 'Treatment'])
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Distribution Comparison')
    ax.grid(True, alpha=0.3)
    
    # Histogram
    ax = axes[2]
    ax.hist(control_values, bins=20, alpha=0.5, label='Control', density=True)
    ax.hist(treatment_values, bins=20, alpha=0.5, label='Treatment', density=True)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_ylabel('Density')
    ax.set_title('Distribution Overlap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def compute_responsive_neurons(
    video: Video,
    baseline_period: Tuple[int, int],
    response_period: Tuple[int, int],
    threshold: float = 2.0
) -> Tuple[List[int], pd.DataFrame]:
    """
    Identify neurons that respond significantly to stimulation.
    
    Parameters
    ----------
    video : Video
        Video to analyze
    baseline_period : Tuple[int, int]
        (start_frame, end_frame) for baseline
    response_period : Tuple[int, int]
        (start_frame, end_frame) for response
    threshold : float
        Z-score threshold for responsiveness
        
    Returns
    -------
    Tuple[List[int], pd.DataFrame]
        (list of responsive neuron IDs, DataFrame with details)
    """
    stats = compute_treatment_statistics(video, baseline_period, response_period)
    
    # Compute z-scores for spike rate changes
    if len(stats) > 0:
        mean_change = stats['spike_rate_change'].mean()
        std_change = stats['spike_rate_change'].std()
        
        if std_change > 0:
            stats['z_score'] = (stats['spike_rate_change'] - mean_change) / std_change
            stats['responsive'] = np.abs(stats['z_score']) > threshold
        else:
            stats['z_score'] = 0
            stats['responsive'] = False
    
    responsive_ids = stats[stats['responsive']]['neuron_id'].tolist()
    
    return responsive_ids, stats
