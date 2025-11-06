"""
Group analysis functions for neuron grouping results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

if TYPE_CHECKING:
    from data_classes import NeuronGroup

def analyze_group_stability(
    groups_list: List[List[NeuronGroup]],
    method: str = 'ari'
) -> pd.DataFrame:
    """
    Analyze stability of groupings across multiple sessions or conditions.
    
    Parameters
    ----------
    groups_list : List[List[NeuronGroup]]
        List of grouping results to compare
    method : str
        Similarity metric ('ari' for Adjusted Rand Index, 'nmi' for Normalized Mutual Information)
        
    Returns
    -------
    pd.DataFrame
        Pairwise similarity matrix
    """
    n_sessions = len(groups_list)
    similarity_matrix = np.zeros((n_sessions, n_sessions))
    
    for i in range(n_sessions):
        for j in range(i, n_sessions):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = _compare_groupings(groups_list[i], groups_list[j], method)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    df = pd.DataFrame(
        similarity_matrix,
        index=[f'Session_{i}' for i in range(n_sessions)],
        columns=[f'Session_{i}' for i in range(n_sessions)]
    )
    
    return df


def _compare_groupings(
    groups1: List[NeuronGroup],
    groups2: List[NeuronGroup],
    method: str = 'ari'
) -> float:
    """
    Compare two grouping results using similarity metrics.
    
    Parameters
    ----------
    groups1 : List[NeuronGroup]
        First grouping
    groups2 : List[NeuronGroup]
        Second grouping
    method : str
        Similarity metric
        
    Returns
    -------
    float
        Similarity score
    """
    # Create label arrays
    all_neuron_ids = set()
    for g in groups1:
        all_neuron_ids.update([n.roi.roi_id for n in g.neurons])
    for g in groups2:
        all_neuron_ids.update([n.roi.roi_id for n in g.neurons])
    
    neuron_ids = sorted(list(all_neuron_ids))
    n_neurons = len(neuron_ids)
    
    labels1 = np.full(n_neurons, -1, dtype=int)
    labels2 = np.full(n_neurons, -1, dtype=int)
    
    # Assign labels for first grouping
    for group_idx, group in enumerate(groups1):
        for neuron in group.neurons:
            idx = neuron_ids.index(neuron.roi.roi_id)
            labels1[idx] = group_idx
    
    # Assign labels for second grouping
    for group_idx, group in enumerate(groups2):
        for neuron in group.neurons:
            idx = neuron_ids.index(neuron.roi.roi_id)
            labels2[idx] = group_idx
    
    # Compute similarity
    if method == 'ari':
        return adjusted_rand_score(labels1, labels2)
    elif method == 'nmi':
        return normalized_mutual_info_score(labels1, labels2)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_group_coherence(
    group: NeuronGroup,
    method: str = 'cross_correlation'
) -> Dict[str, float]:
    """
    Compute coherence metrics for a neuron group.
    
    Parameters
    ----------
    group : NeuronGroup
        Neuron group to analyze
    method : str
        Coherence metric ('cross_correlation', 'synchrony')
        
    Returns
    -------
    Dict[str, float]
        Dictionary of coherence metrics
    """
    if len(group.neurons) < 2:
        return {
            'mean_coherence': 0.0,
            'std_coherence': 0.0,
            'median_coherence': 0.0,
            'min_coherence': 0.0,
            'max_coherence': 0.0
        }
    
    n_neurons = len(group.neurons)
    coherences = []
    
    if method == 'cross_correlation':
        # Compute pairwise cross-correlations
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                trace1 = group.neurons[i].roi.fluorescence
                trace2 = group.neurons[j].roi.fluorescence
                
                # Normalize traces
                trace1 = (trace1 - np.mean(trace1)) / (np.std(trace1) + 1e-10)
                trace2 = (trace2 - np.mean(trace2)) / (np.std(trace2) + 1e-10)
                
                # Compute zero-lag correlation
                corr = np.corrcoef(trace1, trace2)[0, 1]
                coherences.append(corr)
                
    elif method == 'synchrony':
        # Compute spike synchrony
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                spikes1 = group.neurons[i].binary_spike_train
                spikes2 = group.neurons[j].binary_spike_train
                
                # Compute Jaccard index
                intersection = np.sum(spikes1 & spikes2)
                union = np.sum(spikes1 | spikes2)
                
                if union > 0:
                    synchrony = intersection / union
                else:
                    synchrony = 0.0
                
                coherences.append(synchrony)
    
    coherences = np.array(coherences)
    
    return {
        'mean_coherence': np.mean(coherences),
        'std_coherence': np.std(coherences),
        'median_coherence': np.median(coherences),
        'min_coherence': np.min(coherences),
        'max_coherence': np.max(coherences)
    }


def analyze_group_dynamics(
    group: NeuronGroup,
    window_size: int = 100,
    overlap: int = 50
) -> pd.DataFrame:
    """
    Analyze temporal dynamics of group coherence using sliding windows.
    
    Parameters
    ----------
    group : NeuronGroup
        Neuron group to analyze
    window_size : int
        Size of sliding window in frames
    overlap : int
        Overlap between windows in frames
        
    Returns
    -------
    pd.DataFrame
        Time series of coherence metrics
    """
    if len(group.neurons) < 2:
        return pd.DataFrame()
    
    # Get fluorescence traces
    traces = np.array([n.roi.fluorescence for n in group.neurons])
    n_neurons, n_frames = traces.shape
    
    # Sliding window analysis
    step = window_size - overlap
    n_windows = (n_frames - window_size) // step + 1
    
    results = []
    
    for win_idx in range(n_windows):
        start = win_idx * step
        end = start + window_size
        
        window_traces = traces[:, start:end]
        
        # Compute pairwise correlations
        coherences = []
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                trace1 = window_traces[i]
                trace2 = window_traces[j]
                
                # Normalize
                trace1 = (trace1 - np.mean(trace1)) / (np.std(trace1) + 1e-10)
                trace2 = (trace2 - np.mean(trace2)) / (np.std(trace2) + 1e-10)
                
                corr = np.corrcoef(trace1, trace2)[0, 1]
                coherences.append(corr)
        
        results.append({
            'window_idx': win_idx,
            'start_frame': start,
            'end_frame': end,
            'mean_coherence': np.mean(coherences),
            'std_coherence': np.std(coherences),
            'median_coherence': np.median(coherences)
        })
    
    return pd.DataFrame(results)


def compare_group_methods(
    sttc_groups: List[NeuronGroup],
    dtw_groups: List[NeuronGroup]
) -> Dict[str, any]:
    """
    Compare STTC and DTW grouping methods.
    
    Parameters
    ----------
    sttc_groups : List[NeuronGroup]
        Groups from STTC method
    dtw_groups : List[NeuronGroup]
        Groups from DTW method
        
    Returns
    -------
    Dict[str, any]
        Comparison metrics
    """
    # Compute similarity between methods
    ari = _compare_groupings(sttc_groups, dtw_groups, method='ari')
    nmi = _compare_groupings(sttc_groups, dtw_groups, method='nmi')
    
    # Compute group size distributions
    sttc_sizes = [len(g.neurons) for g in sttc_groups]
    dtw_sizes = [len(g.neurons) for g in dtw_groups]
    
    # Compute coherence for each method
    sttc_coherences = []
    for group in sttc_groups:
        if len(group.neurons) >= 2:
            coh = compute_group_coherence(group)
            sttc_coherences.append(coh['mean_coherence'])
    
    dtw_coherences = []
    for group in dtw_groups:
        if len(group.neurons) >= 2:
            coh = compute_group_coherence(group)
            dtw_coherences.append(coh['mean_coherence'])
    
    results = {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'sttc_n_groups': len(sttc_groups),
        'dtw_n_groups': len(dtw_groups),
        'sttc_mean_group_size': np.mean(sttc_sizes) if sttc_sizes else 0,
        'dtw_mean_group_size': np.mean(dtw_sizes) if dtw_sizes else 0,
        'sttc_std_group_size': np.std(sttc_sizes) if sttc_sizes else 0,
        'dtw_std_group_size': np.std(dtw_sizes) if dtw_sizes else 0,
        'sttc_mean_coherence': np.mean(sttc_coherences) if sttc_coherences else 0,
        'dtw_mean_coherence': np.mean(dtw_coherences) if dtw_coherences else 0
    }
    
    return results


def compute_group_overlap(
    sttc_groups: List[NeuronGroup],
    dtw_groups: List[NeuronGroup],
    min_overlap: float = 0.5
) -> List[Tuple[int, int, float]]:
    """
    Find overlapping groups between STTC and DTW methods.
    
    Parameters
    ----------
    sttc_groups : List[NeuronGroup]
        Groups from STTC method
    dtw_groups : List[NeuronGroup]
        Groups from DTW method
    min_overlap : float
        Minimum overlap fraction to report
        
    Returns
    -------
    List[Tuple[int, int, float]]
        List of (sttc_idx, dtw_idx, overlap_fraction) tuples
    """
    overlaps = []
    
    for i, sttc_group in enumerate(sttc_groups):
        sttc_ids = set([n.roi.roi_id for n in sttc_group.neurons])
        
        for j, dtw_group in enumerate(dtw_groups):
            dtw_ids = set([n.roi.roi_id for n in dtw_group.neurons])
            
            # Compute Jaccard index
            intersection = len(sttc_ids & dtw_ids)
            union = len(sttc_ids | dtw_ids)
            
            if union > 0:
                overlap = intersection / union
                
                if overlap >= min_overlap:
                    overlaps.append((i, j, overlap))
    
    # Sort by overlap descending
    overlaps.sort(key=lambda x: x[2], reverse=True)
    
    return overlaps
