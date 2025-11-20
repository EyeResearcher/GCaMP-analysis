"""IO handlers for saving results with bad ROI tracking."""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from data_classes import Timepoint, Experiment

logger = logging.getLogger(__name__)

def save_video_summary(results: Dict, output_dir: Path) -> pd.DataFrame:
    """Save video processing results."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    neurons = results.get('filtered_neurons', [])
    if not neurons:
        return pd.DataFrame()
    
    # Build summary
    rows = []
    for neuron in neurons:
        row = {
            'neuron_id': neuron.row_index,
            'n_spikes': len(neuron.spikes),
            'spike_frequency': len(neuron.spikes) / (len(neuron.raw_fluorescence) / 30.0)
        }
        
        # Add group memberships
        for i, group in enumerate(results.get('sttc_groups', [])):
            if neuron in group:
                row['sttc_group'] = i
                break
        else:
            row['sttc_group'] = -1
        
        for i, group in enumerate(results.get('dtw_groups', [])):
            if neuron in group:
                row['dtw_group'] = i
                break
        else:
            row['dtw_group'] = -1
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    
    # Save files
    summary_df.to_csv(output_dir / 'neuron_summary.csv', index=False)
    
    if results.get('sttc_matrix') is not None:
        np.save(output_dir / 'sttc_matrix.npy', results['sttc_matrix'])
    
    if results.get('dtw_matrix') is not None:
        np.save(output_dir / 'dtw_matrix.npy', results['dtw_matrix'])
    
    # Save Excel with multiple sheets including bad ROI tracking
    create_excel_report(results, output_dir / 'analysis.xlsx')
    
    return summary_df

def create_excel_report(results: Dict, output_path: Path):
    """
    Create comprehensive multi-sheet Excel report.
    
    Sheets:
    1. Neuron_Summary - Per-neuron summary statistics
    2. Supplementary_Data - Per-spike detailed data for each neuron
    3. Group_Analysis - Per-group summary statistics (STTC/DTW)
    4. Bad_ROIs - Indices and reasons for filtered ROIs
    5. ROI_Filter_Summary - Statistics about filtering
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Neuron Summary (one row per neuron)
        neurons = results.get('filtered_neurons', [])
        if neurons:
            neuron_summary = []
            for neuron in neurons:
                spikes = neuron.spikes
                
                # Calculate spike statistics using actual Spike attributes
                f_values = [s.f_value for s in spikes] if spikes else []
                prob_heights = [s.prob_height for s in spikes] if spikes else []
                prominences = [s.prominence for s in spikes if hasattr(s, 'prominence')] if spikes else []
                decay_taus = [s.decay_tau for s in spikes if hasattr(s, 'decay_tau') and s.decay_tau is not None] if spikes else []
                rise_slopes = [s.rise_slope for s in spikes if hasattr(s, 'rise_slope') and s.rise_slope is not None] if spikes else []
                
                row = {
                    'original_roi_index': neuron.row_index,
                    'n_spikes': len(spikes),
                    'spike_frequency_hz': neuron.get_spike_rate() if hasattr(neuron, 'get_spike_rate') else len(spikes) / (len(neuron.raw_fluorescence) / 30.0),
                    'avg_f_value': np.mean(f_values) if f_values else np.nan,
                    'var_f_value': np.var(f_values) if f_values else np.nan,
                    'avg_prob_height': np.mean(prob_heights) if prob_heights else np.nan,
                    'var_prob_height': np.var(prob_heights) if prob_heights else np.nan,
                    'avg_prominence': np.mean(prominences) if prominences else np.nan,
                    'var_prominence': np.var(prominences) if prominences else np.nan,
                    'avg_decay_tau': np.mean(decay_taus) if decay_taus else np.nan,
                    'var_decay_tau': np.var(decay_taus) if decay_taus else np.nan,
                    'avg_rise_slope': np.mean(rise_slopes) if rise_slopes else np.nan,
                    'var_rise_slope': np.var(rise_slopes) if rise_slopes else np.nan,
                }
                neuron_summary.append(row)
            
            pd.DataFrame(neuron_summary).to_excel(writer, sheet_name='Neuron_Summary', index=False)
        
        # Sheet 2: Supplementary Data (per-spike values for each neuron)
        if neurons:
            supp_data = []
            for neuron in neurons:
                spikes = neuron.spikes
                
                # Collect arrays of spike data using actual Spike attributes
                spike_indices = [s.frame_index for s in spikes] if spikes else []
                cascade_indices = [s.cascade_peak_idx for s in spikes] if spikes else []
                f_values = [s.f_value for s in spikes] if spikes else []
                prob_heights = [s.prob_height for s in spikes] if spikes else []
                prominences = [s.prominence for s in spikes if hasattr(s, 'prominence')] if spikes else []
                decay_taus = [s.decay_tau for s in spikes if hasattr(s, 'decay_tau') and s.decay_tau is not None] if spikes else []
                rise_slopes = [s.rise_slope for s in spikes if hasattr(s, 'rise_slope') and s.rise_slope is not None] if spikes else []
                
                row = {
                    'original_roi_index': neuron.row_index,
                    'spike_frame_indices': str(spike_indices),  # Convert to string for Excel
                    'cascade_peak_indices': str(cascade_indices),
                    'f_values': str(f_values),
                    'prob_heights': str(prob_heights),
                    'prominences': str(prominences),
                    'decay_taus': str(decay_taus),
                    'rise_slopes': str(rise_slopes),
                }
                supp_data.append(row)
            
            pd.DataFrame(supp_data).to_excel(writer, sheet_name='Supplementary_Data', index=False)
        
        # Sheet 3: Group Analysis (one row per group)
        group_analysis = []
        
        # STTC Groups
        sttc_groups = results.get('sttc_groups', [])
        sttc_matrix = results.get('sttc_matrix')
        if sttc_groups:
            for i, group in enumerate(sttc_groups):
                if not group:
                    continue
                    
                # Get all neurons in group
                group_neurons = list(group)
                neuron_indices = [n.row_index for n in group_neurons]
                
                # Aggregate spike statistics using actual Spike attributes
                all_f_values = []
                all_prob_heights = []
                all_prominences = []
                total_spikes = 0
                
                for neuron in group_neurons:
                    total_spikes += len(neuron.spikes)
                    all_f_values.extend([s.f_value for s in neuron.spikes])
                    all_prob_heights.extend([s.prob_height for s in neuron.spikes])
                    all_prominences.extend([s.prominence for s in neuron.spikes if hasattr(s, 'prominence')])
                
                # Calculate STTC values within group
                sttc_values = []
                if sttc_matrix is not None and len(group_neurons) > 1:
                    for idx1, n1 in enumerate(group_neurons):
                        for idx2, n2 in enumerate(group_neurons):
                            if idx1 < idx2:
                                # Find matrix indices for these neurons
                                # This assumes neurons list matches matrix order
                                all_neurons = results.get('filtered_neurons', [])
                                if n1 in all_neurons and n2 in all_neurons:
                                    mat_idx1 = all_neurons.index(n1)
                                    mat_idx2 = all_neurons.index(n2)
                                    if mat_idx1 < sttc_matrix.shape[0] and mat_idx2 < sttc_matrix.shape[1]:
                                        sttc_values.append(sttc_matrix[mat_idx1, mat_idx2])
                
                # Calculate shared spikes (simplified - spikes within same time window)
                shared_spike_count = 0
                if len(group_neurons) > 1:
                    # Count spikes that occur within ±1 frame across neurons
                    all_spike_times = []
                    for neuron in group_neurons:
                        all_spike_times.extend([s.frame_index for s in neuron.spikes])
                    all_spike_times = sorted(all_spike_times)
                    
                    # Count duplicates (shared spikes within 1 frame)
                    for i in range(len(all_spike_times) - 1):
                        if all_spike_times[i+1] - all_spike_times[i] <= 1:
                            shared_spike_count += 1
                
                row = {
                    'group_id': f'STTC_{i}',
                    'group_type': 'STTC',
                    'n_neurons': len(group_neurons),
                    'neuron_indices': str(neuron_indices),
                    'total_spikes': total_spikes,
                    'avg_f_value': np.mean(all_f_values) if all_f_values else np.nan,
                    'var_f_value': np.var(all_f_values) if all_f_values else np.nan,
                    'avg_prob_height': np.mean(all_prob_heights) if all_prob_heights else np.nan,
                    'var_prob_height': np.var(all_prob_heights) if all_prob_heights else np.nan,
                    'avg_prominence': np.mean(all_prominences) if all_prominences else np.nan,
                    'var_prominence': np.var(all_prominences) if all_prominences else np.nan,
                    'mean_sttc': np.mean(sttc_values) if sttc_values else np.nan,
                    'var_sttc': np.var(sttc_values) if sttc_values else np.nan,
                    'mean_dtw': np.nan,  # Not applicable for STTC groups
                    'var_dtw': np.nan,
                    'n_shared_spikes': shared_spike_count,
                }
                group_analysis.append(row)
        
        # DTW Groups
        dtw_groups = results.get('dtw_groups', [])
        dtw_matrix = results.get('dtw_matrix')
        if dtw_groups:
            for i, group in enumerate(dtw_groups):
                if not group:
                    continue
                    
                group_neurons = list(group)
                neuron_indices = [n.row_index for n in group_neurons]
                
                # Aggregate spike statistics using actual Spike attributes
                all_f_values = []
                all_prob_heights = []
                all_prominences = []
                total_spikes = 0
                
                for neuron in group_neurons:
                    total_spikes += len(neuron.spikes)
                    all_f_values.extend([s.f_value for s in neuron.spikes])
                    all_prob_heights.extend([s.prob_height for s in neuron.spikes])
                    all_prominences.extend([s.prominence for s in neuron.spikes if hasattr(s, 'prominence')])
                
                # Calculate DTW values within group
                dtw_values = []
                if dtw_matrix is not None and len(group_neurons) > 1:
                    for idx1, n1 in enumerate(group_neurons):
                        for idx2, n2 in enumerate(group_neurons):
                            if idx1 < idx2:
                                all_neurons = results.get('filtered_neurons', [])
                                if n1 in all_neurons and n2 in all_neurons:
                                    mat_idx1 = all_neurons.index(n1)
                                    mat_idx2 = all_neurons.index(n2)
                                    if mat_idx1 < dtw_matrix.shape[0] and mat_idx2 < dtw_matrix.shape[1]:
                                        dtw_values.append(dtw_matrix[mat_idx1, mat_idx2])
                
                # Calculate shared spikes
                shared_spike_count = 0
                if len(group_neurons) > 1:
                    all_spike_times = []
                    for neuron in group_neurons:
                        all_spike_times.extend([s.frame_index for s in neuron.spikes])
                    all_spike_times = sorted(all_spike_times)
                    
                    for i in range(len(all_spike_times) - 1):
                        if all_spike_times[i+1] - all_spike_times[i] <= 1:
                            shared_spike_count += 1
                
                row = {
                    'group_id': f'DTW_{i}',
                    'group_type': 'DTW',
                    'n_neurons': len(group_neurons),
                    'neuron_indices': str(neuron_indices),
                    'total_spikes': total_spikes,
                    'avg_f_value': np.mean(all_f_values) if all_f_values else np.nan,
                    'var_f_value': np.var(all_f_values) if all_f_values else np.nan,
                    'avg_prob_height': np.mean(all_prob_heights) if all_prob_heights else np.nan,
                    'var_prob_height': np.var(all_prob_heights) if all_prob_heights else np.nan,
                    'avg_prominence': np.mean(all_prominences) if all_prominences else np.nan,
                    'var_prominence': np.var(all_prominences) if all_prominences else np.nan,
                    'mean_sttc': np.nan,  # Not applicable for DTW groups
                    'var_sttc': np.nan,
                    'mean_dtw': np.mean(dtw_values) if dtw_values else np.nan,
                    'var_dtw': np.var(dtw_values) if dtw_values else np.nan,
                    'n_shared_spikes': shared_spike_count,
                }
                group_analysis.append(row)
        
        if group_analysis:
            pd.DataFrame(group_analysis).to_excel(writer, sheet_name='Group_Analysis', index=False)
        
        # 4. Bad ROIs sheet - IMPORTANT FOR CROSS-REFERENCING
        if 'bad_roi_indices' in results and results['bad_roi_indices']:
            bad_roi_data = []
            for i, idx in enumerate(results['bad_roi_indices']):
                row = {
                    'roi_index': idx,
                    'reason': 'ROI classifier rejection'
                }
                # Add features if available
                if 'bad_roi_features' in results:
                    if 'derivative_skew' in results['bad_roi_features']:
                        row['derivative_skew'] = results['bad_roi_features']['derivative_skew'][i] if i < len(results['bad_roi_features']['derivative_skew']) else np.nan
                    if 'spike_prom_mean' in results['bad_roi_features']:
                        row['spike_prom_mean'] = results['bad_roi_features']['spike_prom_mean'][i] if i < len(results['bad_roi_features']['spike_prom_mean']) else np.nan
                bad_roi_data.append(row)
            
            bad_roi_df = pd.DataFrame(bad_roi_data)
            bad_roi_df.to_excel(writer, sheet_name='Bad_ROIs', index=False)
        
        # 5. ROI Filter Summary sheet
        total_rois = len(results.get('all_rois', []))
        good_rois = len(results.get('good_rois', []))
        bad_rois = len(results.get('bad_roi_indices', []))
        
        filter_summary = pd.DataFrame([{
            'total_rois': total_rois,
            'good_rois': good_rois,
            'bad_rois': bad_rois,
            'filter_rate': f"{bad_rois/total_rois*100:.1f}%" if total_rois > 0 else "0%",
            'retention_rate': f"{good_rois/total_rois*100:.1f}%" if total_rois > 0 else "0%",
            'neurons_with_spikes': len(results.get('filtered_neurons', [])),
            'total_spikes': sum(len(n.spikes) for n in results.get('filtered_neurons', [])),
            'sttc_groups': len(results.get('sttc_groups', [])),
            'dtw_groups': len(results.get('dtw_groups', []))
        }])
        filter_summary.to_excel(writer, sheet_name='ROI_Filter_Summary', index=False)
    
    logger.info(f"Saved Excel report to {output_path}")

def save_filtered_suite2p(video_path: Path, 
                         good_roi_mask: np.ndarray,
                         suite2p_data: Dict,
                         cascade_prob: Optional[np.ndarray] = None) -> Path:
    """
    Save filtered Suite2p files with only good ROIs.
    
    Creates filtered_suite2p/plane0/ with filtered arrays.
    """
    # Create filtered_suite2p directory
    filtered_dir = video_path / 'filtered_suite2p' / 'plane0'
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to filter (2D arrays where first dimension is ROIs)
    files_to_filter = {
        'F': suite2p_data.get('F'),
        'Fneu': suite2p_data.get('Fneu'),
        'spks': suite2p_data.get('spks'),
        'iscell': suite2p_data.get('iscell')
    }
    
    # Save filtered arrays
    for name, data in files_to_filter.items():
        if data is not None:
            filtered_data = data[good_roi_mask]
            save_path = filtered_dir / f'{name}.npy'
            np.save(save_path, filtered_data)
            logger.debug(f"Saved filtered {name}: {data.shape} -> {filtered_data.shape}")
    
    # Handle stat (list of dicts)
    if 'stat' in suite2p_data and suite2p_data['stat'] is not None:
        filtered_stat = [suite2p_data['stat'][i] for i in np.where(good_roi_mask)[0]]
        np.save(filtered_dir / 'stat.npy', filtered_stat, allow_pickle=True)
        logger.debug(f"Saved filtered stat: {len(suite2p_data['stat'])} -> {len(filtered_stat)}")
    
    # Save cascade probabilities if available
    if cascade_prob is not None:
        filtered_cascade = cascade_prob[good_roi_mask]
        np.save(filtered_dir / 'cascade_spike_prob.npy', filtered_cascade)
        logger.debug(f"Saved filtered cascade_spike_prob: {cascade_prob.shape} -> {filtered_cascade.shape}")
    
    # Copy ops unchanged
    if 'ops' in suite2p_data and suite2p_data['ops'] is not None:
        np.save(filtered_dir / 'ops.npy', suite2p_data['ops'], allow_pickle=True)
        logger.debug("Copied ops.npy unchanged")
    
    # Save the indices of good and bad ROIs for reference
    good_indices = np.where(good_roi_mask)[0]
    bad_indices = np.where(~good_roi_mask)[0]
    
    np.save(filtered_dir / 'good_roi_indices.npy', good_indices)
    np.save(filtered_dir / 'bad_roi_indices.npy', bad_indices)
    
    # Save mapping file for cross-reference
    mapping_df = pd.DataFrame({
        'original_index': np.arange(len(good_roi_mask)),
        'is_good': good_roi_mask,
        'filtered_index': [-1] * len(good_roi_mask)
    })
    mapping_df.loc[good_roi_mask, 'filtered_index'] = np.arange(sum(good_roi_mask))
    mapping_df.to_csv(filtered_dir / 'roi_mapping.csv', index=False)
    
    logger.info(f"Filtered Suite2p saved: {len(good_indices)} good, {len(bad_indices)} bad ROIs")
    
    return filtered_dir

def save_timepoint_summary(timepoint: Timepoint, output_path: Path):
    """Save timepoint-level summary."""
    # Aggregate from videos
    all_summaries = []
    for video in timepoint.videos:
        summary = video.get_summary() if hasattr(video, 'get_summary') else None
        if summary is not None and not summary.empty:
            summary['video_id'] = video.video_id
            summary['treatment'] = video.treatment
            summary['region'] = video.region
            all_summaries.append(summary)
    
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main summary
            combined.to_excel(writer, sheet_name='Summary', index=False)
            
            # Statistics by treatment
            treatment_stats = combined.groupby('treatment').agg({
                'n_spikes': ['mean', 'std', 'count'],
                'spike_frequency': ['mean', 'std']
            }).round(3)
            treatment_stats.to_excel(writer, sheet_name='Treatment_Stats')
            
        logger.info(f"Saved timepoint summary to {output_path}")

def save_timepoint_summary_by_video(timepoint: Timepoint, output_path: Path):
    """
    Save timepoint summary with videos as rows.
    
    Each row represents one video with aggregate statistics across all neurons.
    Columns include treatment information and video-level averages/variances.
    """
    video_summary = []
    
    for video in timepoint.videos:
        neurons = video.neurons if hasattr(video, 'neurons') else []
        if not neurons:
            continue
        
        # Aggregate across all neurons in this video
        all_f_values = []
        all_prob_heights = []
        all_prominences = []
        all_decay_taus = []
        all_rise_slopes = []
        all_spike_frequencies = []
        total_spikes = 0
        
        for neuron in neurons:
            spikes = neuron.spikes
            total_spikes += len(spikes)
            
            # Calculate spike frequency for this neuron
            if hasattr(neuron, 'get_spike_rate'):
                spike_freq = neuron.get_spike_rate()
            else:
                spike_freq = len(spikes) / (len(neuron.raw_fluorescence) / 30.0) if len(neuron.raw_fluorescence) > 0 else 0
            all_spike_frequencies.append(spike_freq)
            
            # Collect spike properties (using actual Spike attributes)
            all_f_values.extend([s.f_value for s in spikes])
            all_prob_heights.extend([s.prob_height for s in spikes])
            all_prominences.extend([s.prominence for s in spikes])
            all_decay_taus.extend([s.decay_tau for s in spikes if hasattr(s, 'decay_tau') and s.decay_tau is not None])
            all_rise_slopes.extend([s.rise_slope for s in spikes if hasattr(s, 'rise_slope') and s.rise_slope is not None])
        
        row = {
            'video_id': video.video_id,
            'treatment': video.treatment if hasattr(video, 'treatment') else 'unknown',
            'n_neurons': len(neurons),
            'total_spikes': total_spikes,
            'avg_spike_frequency_hz': np.mean(all_spike_frequencies) if all_spike_frequencies else np.nan,
            'var_spike_frequency_hz': np.var(all_spike_frequencies) if all_spike_frequencies else np.nan,
            'avg_f_value': np.mean(all_f_values) if all_f_values else np.nan,
            'var_f_value': np.var(all_f_values) if all_f_values else np.nan,
            'avg_prob_height': np.mean(all_prob_heights) if all_prob_heights else np.nan,
            'var_prob_height': np.var(all_prob_heights) if all_prob_heights else np.nan,
            'avg_prominence': np.mean(all_prominences) if all_prominences else np.nan,
            'var_prominence': np.var(all_prominences) if all_prominences else np.nan,
            'avg_decay_tau': np.mean(all_decay_taus) if all_decay_taus else np.nan,
            'var_decay_tau': np.var(all_decay_taus) if all_decay_taus else np.nan,
            'avg_rise_slope': np.mean(all_rise_slopes) if all_rise_slopes else np.nan,
            'var_rise_slope': np.var(all_rise_slopes) if all_rise_slopes else np.nan,
        }
        
        # Add group information
        sttc_groups = video.sttc_groups if hasattr(video, 'sttc_groups') else []
        dtw_groups = video.dtw_groups if hasattr(video, 'dtw_groups') else []
        row['n_sttc_groups'] = len(sttc_groups)
        row['n_dtw_groups'] = len(dtw_groups)
        
        video_summary.append(row)
    
    if video_summary:
        df = pd.DataFrame(video_summary)
        df.to_excel(output_path, index=False, engine='openpyxl')
        logger.info(f"Saved timepoint video summary to {output_path}")
    else:
        logger.warning(f"No video data to save for timepoint summary")

def save_experiment_summary(experiment: Experiment, output_path: Path):
    """Save experiment-level summary."""
    # Aggregate from timepoints
    all_summaries = []
    for tp in experiment.timepoints:
        summary = tp.get_summary()
        if summary is not None and not summary.empty:
            summary['timepoint'] = tp.name
            all_summaries.append(summary)
    
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Full data
            combined.to_excel(writer, sheet_name='All_Data', index=False)
            
            # Summary stats - group by available columns
            group_cols = ['timepoint']
            if 'treatment' in combined.columns:
                group_cols.append('treatment')
            
            if 'n_spikes' in combined.columns and 'spike_frequency' in combined.columns:
                summary_stats = combined.groupby(group_cols).agg({
                    'n_spikes': ['mean', 'std', 'count'],
                    'spike_frequency': ['mean', 'std']
                }).round(3)
                summary_stats.to_excel(writer, sheet_name='Summary_Stats')
            
        logger.info(f"Saved experiment summary to {output_path}")