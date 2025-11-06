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
    Create multi-sheet Excel report with bad ROI tracking.
    
    Sheets:
    1. Neurons - Summary of good neurons
    2. STTC_Groups - STTC grouping results
    3. DTW_Groups - DTW grouping results  
    4. Bad_ROIs - Indices and reasons for filtered ROIs
    5. ROI_Filter_Summary - Statistics about filtering
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # 1. Neuron summary sheet
        if 'summary' in results and results['summary'] is not None:
            results['summary'].to_excel(writer, sheet_name='Neurons', index=False)
        elif 'filtered_neurons' in results:
            # Build summary if not exists
            neuron_data = []
            for neuron in results['filtered_neurons']:
                neuron_data.append({
                    'neuron_id': neuron.row_index,
                    'n_spikes': len(neuron.spikes),
                    'spike_rate_hz': neuron.get_spike_rate()
                })
            pd.DataFrame(neuron_data).to_excel(writer, sheet_name='Neurons', index=False)
        
        # 2. STTC groups sheet
        if results.get('sttc_groups'):
            sttc_data = []
            for i, group in enumerate(results['sttc_groups']):
                for neuron in group:
                    sttc_data.append({
                        'group_id': i,
                        'neuron_id': neuron.row_index,
                        'n_spikes': len(neuron.spikes),
                        'spike_frequency': len(neuron.spikes) / (len(neuron.raw_fluorescence) / 30.0)
                    })
            if sttc_data:
                pd.DataFrame(sttc_data).to_excel(writer, sheet_name='STTC_Groups', index=False)
        
        # 3. DTW groups sheet
        if results.get('dtw_groups'):
            dtw_data = []
            for i, group in enumerate(results['dtw_groups']):
                for neuron in group:
                    dtw_data.append({
                        'group_id': i, 
                        'neuron_id': neuron.row_index,
                        'n_spikes': len(neuron.spikes),
                        'spike_frequency': len(neuron.spikes) / (len(neuron.raw_fluorescence) / 30.0)
                    })
            if dtw_data:
                pd.DataFrame(dtw_data).to_excel(writer, sheet_name='DTW_Groups', index=False)
        
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