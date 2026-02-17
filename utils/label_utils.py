"""
Centralized label utilities for ROI and spike classification.

All label creation, normalization, querying, and mutation functions live here.
"""
import numpy as np
from typing import Any


# =============================================================================
# Label Creation & Querying
# =============================================================================
def parse_spike_key(spike_key: str) -> tuple[str, int]:
    """Parse 'roikey-3' → ('roikey', 3)."""
    roi_key, spike_idx_str = spike_key.rsplit("-", 1)
    return roi_key, int(spike_idx_str)


def make_spike_key(roi_key: str, spike_idx: int) -> str:
    """Inverse of parse_spike_key."""
    return f"{roi_key}-{int(spike_idx)}"

def create_label_dict(value: int, source: str = 'manual') -> dict:
    """
    Create a standardized label dictionary.

    Parameters
    ----------
    value : int
        Label value (0, 1, or -1 for unlabeled)
    source : str, optional
        Source of the label, by default 'manual'

    Returns
    -------
    label : dict
        Dictionary with 'value' and 'source' keys
    """
    return {'value': value, 'source': source}


def get_label_value(label: dict | int) -> int:
    """
    Extract numeric label value from either dict or int format.

    Parameters
    ----------
    label : dict | int
        Label as dict with 'value' key or raw int

    Returns
    -------
    value : int
        Numeric label value, -1 if not found
    """
    if isinstance(label, dict):
        return label.get('value', -1)
    return label


def get_label_source(label: dict | int) -> str:
    """
    Extract label source from either dict or int format.

    Parameters
    ----------
    label : dict | int
        Label as dict with 'source' key or raw int

    Returns
    -------
    source : str
        Label source, 'unknown' if not found
    """
    if isinstance(label, dict):
        return label.get('source', 'unknown')
    return 'unknown'


def label_to_text(label) -> str:
    """
    Convert a label (dict or int) to a human-readable string.

    Returns
    -------
    text : str
        'good', 'bad', or 'unlabeled'
    """
    value = get_label_value(label) if isinstance(label, dict) else int(label)
    if value == 1:
        return "good"
    if value == 0:
        return "bad"
    return "unlabeled"

def compute_data_summary(roi_dict: dict, level: str = "roi") -> dict:
    """
    Compute label counts for ROI or spike data.

    Parameters
    ----------
    roi_dict : dict
        ROI data dictionary.
    level : str
        ``"roi"`` for ROI-level labels, ``"spike"`` for spike-level labels.

    Returns
    -------
    dict
    """
    if level == "roi":
        items = list(roi_dict.values())
        get_label = lambda item: item.get("label", {})
    elif level == "spike":
        items = [
            spike_data
            for roi_data in roi_dict.values()
            for spike_data in roi_data.get("spikes", {}).values()
        ]
        get_label = lambda item: item.get("label", {})
    else:
        raise ValueError(f"Unknown level: {level!r}. Use 'roi' or 'spike'.")

    summary = {
        "level": level,
        "n_total": len(items),
        "n_good": sum(1 for item in items if get_label_value(get_label(item)) == 1),
        "n_bad": sum(1 for item in items if get_label_value(get_label(item)) == 0),
        "n_unlabeled": sum(1 for item in items if get_label_value(get_label(item)) == -1),
        "n_manual": sum(1 for item in items if get_label_source(get_label(item)) == "manual"),
        "n_auto": sum(1 for item in items if get_label_source(get_label(item)) == "auto"),
    }

    if level == "roi":
        summary["total_spikes"] = sum(len(r.get("spikes", {})) for r in roi_dict.values())
    elif level == "spike":
        summary["n_rois"] = len(roi_dict)
        summary["n_rois_with_spikes"] = sum(1 for r in roi_dict.values() if r.get("spikes"))

    return summary

# =============================================================================
# Label Normalization
# =============================================================================

def normalize_label(label) -> dict:
    """
    Convert any label format (int, dict, None) to the standardized dict format.

    Handles:
    - None  -> unlabeled
    - int / np.integer  -> auto-sourced if 0 or 1, else unlabeled
    - dict with 'value' and 'source' keys -> returned as-is

    This unifies the former ``normalize_label_format`` (ROI) and
    ``normalize_spike_label`` (spike) functions.

    Parameters
    ----------
    label : int | dict | None
        Label in any legacy or current format.

    Returns
    -------
    label : dict
        Standardized label dict with 'value' and 'source' keys.
    """
    if label is None:
        return create_label_dict(-1, 'unlabeled')
    if isinstance(label, dict) and 'value' in label and 'source' in label:
        return label
    if isinstance(label, (int, np.integer)):
        if label in (0, 1):
            return create_label_dict(int(label), 'auto')
        return create_label_dict(-1, 'unlabeled')
    return create_label_dict(-1, 'unlabeled')


# Keep old names as aliases for backwards compatibility
normalize_label_format = normalize_label
normalize_spike_label = normalize_label


# =============================================================================
# Label Mutation
# =============================================================================

def update_spike_label(npy_dict: dict, roi_key: str, spike_idx: int, new_label: int) -> bool:
    """
    Update the label for a spike in the npy_dict using standardized label dicts.

    Parameters
    ----------
    npy_dict : dict
        Dictionary of ROI data.
    roi_key : str
        Key identifying the ROI.
    spike_idx : int
        Index of the spike within the ROI.
    new_label : int
        New label value (0 = bad, 1 = good).

    Returns
    -------
    changed : bool
        True if the label was different from the existing one.
    """
    spike_idx = int(spike_idx)
    current_label = get_label_value(
        npy_dict[roi_key]["spikes"][spike_idx].get("label", create_label_dict(-1, 'unlabeled'))
    )
    changed = (int(new_label) != current_label)
    npy_dict[roi_key]["spikes"][spike_idx]["label"] = create_label_dict(int(new_label), 'manual')
    return changed


def preserve_existing_label(existing_spikes: dict, spike_idx, new_label: dict) -> dict:
    """
    Preserve a manually-annotated label from a prior session if one exists
    for this spike index, otherwise return the new (detection-assigned) label.

    Parameters
    ----------
    existing_spikes : dict
        Previously stored spike data keyed by spike index.
    spike_idx : int
        Index of the spike to look up.
    new_label : dict
        Default label assigned during current detection.

    Returns
    -------
    label : dict
        Normalized label dict, preferring the existing manual label.
    """
    if spike_idx not in existing_spikes:
        return new_label

    old_label = existing_spikes[spike_idx].get('label', None)
    normalized = normalize_label(old_label)

    # Only preserve labels that were explicitly set (manual or auto with value 0/1)
    if get_label_value(normalized) != -1:
        return normalized
    return new_label

def reset_spike_labels(roi_dict: dict) -> tuple[dict, int]:
    """Reset all spike labels to unlabeled.

    Parameters
    ----------
    roi_dict : dict
        ROI data dictionary with spike labels.

    Returns
    -------
    tuple[dict, int]
        Modified dict and count of labels that were reset.
    """
    n_reset = 0
    for roi_data in roi_dict.values():
        if 'spikes' in roi_data:
            for spike_idx in roi_data['spikes']:
                val = get_label_value(roi_data['spikes'][spike_idx].get('label', {}))
                if val != -1:
                    n_reset += 1
                roi_data['spikes'][spike_idx]['label'] = create_label_dict(-1, 'unlabeled')
    return roi_dict, n_reset
# =============================================================================
# Label-Based Filtering
# =============================================================================

def matches_label_mode(label, *, unlabeled_only: bool, labeled_only: bool) -> bool:
    """
    Check whether a label matches the requested filtering mode.

    Parameters
    ----------
    label : dict | int
        The label to check.
    unlabeled_only : bool
        If True, only unlabeled items match.
    labeled_only : bool
        If True, only labeled items match.

    Returns
    -------
    matches : bool

    Raises
    ------
    ValueError
        If both flags are True.
    """
    if unlabeled_only and labeled_only:
        raise ValueError("Choose at most one of unlabeled_only or labeled_only.")

    value = get_label_value(label) if isinstance(label, dict) else int(label)
    if unlabeled_only:
        return value == -1
    if labeled_only:
        return value != -1
    return True




def get_keys(
    npy_dict: dict,
    *,
    level: str = "roi",
    unlabeled_only: bool = False,
    labeled_only: bool = False,
    verbose: bool = False,
) -> list[str]:
    """
    Return keys from npy_dict matching the label filter.

    Parameters
    ----------
    npy_dict : dict
    level : str
        ``"roi"`` filters on ROI-level labels.
        ``"spike"`` returns ROI keys that have at least one spike matching the filter.
    unlabeled_only : bool
    labeled_only : bool
    verbose : bool
    """
    keys: list[str] = []

    for roi_key, roi_data in npy_dict.items():
        if level == "roi":
            lbl = roi_data.get("label", {})
            if matches_label_mode(lbl, unlabeled_only=unlabeled_only, labeled_only=labeled_only):
                keys.append(str(roi_key))

        elif level == "spike":
            spikes = roi_data.get("spikes", {})
            if not isinstance(spikes, dict) or len(spikes) == 0:
                continue
            for spk_data in spikes.values():
                lbl = spk_data.get("label", create_label_dict(-1, "unlabeled"))
                if matches_label_mode(lbl, unlabeled_only=unlabeled_only, labeled_only=labeled_only):
                    keys.append(str(roi_key))
                    break

    if verbose:
        print(f"Found {len(keys)} {level} keys matching filter")

    return keys