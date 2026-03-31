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
    """Return ``{'value': value, 'source': source}``."""
    return {'value': value, 'source': source}


def get_label_value(label: dict | int) -> int:
    """Extract numeric label value from dict or int. Returns -1 if missing."""
    if isinstance(label, dict):
        return label.get('value', -1)
    return label


def get_label_source(label: dict | int) -> str:
    """Extract label source string. Returns 'unknown' for raw ints."""
    if isinstance(label, dict):
        return label.get('source', 'unknown')
    return 'unknown'


def label_to_text(label) -> str:
    """Convert label to 'good', 'bad', or 'unlabeled'."""
    value = get_label_value(label) if isinstance(label, dict) else int(label)
    if value == 1:
        return "good"
    if value == 0:
        return "bad"
    return "unlabeled"

def compute_data_summary(roi_dict: dict, level: str = "roi") -> dict:
    """Compute label counts at the 'roi' or 'spike' level."""
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

def validate_roi_label(roi_key: str, roi_data: dict) -> bool:
    """Return True if the ROI label value is 1. Raises ValueError if label is missing."""
    roi_label = roi_data.get('label', None)
    if roi_label is None:
        raise ValueError(f"ROI {roi_key} is missing 'label' data.")
    
    return get_label_value(roi_label) == 1

def normalize_label(label) -> dict:
    """Convert any label format (int, dict, None) to standardized dict format."""
    if label is None:
        return create_label_dict(-1, 'unlabeled')
    if isinstance(label, dict) and 'value' in label and 'source' in label:
        return label
    if isinstance(label, (int, np.integer)):
        if label in (0, 1):
            return create_label_dict(int(label), 'auto')
        return create_label_dict(-1, 'unlabeled')
    return create_label_dict(-1, 'unlabeled')




# =============================================================================
# Label Mutation
# =============================================================================

def update_spike_label(npy_dict: dict, roi_key: str, spike_idx: int, new_label: int) -> bool:
    """Set a spike's label to *new_label* (manual). Returns True if it changed."""
    spike_idx = int(spike_idx)
    current_label = get_label_value(
        npy_dict[roi_key]["spikes"][spike_idx].get("label", create_label_dict(-1, 'unlabeled'))
    )
    changed = (int(new_label) != current_label)
    npy_dict[roi_key]["spikes"][spike_idx]["label"] = create_label_dict(int(new_label), 'manual')
    return changed


def preserve_existing_label(existing_spikes: dict, spike_idx, new_label: dict) -> dict:
    """Keep an existing non-unlabeled label for *spike_idx*; otherwise use *new_label*."""
    if spike_idx not in existing_spikes:
        return new_label

    old_label = existing_spikes[spike_idx].get('label', None)
    normalized = normalize_label(old_label)

    # Only preserve labels that were explicitly set (manual or auto with value 0/1)
    if get_label_value(normalized) != -1:
        return normalized
    return new_label

def reset_spike_labels(roi_dict: dict) -> tuple[dict, int]:
    """Reset all spike labels to unlabeled. Returns (roi_dict, n_reset)."""
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

def matches_label_mode(label, *, unlabeled_only: bool, labeled_only: bool, auto: bool = False) -> bool:
    """Check whether a label passes the unlabeled_only / labeled_only filter."""
    if unlabeled_only and labeled_only:
        raise ValueError("Choose at most one of unlabeled_only or labeled_only.")

    value = get_label_value(label) if isinstance(label, dict) else int(label)
    if unlabeled_only:
        return value == -1
    if labeled_only:
        return value != -1
    if auto:   
        return isinstance(label, dict) and get_label_source(label) == 'auto'
    return True

def get_keys(
    npy_dict: dict,
    *,
    level: str = "roi",
    unlabeled_only: bool = False,
    labeled_only: bool = False,
    auto: bool = False,
    verbose: bool = False,
) -> list[str]:
    """Return ROI keys matching the label filter at the given level ('roi' or 'spike')."""
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