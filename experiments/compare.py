from typing import Protocol, Optional
import pandas as pd
from pathlib import Path

from experiments.processor import VideoRunRecord
from experiments.tree import TreeNode

class SiblingComparator(Protocol):
    def compare(self, parent: TreeNode) -> Optional[pd.DataFrame]:
        ...

        
class BasicSiblingComparator:
    def compare(self, parent: TreeNode) -> Optional[pd.DataFrame]:
        if len(parent.children) < 2:
            return None

        rows = []
        for child in parent.children.values():
            leaves = self._collect_video_records(child)
            if not leaves:
                continue

            rows.append({
                "parent": parent.path.as_posix(),
                "child": child.name,
                "n_videos": len(leaves),
                "rois_total_mean": sum(r.n_rois_total for r in leaves) / len(leaves),
                "rois_good_mean": sum(r.n_rois_good for r in leaves) / len(leaves),
                "neurons_mean": sum(r.n_neurons for r in leaves) / len(leaves),
                "spikes_kept_mean": sum(r.n_spikes_kept for r in leaves) / len(leaves),
                "groups_mean": sum(r.n_groups for r in leaves) / len(leaves),
            })

        if not rows:
            return None

        return pd.DataFrame(rows).sort_values("child")

    def _collect_video_records(self, node: TreeNode) -> list[VideoRunRecord]:
        out = []
        for n in node.iter_nodes():
            if isinstance(n.payload, VideoRunRecord):
                out.append(n.payload)
        return out
    
class ExperimentComparer:
    def __init__(self, comparator: SiblingComparator):
        self.comparator = comparator

    def compare_all(self, root: TreeNode) -> dict[Path, pd.DataFrame]:
        results = {}
        for node in root.iter_nodes():
            if node.children:
                df = self.comparator.compare(node)
                if df is not None:
                    results[node.path] = df
        return results