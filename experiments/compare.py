# experiments/compare.py
from typing import Protocol, Optional
import pandas as pd
from pathlib import Path

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
            # Skip children with no videos under them
            if child.n_videos <= 0:
                continue

            row = {
                "parent": parent.path.as_posix(),
                "child": child.name,
                "n_videos": child.n_videos,
                "n_neurons": child.n_neurons,
            }

            # ---- Add kinetic stats (unweighted + weighted)
            kin_stats = sorted(set(child.kin_unweighted.means.keys()) | set(child.kin_weighted.means.keys()))
            for stat in kin_stats:
                if stat in child.kin_unweighted.means:
                    row[f"{stat}_mean_unweighted"] = child.kin_unweighted.means[stat]
                    row[f"{stat}_var_unweighted"] = child.kin_unweighted.vars_total.get(stat, 0.0)
                    row[f"{stat}_within_unweighted"] = child.kin_unweighted.vars_within.get(stat, 0.0)
                    row[f"{stat}_between_unweighted"] = child.kin_unweighted.vars_between.get(stat, 0.0)

                if stat in child.kin_weighted.means:
                    row[f"{stat}_mean_weighted"] = child.kin_weighted.means[stat]
                    row[f"{stat}_var_weighted"] = child.kin_weighted.vars_total.get(stat, 0.0)
                    row[f"{stat}_within_weighted"] = child.kin_weighted.vars_within.get(stat, 0.0)
                    row[f"{stat}_between_weighted"] = child.kin_weighted.vars_between.get(stat, 0.0)

            # ---- Add spike_frequency (unweighted + weighted)
            # Frequency lives under the stat name "spike_frequency"
            if "spike_frequency" in child.freq_unweighted.means:
                row["spike_frequency_mean_unweighted"] = child.freq_unweighted.means["spike_frequency"]
                row["spike_frequency_var_unweighted"] = child.freq_unweighted.vars_total.get("spike_frequency", 0.0)
                row["spike_frequency_within_unweighted"] = child.freq_unweighted.vars_within.get("spike_frequency", 0.0)
                row["spike_frequency_between_unweighted"] = child.freq_unweighted.vars_between.get("spike_frequency", 0.0)
            if "spike_frequency" in child.freq_weighted.means:
                row["spike_frequency_mean_weighted"] = child.freq_weighted.means["spike_frequency"]
                row["spike_frequency_var_weighted"] = child.freq_weighted.vars_total.get("spike_frequency", 0.0)
                row["spike_frequency_within_weighted"] = child.freq_weighted.vars_within.get("spike_frequency", 0.0)
                row["spike_frequency_between_weighted"] = child.freq_weighted.vars_between.get("spike_frequency", 0.0)
            rows.append(row)

        if not rows:
            return None

        return pd.DataFrame(rows).sort_values("child")


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
