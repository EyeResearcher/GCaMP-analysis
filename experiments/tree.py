# experiments/tree.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any

from experiments.summary_utils import StatSummary

@dataclass
class TreeNode:
    name: str
    path: Path
    parent: Optional["TreeNode"] = None
    children: dict[str, "TreeNode"] = field(default_factory=dict)

    # populated for video leaf nodes
    payload: Any = None  # VideoRunRecord for leaves

    # support counts for weighting
    n_videos: int = 0
    n_neurons: int = 0

    # spike summaries over this node's subtree
    kin_unweighted: StatSummary = field(default_factory=StatSummary)
    kin_weighted: StatSummary = field(default_factory=StatSummary)      # level-dependent weighting
    freq_unweighted: StatSummary = field(default_factory=StatSummary)
    freq_weighted: StatSummary = field(default_factory=StatSummary)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(self, child: "TreeNode") -> None:
        self.children[child.name] = child
        child.parent = self

    def iter_nodes(self):
        yield self
        for c in self.children.values():
            yield from c.iter_nodes()


class ExperimentTreeBuilder:
    def __init__(self, is_video_dir: Callable[[Path], bool]):
        self.is_video_dir = is_video_dir

    def build(self, root_dir: Path) -> TreeNode:
        root_dir = Path(root_dir)
        root = TreeNode(name=root_dir.name, path=root_dir)
        self._build_rec(root)
        return root

    def _build_rec(self, node: TreeNode) -> None:
        if self.is_video_dir(node.path):
            return

        subdirs = [p for p in node.path.iterdir() if p.is_dir()]
        for sd in sorted(subdirs):
            child = TreeNode(name=sd.name, path=sd)
            node.add_child(child)
            self._build_rec(child)


def is_video_dir(path: Path) -> bool:
    suite2p_plane0 = path / "suite2p" / "plane0"
    return suite2p_plane0.exists()
