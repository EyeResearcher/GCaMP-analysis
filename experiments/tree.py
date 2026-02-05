# experiments/tree.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from posixpath import sep
from typing import Callable, Optional, Any

from experiments.summary_utils import StatSummary

@dataclass
class TreeNode:
    name: str
    path: Path
    parent: Optional["TreeNode"] = None
    children: dict[str, "TreeNode"] = field(default_factory=dict)
    # metadata
    child_idx: Optional[int] = None          # index within parent sibling set
    code_path: tuple[int, ...] = field(default_factory=tuple)
    # populated for video leaf nodes
    payload: Any = None  # VideoRunRecord for leaves

    # support counts for weighting
    n_videos: int = 0
    n_neurons: int = 0
    n_groups: int = 0

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
    def code_str(self, sep: str = "-") -> str:
        return sep.join(map(str, self.code_path))


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

def assign_sibling_indices(root: TreeNode) -> None:
    """
    Assign deterministic sibling indices and code paths.
    Ordering is based on sorted(child.name) to keep it stable across runs.
    """
    root.child_idx = None
    root.code_path = ()

    def dfs(node: TreeNode) -> None:
        # Deterministic ordering
        for idx, name in enumerate(sorted(node.children.keys())):
            child = node.children[name]
            child.child_idx = idx
            child.code_path = node.code_path + (idx,)
            dfs(child)

    dfs(root)

def pretty_print(node: TreeNode, *, indent: str = "", is_last: bool = True) -> None:
    connector = "└── " if is_last else "├── "
    code = node.code_str()
    label = f"{node.name}" + (f" [{code}]" if code else "")

    print(indent + connector + label)

    next_indent = indent + ("    " if is_last else "│   ")
    children = [node.children[k] for k in sorted(node.children.keys())]
    for i, child in enumerate(children):
        pretty_print(child, indent=next_indent, is_last=(i == len(children) - 1))
