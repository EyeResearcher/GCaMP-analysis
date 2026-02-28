"""Experiment tree structure for hierarchical experiment layouts.

Builds a tree from the filesystem where leaf nodes are video directories
(identified by the presence of ``suite2p/plane0/``) and internal nodes
represent experimental groupings (e.g. treatment, timepoint).

After processing, each node carries bottom-up summary statistics that
enable sibling comparisons at every level of the hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Any

from gcamp_analysis.experiments.summary_utils import StatSummary


@dataclass
class TreeNode:
    """A single node in the experiment tree.

    Attributes
    ----------
    name : str
        Directory name for this node.
    path : Path
        Absolute filesystem path.
    parent : TreeNode or None
        Parent node (``None`` for the root).
    children : dict[str, TreeNode]
        Child nodes keyed by directory name.
    payload : Any
        Attached data — a ``VideoRunRecord`` for leaf nodes, ``None``
        for internal nodes before processing.
    n_videos : int
        Number of video leaves in this subtree.
    n_neurons : int
        Total neurons across all videos in this subtree.
    n_groups : dict[str, int]
        Neuron groups per grouping strategy across all videos in this
        subtree, keyed by strategy name (e.g. ``{"corr": 5, "sttc": 3}``).
    kin_unweighted : StatSummary
        Kinetics summary — children weighted equally.
    kin_weighted : StatSummary
        Kinetics summary — children weighted by neuron count (leaf level)
        or video count (higher levels).
    freq_unweighted : StatSummary
        Spike-frequency summary — children weighted equally.
    freq_weighted : StatSummary
        Spike-frequency summary — children weighted as above.
    kin_grouped : StatSummary
        Kinetics summary for neurons belonging to at least one group.
    kin_ungrouped : StatSummary
        Kinetics summary for neurons not in any group.
    freq_grouped : StatSummary
        Spike-frequency summary for grouped neurons.
    freq_ungrouped : StatSummary
        Spike-frequency summary for ungrouped neurons.
    """

    name: str
    path: Path
    parent: Optional["TreeNode"] = None
    children: dict[str, "TreeNode"] = field(default_factory=dict)
    payload: Any = None

    # support counts for weighting
    n_videos: int = 0
    n_neurons: int = 0
    n_groups: dict[str, int] = field(default_factory=dict)

    # spike summaries over this node's subtree
    kin_unweighted: StatSummary = field(default_factory=StatSummary)
    kin_weighted: StatSummary = field(default_factory=StatSummary)
    freq_unweighted: StatSummary = field(default_factory=StatSummary)
    freq_weighted: StatSummary = field(default_factory=StatSummary)

    # grouped vs ungrouped neuron summaries
    kin_grouped: StatSummary = field(default_factory=StatSummary)
    kin_ungrouped: StatSummary = field(default_factory=StatSummary)
    freq_grouped: StatSummary = field(default_factory=StatSummary)
    freq_ungrouped: StatSummary = field(default_factory=StatSummary)

    def is_leaf(self) -> bool:
        """Return ``True`` if the node has no children."""
        return len(self.children) == 0

    def add_child(self, child: "TreeNode") -> None:
        """Attach *child* under this node and set its parent back-link."""
        self.children[child.name] = child
        child.parent = self

    def iter_nodes(self) -> Iterator["TreeNode"]:
        """Yield this node and all descendants in pre-order."""
        yield self
        for c in self.children.values():
            yield from c.iter_nodes()


class ExperimentTreeBuilder:
    """Build a `TreeNode` hierarchy from a directory tree.

    Parameters
    ----------
    is_video_dir : callable
        Predicate that returns ``True`` for directories that should be
        treated as video leaves (no further recursion).
    """

    def __init__(self, is_video_dir: Callable[[Path], bool]):
        self.is_video_dir = is_video_dir

    def build(self, root_dir: Path) -> TreeNode:
        """Recursively build the tree starting from *root_dir*.

        Parameters
        ----------
        root_dir : Path
            Top-level experiment directory.

        Returns
        -------
        TreeNode
            Root of the constructed tree.
        """
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
    """Default leaf detector: ``True`` when ``suite2p/plane0/`` exists.

    Parameters
    ----------
    path : Path
        Directory to test.

    Returns
    -------
    bool
    """
    return (path / "suite2p" / "plane0").exists()


def print_tree(node: TreeNode, *, indent: str = "", is_last: bool = True) -> None:
    """Pretty-print the experiment tree to stdout.

    Parameters
    ----------
    node : TreeNode
        Subtree root to print.
    indent : str
        Current line prefix (used for recursion).
    is_last : bool
        Whether *node* is the last sibling (used for recursion).
    """
    connector = "└── " if is_last else "├── "
    print(indent + connector + node.name)

    next_indent = indent + ("    " if is_last else "│   ")
    children = [node.children[k] for k in sorted(node.children.keys())]
    for i, child in enumerate(children):
        print_tree(child, indent=next_indent, is_last=(i == len(children) - 1))
