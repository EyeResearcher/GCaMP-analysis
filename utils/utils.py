from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, Iterable, List

@dataclass
class Node:
    level: int
    value: str
    children: Dict[str, "Node"] = field(default_factory=dict)
    agg: Dict[str, float] = field(default_factory=dict)  # store rollups like n, sum, etc.

class FactorTree:
    def __init__(self, levels: List[str], root_name: str = "ROOT"):
        if not levels:
            raise ValueError("levels must be non-empty")
        self.levels = levels
        self.D = len(levels)
        self.root = Node(level=0, value=root_name)

    def add_row(
        self,
        factors: Dict[str, Any],
        payload: Any = None,
        update_agg: Optional[Callable[[Node, Any], None]] = None,
    ) -> Node:
        # factors must have keys for all levels
        path = [str(factors[k]) for k in self.levels]  # KeyError if missing (good early signal)

        cur = self.root
        if update_agg:
            update_agg(cur, payload)

        for i, v in enumerate(path, start=1):
            cur = cur.children.setdefault(v, Node(level=i, value=v))
            if update_agg:
                update_agg(cur, payload)

        return cur

    def get_node(self, prefix: Iterable[Any]) -> Optional[Node]:
        cur = self.root
        for v in prefix:
            cur = cur.children.get(str(v))
            if cur is None:
                return None
        return cur
