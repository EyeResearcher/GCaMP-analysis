"""
Generate a structural map of the entire project: directories, modules, classes, and functions.
Useful for spotting redundancies and consolidation opportunities.
"""
import ast
import sys
from pathlib import Path
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────
SKIP_DIRS = {
    "__pycache__", ".git", ".vscode", ".pytest_cache",
    "node_modules", ".ipynb_checkpoints", "v",
}
SKIP_FILES = {".gitignore", "CACHEDIR.TAG", "README.md"}
INDENT_GUIDE = "│   "
INDENT_LAST  = "    "
TEE          = "├── "
CORNER       = "└── "


# ── AST helpers ────────────────────────────────────────────────────────────
def _signature(node: ast.FunctionDef) -> str:
    """Return a short signature string for a function/method node."""
    args = []
    for a in node.args.args:
        name = a.arg
        if a.annotation:
            name += f": {ast.unparse(a.annotation)}"
        args.append(name)
    sig = ", ".join(args)
    ret = ""
    if node.returns:
        ret = f" -> {ast.unparse(node.returns)}"
    return f"({sig}){ret}"


def parse_python_file(path: Path) -> list[dict]:
    """
    Parse a .py file and return top-level classes and functions.
    Each entry: {"kind": "class"|"function", "name": ..., "sig": ..., "methods": [...]}
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    items = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({
                        "kind": "method",
                        "name": child.name,
                        "sig": _signature(child),
                        "lineno": child.lineno,
                    })
            items.append({
                "kind": "class",
                "name": node.name,
                "methods": methods,
                "lineno": node.lineno,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append({
                "kind": "function",
                "name": node.name,
                "sig": _signature(node),
                "lineno": node.lineno,
            })
    return items


# ── Notebook helpers ───────────────────────────────────────────────────────
def parse_notebook(path: Path) -> list[dict]:
    """Extract top-level classes/functions from code cells in a .ipynb."""
    import json
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    source_lines = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source_lines.extend(cell.get("source", []))
            source_lines.append("\n")

    source = "".join(source_lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    items = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({"kind": "method", "name": child.name, "sig": _signature(child), "lineno": child.lineno})
            items.append({"kind": "class", "name": node.name, "methods": methods, "lineno": node.lineno})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append({"kind": "function", "name": node.name, "sig": _signature(node), "lineno": node.lineno})
    return items
def find_duplicates(directory: Path) -> None:
    """Find function/class names that appear in multiple files."""
    from collections import defaultdict
    
    registry: dict[str, list[str]] = defaultdict(list)  # name -> [file, file, ...]
    
    for py_file in directory.rglob("*.py"):
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue
        rel = py_file.relative_to(directory)
        for item in parse_python_file(py_file):
            key = f"{item['kind']}:{item['name']}"
            registry[key].append(str(rel))
    
    dupes = {k: v for k, v in registry.items() if len(v) > 1}
    if dupes:
        print(f"\n{'='*80}")
        print("  ⚠️  POTENTIAL REDUNDANCIES (same name in multiple files)")
        print(f"{'='*80}")
        for name, files in sorted(dupes.items()):
            kind, symbol = name.split(":", 1)
            print(f"\n  {kind} '{symbol}' found in:")
            for f in files:
                print(f"    - {f}")

# ── Tree printer ───────────────────────────────────────────────────────────
ICONS = {
    "dir": "📁",
    "py": "🐍",
    "ipynb": "📓",
    "class": "🏷️",
    "method": "  ⚙️",
    "function": "🔹",
    "file": "📄",
}


def _print_items(items: list[dict], prefix: str) -> None:
    """Print parsed AST items (classes, functions) with tree lines."""
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = CORNER if is_last else TEE
        next_prefix = prefix + (INDENT_LAST if is_last else INDENT_GUIDE)

        if item["kind"] == "class":
            print(f"{prefix}{connector}{ICONS['class']} class {item['name']}  (L{item['lineno']})")
            methods = item.get("methods", [])
            for j, m in enumerate(methods):
                m_last = j == len(methods) - 1
                m_conn = CORNER if m_last else TEE
                print(f"{next_prefix}{m_conn}{ICONS['method']} {m['name']}{m['sig']}  (L{m['lineno']})")
        else:
            print(f"{prefix}{connector}{ICONS['function']} {item['name']}{item.get('sig', '')}  (L{item['lineno']})")


def walk_tree(directory: Path, prefix: str = "", depth: int = 0, max_depth: int = 10) -> None:
    """Recursively walk and print the project tree with AST details."""
    if depth > max_depth:
        return

    entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    # Filter
    entries = [
        e for e in entries
        if e.name not in SKIP_DIRS and e.name not in SKIP_FILES
        and not e.name.endswith(".pyc")
    ]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = CORNER if is_last else TEE
        next_prefix = prefix + (INDENT_LAST if is_last else INDENT_GUIDE)

        if entry.is_dir():
            # Check if it's a Python package
            is_pkg = (entry / "__init__.py").exists()
            label = f"{ICONS['dir']} {entry.name}/" + (" [package]" if is_pkg else "")
            print(f"{prefix}{connector}{label}")
            walk_tree(entry, next_prefix, depth + 1, max_depth)

        elif entry.suffix == ".py":
            items = parse_python_file(entry)
            detail = f"  ({len(items)} symbols)" if items else ""
            print(f"{prefix}{connector}{ICONS['py']} {entry.name}{detail}")
            _print_items(items, next_prefix)

        elif entry.suffix == ".ipynb":
            items = parse_notebook(entry)
            detail = f"  ({len(items)} symbols)" if items else ""
            print(f"{prefix}{connector}{ICONS['ipynb']} {entry.name}{detail}")
            _print_items(items, next_prefix)

        elif entry.suffix in (".yaml", ".yml", ".json", ".txt", ".md", ".joblib", ".npy"):
            print(f"{prefix}{connector}{ICONS['file']} {entry.name}")


def main():
    root = Path(__file__).resolve().parent
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).resolve()
    print(f"\n{'='*80}")
    print(f"  PROJECT STRUCTURE MAP: {root.name}")
    print(f"{'='*80}\n")
    print(f"{ICONS['dir']} {root.name}/")
    walk_tree(root, prefix="")
    print(f"\n{'='*80}")
    print("  TIP: Look for duplicate function names, similar class names,")
    print("  or modules with overlapping responsibilities.")
    print(f"{'='*80}\n")
    find_duplicates(root)

if __name__ == "__main__":
    main()