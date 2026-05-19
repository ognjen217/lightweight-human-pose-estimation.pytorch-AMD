#!/usr/bin/env python3
from pathlib import Path
import argparse

DEFAULT_IGNORE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "build",
    "dist",
    "outputs",
    "runs",
    "logs",
    "rocprof_results",
}

DEFAULT_IGNORE_FILES = {
    "FILETREE.md",
    ".DS_Store",
}

def should_ignore(path: Path, ignore_dirs, ignore_files):
    if path.is_dir() and path.name in ignore_dirs:
        return True
    if path.is_file() and path.name in ignore_files:
        return True
    return False

def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.1f} MB"
    return f"{num_bytes / (1024 ** 3):.1f} GB"

def build_tree(
    root: Path,
    max_depth: int,
    show_size: bool,
    ignore_dirs,
    ignore_files,
):
    lines = []
    files_index = []

    root = root.resolve()
    lines.append(f"# File tree")
    lines.append("")
    lines.append(f"Root: `{root}`")
    lines.append("")
    lines.append("```text")
    lines.append(f"{root.name}/")

    def walk(current: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return

        try:
            entries = sorted(
                [p for p in current.iterdir() if not should_ignore(p, ignore_dirs, ignore_files)],
                key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            lines.append(prefix + "└── [Permission denied]")
            return

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            branch = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            rel_path = entry.relative_to(root)

            if entry.is_dir():
                lines.append(prefix + branch + entry.name + "/")
                walk(entry, next_prefix, depth + 1)
            else:
                size_text = ""
                if show_size:
                    try:
                        size_text = f" ({format_size(entry.stat().st_size)})"
                    except OSError:
                        size_text = ""
                lines.append(prefix + branch + entry.name + size_text)
                files_index.append(str(rel_path))

    walk(root)
    lines.append("```")
    lines.append("")

    lines.append("## File index")
    lines.append("")
    lines.append("Ovo je lista relativnih putanja koje možeš da koristiš u promptu:")
    lines.append("")

    for path in files_index:
        lines.append(f"- `{path}`")

    lines.append("")
    lines.append("## Suggested prompt format")
    lines.append("")
    lines.append("```text")
    lines.append("Na osnovu ovog FILETREE.md, fokusiraj se na sledeće fajlove:")
    lines.append("- path/to/file1.py")
    lines.append("- path/to/file2.py")
    lines.append("")
    lines.append("Zadatak:")
    lines.append("...")
    lines.append("```")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="Generate FILETREE.md from the current project folder."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root folder to scan. Default: current folder."
    )
    parser.add_argument(
        "--output",
        default="FILETREE.md",
        help="Output markdown file. Default: FILETREE.md"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum folder depth to scan. Default: 6"
    )
    parser.add_argument(
        "--show-size",
        action="store_true",
        help="Show file sizes in the tree."
    )
    parser.add_argument(
        "--ignore-dir",
        action="append",
        default=[],
        help="Additional directory name to ignore. Can be used multiple times."
    )
    parser.add_argument(
        "--ignore-file",
        action="append",
        default=[],
        help="Additional file name to ignore. Can be used multiple times."
    )

    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)

    ignore_dirs = DEFAULT_IGNORE_DIRS | set(args.ignore_dir)
    ignore_files = DEFAULT_IGNORE_FILES | set(args.ignore_file)

    tree = build_tree(
        root=root,
        max_depth=args.max_depth,
        show_size=args.show_size,
        ignore_dirs=ignore_dirs,
        ignore_files=ignore_files,
    )

    output.write_text(tree, encoding="utf-8")
    print(f"Generated: {output.resolve()}")

if __name__ == "__main__":
    main()