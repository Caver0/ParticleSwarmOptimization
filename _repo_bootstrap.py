from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_src_path() -> None:
    """Allow repository-root scripts to import the package from src/."""
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    src_dir_str = str(src_dir)

    if src_dir.is_dir() and src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
