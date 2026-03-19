from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def resolve_data_root(env_var: str, default_leaf: str) -> Path:
    env_value = os.getenv(env_var)
    candidates: list[Path] = []
    if env_value:
        candidates.append(Path(env_value).expanduser())

    candidates.extend(
        [
            WORKSPACE_ROOT / "data" / "raw" / default_leaf,
            PROJECT_ROOT / "data" / "raw" / default_leaf,
            Path.home() / "data" / "raw" / default_leaf,
            Path.home() / "projects" / "data" / "raw" / default_leaf,
            Path("/Users/joyce/projects/data/raw") / default_leaf,
            Path("/home/joyce/projects/data/raw") / default_leaf,
        ]
    )

    unique_candidates = _dedupe_paths(candidates)
    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    return unique_candidates[0]


def results_root() -> Path:
    return PROJECT_ROOT / "results"
