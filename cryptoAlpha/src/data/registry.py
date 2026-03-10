"""Dataset registry for cryptoAlpha data sources."""
from __future__ import annotations

from pathlib import Path
import yaml


class DataRegistry:
    def __init__(self, config_path: str = "configs/paths.yaml") -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = self.project_root / cfg_path

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def coinglass_root(self) -> Path:
        return self._resolve_path(self.cfg["sources"]["coinglass_root"])

    @property
    def coingecko_root(self) -> Path:
        return self._resolve_path(self.cfg["sources"]["coingecko_root"])

    @property
    def cryptoracle_root(self) -> Path:
        return self._resolve_path(self.cfg["sources"]["cryptoracle_root"])

    @property
    def default_frequency(self) -> str:
        return self.cfg["defaults"]["frequency"]

    @property
    def default_exchange(self) -> str:
        return self.cfg["defaults"]["exchange"]

    @property
    def cache_dir(self) -> Path:
        return self._resolve_path(self.cfg["storage"]["cache_dir"])

    @property
    def feature_dir(self) -> Path:
        return self._resolve_path(self.cfg["storage"]["feature_dir"])

    @property
    def label_dir(self) -> Path:
        return self._resolve_path(self.cfg["storage"]["label_dir"])

    def ensure_storage_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)