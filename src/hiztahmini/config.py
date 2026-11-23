"""YAML yapılandırma dosyasını okuma."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Config dosyasını sözlük olarak döndürür. Göreli yollar config dosyasına göre çözülür."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path.resolve())
    return cfg
