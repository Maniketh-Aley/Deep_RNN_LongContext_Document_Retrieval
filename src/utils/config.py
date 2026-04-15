from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: Dict[str, Any], config_path: str | Path) -> None:
    with Path(config_path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

