from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROUTER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROUTER_ROOT / "config" / "router.yaml"


class ConfigError(RuntimeError):
    pass


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.environ.get("ROUTER_CONFIG", DEFAULT_CONFIG_PATH)).resolve()
    if not config_path.is_file():
        raise ConfigError(f"router config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigError("router config root must be a mapping")

    models_path = Path(payload.get("models_config", ROUTER_ROOT / "config" / "models.yaml"))
    if not models_path.is_absolute():
        models_path = (config_path.parent / models_path).resolve()
    if not models_path.is_file():
        raise ConfigError(f"models config not found: {models_path}")
    models_payload = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
    if not isinstance(models_payload, dict):
        raise ConfigError("models config root must be a mapping")
    payload["models"] = models_payload.get("models", models_payload)
    payload["config_path"] = str(config_path)
    payload["models_config_path"] = str(models_path)
    return payload


def resolve_router_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROUTER_ROOT / path).resolve()
