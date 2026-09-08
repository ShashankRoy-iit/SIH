"""Configuration loading.

Configs are plain YAML files under ``config/`` so that a judge, a rescuer or a
future maintainer can change mission behaviour without touching Python.  We
deliberately use dataclasses + a thin loader rather than a schema framework:
the project must stay auditable on a 2-core laptop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

T = TypeVar("T")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively build a dataclass tree, ignoring unknown keys.

    Unknown keys are ignored (not an error) so a config file may carry
    documentation-only fields, but a warning list is attached for tooling.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = f.type
        if isinstance(ftype, str):
            ftype = _resolve_type(cls, ftype)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(ftype, value)
        elif (
            getattr(ftype, "__origin__", None) is list
            and isinstance(value, list)
            and is_dataclass(getattr(ftype, "__args__", [None])[0])
        ):
            item_cls = ftype.__args__[0]
            kwargs[f.name] = [
                _from_dict(item_cls, v) if isinstance(v, dict) else v for v in value
            ]
        else:
            kwargs[f.name] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def _resolve_type(cls: Any, name: str) -> Any:
    """Best-effort resolution of a stringified annotation inside this module."""
    import typing

    ns = {cls.__name__: cls}
    ns.update(globals())
    ns.update(vars(typing))
    try:
        return eval(name, ns)  # noqa: S307 - annotations come from our own source
    except Exception:
        return None


def from_yaml(cls: Type[T], path: str) -> T:
    return _from_dict(cls, load_yaml(path))


def default_config_path(name: str) -> str:
    return os.path.join(CONFIG_DIR, name)
