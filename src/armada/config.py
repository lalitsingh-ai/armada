"""Configuration loading.

Loads a YAML file into typed dataclasses and supports dotted-path overrides from the CLI
(e.g. ``--set run.concurrency=16 --set cost.usd_per_hour=0.06``).

Note: this module deliberately does *not* use ``from __future__ import annotations`` so that
``dataclasses.fields(...).type`` resolves to real classes (needed to recurse into the nested
config sections).
"""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunCfg:
    name: str = "default"
    concurrency: int = 8
    repeats: int = 1
    latency_sla_s: float = 15.0
    seed: int = 1234


@dataclass
class CostCfg:
    usd_per_hour: float = 0.0
    instance_label: str = "unknown"


@dataclass
class ServerCfg:
    engine: str = "llama.cpp"
    host: str = "127.0.0.1"
    port: int = 8080
    model_path: str = "models/model.gguf"
    threads: int | None = None
    parallel: int = 8
    continuous_batching: bool = True
    prompt_cache: bool = True
    ctx_size: int = 4096
    startup_timeout_s: int = 120
    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class ModelCfg:
    name: str = "qwen2.5-3b-instruct-q4_0"
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass
class TasksCfg:
    suite: str = "default"


@dataclass
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    cost: CostCfg = field(default_factory=CostCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    tasks: TasksCfg = field(default_factory=TasksCfg)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "Config":
        data: dict[str, Any] = {}
        if path:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        cfg = _from_dict(cls, data)
        for ov in overrides or []:
            _apply_override(cfg, ov)
        return cfg


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(f.type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _coerce(current: Any, raw: str) -> Any:
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if current is None:
        # Best-effort: try int, then float, then string.
        for conv in (int, float):
            try:
                return conv(raw)
            except ValueError:
                continue
    return raw


def _apply_override(cfg: Any, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}', expected key.path=value")
    key, raw = override.split("=", 1)
    parts = key.split(".")
    target = cfg
    for part in parts[:-1]:
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ValueError(f"Unknown config key: {key}")
    setattr(target, leaf, _coerce(getattr(target, leaf), raw))


def load_instance(name: str, path: str | Path = "configs/instances.yaml") -> tuple[str, float]:
    """Look up a cloud instance preset, returning ``(label, usd_per_hour)``.

    Presets live in ``configs/instances.yaml`` and feed the cost model so that cost-per-task and
    tasks-per-dollar come out in real money.
    """
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    instances = data.get("instances", {})
    if name not in instances:
        known = ", ".join(sorted(instances)) or "(none)"
        raise ValueError(f"Unknown instance '{name}'. Known presets: {known}")
    entry = instances[name]
    return str(entry["label"]), float(entry["usd_per_hour"])

