"""Experiment configuration loading and run metadata."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
from .config import ExperimentConfig

def load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load experiment configs") from exc
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict): raise ValueError("Experiment config must be a mapping")
    return data

def config_from_yaml(path: str | Path) -> ExperimentConfig:
    d=load_yaml(path)
    allowed={k:v for k,v in d.items() if k in ExperimentConfig.__dataclass_fields__}
    return ExperimentConfig(**allowed)

def write_run_metadata(out_dir: str | Path, config_path: str | Path, config: ExperimentConfig, extra: dict | None=None) -> Path:
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    payload={"created_utc":datetime.now(timezone.utc).isoformat(),"config_path":str(config_path),"config":asdict(config)}
    if extra: payload["extra"]=extra
    p=out/"run.json"; p.write_text(json.dumps(payload,indent=2),encoding="utf-8"); return p
