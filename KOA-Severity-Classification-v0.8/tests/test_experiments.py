from pathlib import Path
from koa_severity.experiments import config_from_yaml, write_run_metadata

def test_config_loads():
    cfg=config_from_yaml(Path('configs/resnet18_baseline.yaml'))
    assert cfg.num_classes == 5
    assert cfg.seed == 2026

def test_run_metadata(tmp_path):
    cfg=config_from_yaml(Path('configs/resnet18_baseline.yaml'))
    p=write_run_metadata(tmp_path,'configs/resnet18_baseline.yaml',cfg)
    assert p.exists()
