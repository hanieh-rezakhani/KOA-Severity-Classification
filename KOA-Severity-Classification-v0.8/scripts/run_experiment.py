"""Create a reproducible run directory from a YAML experiment config."""
from __future__ import annotations
import argparse
from pathlib import Path
from koa_severity.experiments import config_from_yaml, write_run_metadata
from koa_severity.reproducibility import seed_everything

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--run-dir',default=None); args=p.parse_args()
 cfg=config_from_yaml(args.config); seed_everything(cfg.seed)
 run=Path(args.run_dir or Path('results/runs')/Path(args.config).stem)
 write_run_metadata(run,args.config,cfg)
 print(run/'run.json')
if __name__=='__main__': main()
