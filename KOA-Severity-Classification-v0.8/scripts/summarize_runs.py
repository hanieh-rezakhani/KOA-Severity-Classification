"""Build a compact CSV index from results/runs/*/run.json files."""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',default='results/runs'); p.add_argument('--output',default='results/experiment_summary.csv'); a=p.parse_args()
 rows=[]
 for f in sorted(Path(a.root).glob('*/run.json')):
  d=json.loads(f.read_text()); row={'run':f.parent.name,'created_utc':d.get('created_utc'),'config_path':d.get('config_path')}; row.update(d.get('config',{})); rows.append(row)
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 if rows:
  fields=sorted({k for r in rows for k in r});
  with out.open('w',newline='',encoding='utf-8') as h:
   w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
 print(out)
if __name__=='__main__': main()
