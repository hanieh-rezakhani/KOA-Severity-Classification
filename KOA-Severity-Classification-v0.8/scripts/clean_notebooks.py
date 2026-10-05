from pathlib import Path
import json, re
ROOT=Path(__file__).resolve().parents[1]
NB_DIR=ROOT/'notebooks'
BOOT='''# Repository bootstrap — portable local/Jupyter execution\nfrom pathlib import Path\nimport os\n\ndef _find_project_root(start: Path) -> Path:\n    for candidate in [start, *start.parents]:\n        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "koa_severity").exists():\n            return candidate\n    return start\n\nPROJECT_ROOT = _find_project_root(Path.cwd())\nif str(PROJECT_ROOT / "src") not in __import__("sys").path:\n    __import__("sys").path.insert(0, str(PROJECT_ROOT / "src"))\nfrom koa_severity.data import get_dataset_path\n\nDATASET_PATH = get_dataset_path()\nOUTPUTS_DIR = PROJECT_ROOT / "results"\nCHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"\nOUTPUTS_DIR.mkdir(exist_ok=True)\nCHECKPOINTS_DIR.mkdir(exist_ok=True)\nprint("Project root:", PROJECT_ROOT)\nprint("Dataset:", DATASET_PATH)\n'''
for p in NB_DIR.glob('*.ipynb'):
    nb=json.loads(p.read_text(encoding='utf-8'))
    cells=nb.get('cells',[])
    # Remove environment/setup commands while retaining research cells.
    for c in cells:
        if c.get('cell_type')!='code': continue
        s=''.join(c.get('source',[]))
        s=re.sub(r'^\s*!pip install[^\n]*\n?', '', s, flags=re.M|re.I)
        s=re.sub(r'^\s*%pip install[^\n]*\n?', '', s, flags=re.M|re.I)
        s=s.replace('import kagglehub\n','').replace('import kagglehub\r\n','')
        s=s.replace('from google.colab import files\n','').replace('from google.colab import drive\n','')
        s=s.replace('from google.colab import files, drive\n','')
        # Dataset downloads become explicit, user-configured dataset paths.
        s=re.sub(r'kagglehub\.dataset_download\(\s*"[^"]+"\s*\)', 'get_dataset_path()', s)
        s=re.sub(r"kagglehub\.dataset_download\(\s*'[^']+'\s*\)", 'get_dataset_path()', s)
        # Portable replacement for Colab's /content root.
        s=s.replace('/content', 'str(PROJECT_ROOT)')
        # Colab-only download calls are replaced by a portable message.
        s=re.sub(r'files\.download\(\s*([^\n]+)\s*\)', r'print("Artifact saved to:", \1)', s)
        # Remove drive mounting statements; the repository uses KOA_DATASET_PATH instead.
        s=re.sub(r'\s*drive\.mount\(\s*[^\)]*\)\s*', '\n# Google Drive mounting is intentionally omitted; set KOA_DATASET_PATH instead.\n', s)
        c['source']=s.splitlines(keepends=True)
        c.pop('outputs',None); c.pop('execution_count',None)
    # One canonical bootstrap cell at the beginning.
    cells.insert(0, {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':BOOT.splitlines(keepends=True)})
    nb['cells']=cells
    nb.setdefault('metadata',{})['koa_repository_version']='0.8.0'
    p.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
