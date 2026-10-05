"""Repository sanity checks for CI/local development."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
required = [
    "README.md", "VERSION", "requirements.txt", "pyproject.toml", ".gitignore",
    "src/koa_severity", "tests", "notebooks", "archive/raw_notebooks",
]
missing = [p for p in required if not (ROOT / p).exists()]
if missing:
    raise SystemExit(f"Missing required repository paths: {missing}")

version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
if version != "0.8.0":
    raise SystemExit(f"VERSION must be 0.8.0 for this release, found {version!r}")

notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
if len(notebooks) != 9:
    raise SystemExit(f"Expected 9 curated notebooks, found {len(notebooks)}")

for path in notebooks:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("metadata", {}).get("koa_repository_version") != "0.8.0":
        raise SystemExit(f"Notebook metadata version mismatch: {path}")
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            raise SystemExit(f"Notebook still contains outputs: {path}")
        source = "".join(cell.get("source", []))
        forbidden = ("google.colab", "drive.mount", "kagglehub.dataset_download", "!pip install", "%pip install", "/content")
        hits = [x for x in forbidden if x in source]
        if hits:
            raise SystemExit(f"Non-portable notebook code in {path}: {hits}")
        try:
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            raise SystemExit(f"Notebook syntax error in {path}: {exc}") from exc

print(f"Repository validation passed: {len(notebooks)} curated notebooks checked.")
