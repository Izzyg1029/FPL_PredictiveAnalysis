"""scripts/make_release_zip.py

Create a shareable zip of the repo. Excludes large/raw folders by default.
"""
from pathlib import Path
import zipfile
import argparse

DEFAULT_EXCLUDES = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "data/raw", "data/processed/time_series", "models"
}

def should_skip(path: Path, root: Path, excludes: set[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    for ex in excludes:
        if rel == ex or rel.startswith(ex + "/"):
            return True
    return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="FPL_PredictiveAnalysis_release.zip")
    p.add_argument("--include_models", action="store_true")
    args = p.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()

    excludes = set(DEFAULT_EXCLUDES)
    if args.include_models:
        excludes.discard("models")

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if should_skip(path, root, excludes):
                continue
            z.write(path, arcname=path.relative_to(root))
    print(f"✅ Wrote zip: {out}")

if __name__ == "__main__":
    main()
