import sys
import os
from pathlib import Path
import csv

# ensure project root is on sys.path so `launcher` package imports work
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from launcher.method_scripts.siamese_unet_cd import SiameseUnetCdRunner

runner = SiameseUnetCdRunner("siamese_unet_cd")
outdir = Path("results/training/siamese_unet_cd_smoke2/inference_examples")
outdir.mkdir(parents=True, exist_ok=True)

with open("datasets/TACO/cd_pairs/index_smoke2.csv", newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

val_rows = [r for r in rows if r['split'] == 'val'][:3]
if not val_rows:
    print('No val rows found in index_smoke2.csv')

for i, r in enumerate(val_rows):
    before = r['before_path']
    after = r['after_path']
    sub = outdir / f"sample_{i}"
    os.makedirs(sub, exist_ok=True)
    print(f'Running inference for sample {i}:', before, after)
    result = runner.analyze(before, after)
    # copy artifacts produced by the runner into our sample directory for easy inspection
    artifacts = result.artifacts
    import shutil
    for name, path in artifacts.items():
        try:
            src = Path(path)
            if src.exists():
                dst = sub / src.name
                shutil.copy(str(src), str(dst))
        except Exception as e:
            print('Failed to copy', path, e)

print('Wrote outputs to', outdir)
