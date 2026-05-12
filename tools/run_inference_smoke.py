import sys
import os
from pathlib import Path
import csv

# ensure project root is on sys.path so `launcher` package imports work
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from launcher.method_scripts.siamese_unet_cd import SiameseUnetCdRunner

DATASET_ROOT = Path(r"C:\Coding\AI\Dataset\!tests\AI_TEST")

runner = SiameseUnetCdRunner("siamese_unet_cd")
outdir = Path("results/training/siamese_unet_cd_smoke2/inference_examples")
outdir.mkdir(parents=True, exist_ok=True)

rows = []
for pair_dir in sorted([path for path in DATASET_ROOT.iterdir() if path.is_dir()]):
    before = pair_dir / "before.png"
    after = pair_dir / "AI_after.png"
    if before.exists() and after.exists():
        rows.append({"split": "val", "before_path": str(before), "after_path": str(after)})

val_rows = [r for r in rows if r['split'] == 'val'][:3]
if not val_rows:
    print('No val rows found in C:\\Coding\\AI\\Dataset\\!tests\\AI_TEST')

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
