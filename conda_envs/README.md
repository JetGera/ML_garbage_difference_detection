# Conda environment

Папка содержит единственный файл `base.yml` — полное окружение для всех методов проекта.

## Архитектура (Simplified: Base Only)

### base.yml — единственное окружение
Содержит **все зависимости** для работы со всеми методами:

**Python & Core:**
- python=3.11, pip
- numpy, scipy, scikit-image, scikit-learn, pillow

**Visualization & Utils:**
- matplotlib, tqdm, seaborn

**Deep Learning (GPU):**
- **pytorch** (conda)
- **torchvision** (conda)
- **pytorch-cuda=12.4** (conda, `nvidia` channel)

**Computer Vision & ML:**
- pycocotools
- albumentations, opencv-python, opencv-contrib-python (pip)
- timm (Transformer Image Models)
- ultralytics (YOLOv8)

## Installation

### Create the base environment
```powershell
.\install-conda-envs.ps1
```

or

```powershell
.\install-conda-envs.ps1 -ListOnly   # Show what will be created
.\install-conda-envs.ps1 -Recreate   # Remove and recreate from scratch
```

## Usage

### Run any method through base environment
```powershell
conda run -n projekt-base python -m launcher.gui
```

### Update environment
```powershell
conda env update -f conda_envs\base.yml --prune
```

### Activate environment manually
```powershell
conda activate projekt-base
python -m launcher.gui
```

## Benefits

- **Single installation:** No redundant torch copies (~5 GB saved)
- **Simpler maintenance:** One env to update, not 7
- **Less confusion:** All methods use the same, well-tested base
- **Faster setup:** One install instead of multiple

## Notes

- All methods (sift_ransac, yolov8_seg, changeformer, dinov2_cd, efficientnet_cls, siamese_unet_cd) require only this base environment
- Conda output (solver, install progress) is displayed in real-time during installation
- PyTorch is installed from conda channels (`pytorch` + `nvidia`) for better Windows binary compatibility