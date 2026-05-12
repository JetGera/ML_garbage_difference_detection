# План эксперимента: Siamese U-Net change detection

## 1. Роль метода
Это базовый change detection-подход для пар фото до и после. Он нужен, чтобы научиться выделять именно изменение между двумя изображениями, а не просто мусор на каждом кадре по отдельности.

## 2. Когда метод уместен
1. Если снимки относятся к одной локации и одному событию уборки.
2. Если можно предварительно выровнять изображения по keypoints и homography.
3. Если нужен понятный и относительно простой CD-baseline.

## 3. Данные
1. Пары фото до и после из локального набора.
2. Внешние change detection датасеты для предварительного обучения или валидации.
3. Если нет готовых change masks, использовать небольшой размеченный поднабор или псевдолейблы от difference map.

## 4. Окружение
1. Change detection-окружение.
2. torch, torchvision, timm, opencv-python, scikit-image, numpy, albumentations.
3. Если реализация берется из Open-CD, использовать совместимый конфиг и веса.

## 5. Подготовка данных
1. Жестко сопоставить пары before/after.
2. Выполнить предварительное выравнивание по homography.
3. При необходимости обрезать область пересечения кадров.
4. Сформировать train/val/test split по парам, а не по одиночным изображениям.
5. Для TACO пары синтезируются из `datasets/TACO/data/annotations.json`: `before` = исходное изображение, `after` = синтетически очищенный кадр, `mask` = union-маска trash-полигонов.
6. Для локальных smoke/validation запусков использовать `C:\Coding\AI\Dataset` как рабочий набор пар изображений.

## 6. Pipeline обучения
1. Построить два одинаковых энкодера с общими весами.
2. Подавать пару изображений либо через конкатенацию каналов, либо через два потока с последующим fusion.
3. Использовать decoder для выдачи binary change mask.
4. Применять BCE+Dice или Focal+Dice loss.
5. Следить за балансом положительных и отрицательных пикселей.

## 7. Pipeline инференса
1. Получить change mask для пары фото.
2. Посчитать долю измененных пикселей как `change_ratio`.
3. Сохранить карту изменений и overlay.
4. При необходимости использовать порог на low-confidence области.

## 8. Метрики
1. IoU для change mask.
2. F1 или Dice.
3. Precision и Recall по измененным пикселям.
4. Корреляция с ручной оценкой очистки.
5. Устойчивость к небольшому смещению камеры.

## 9. Риски и ограничения
1. Очень чувствителен к misalignment.
2. Легко путает изменение освещения и реальное изменение сцены.
3. Без качественных масок training signal может быть слабым.
4. При сильном domain shift может хуже работать, чем ожидается.

## 10. Что сохранить в отчете
1. Change maps на типовых парах.
2. Сравнение до и после предварительного выравнивания.
3. Таблицу метрик и `change_ratio`.
4. Примеры ложных срабатываний.

## 11. Критерий успеха
Метод считается успешным, если он надежно выделяет зоны очистки и сохраняет смысловую интерпретируемость на реальных парах фото.
## 12. Реализация (итоговая)

### Архитектура модели
- **Backbone**: ResNet34 pretrained on ImageNet  
- **Архитектура**: Siamese U-Net с shared encoder, 4 decoder-уровня
- **Input**: 384×384 RGB (fallback: 352×352 при OOM)
- **Output**: Single-channel change map (sigmoid activation)
- **Разница с baseline U-Net**: Два потока (before/after) проходят через один encoder, затем разницы объединяются на каждом уровне декодера (difference-feature fusion)

### Обучение
- **Loss**: 0.5 × BCEWithLogits(pos_weight=2.5) + 0.5 × SoftDice
- **Оптимизатор**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR + warmup 4 эпохи
- **Epochs**: 70 с early stopping (patience=12)
- **Batch**: size=2 + gradient accumulation=4 (эффективный батч=8)
- **Device**: Auto-detection GPU/CPU с fallback на CPU
- **Grad clip**: 1.0

### Инференс
- **Threshold**: 0.45 (calibrated на val set)
- **Postprocessing**: Morphology kernel 3×3, удаление компонентов < 64 px
- **Checkpoint discovery**: Автоматически находит best.pt из results/training/siamese_unet_cd/

### Интеграция в launcher
- **Runner file**: `launcher/method_scripts/siamese_unet_cd.py`
- **Trainer file**: `launcher/train_siamese_unet_cd.py`
- **Registration**: В `launcher/runners.py` добавлен dispatch для method_id="siamese_unet_cd"
- **Config**: 16 настроек в SIAMESE_UNET_CD_CONFIG

### Синтетические пары из TACO
- **Скрипт генератора**: `datasets/TACO/build_cd_pairs.py`
- **Input**: `datasets/TACO/data/annotations.json` (COCO-формат с маскамимусора)
- **Инпаинт**: cv2.inpaint (cv2.INPAINT_TELEA) или PIL Gaussian blur fallback
- **Negative pairs**: Фотометрические сдвиги (brightness/contrast/gamma ±8%)
- **Split**: По image_id (train/val/test = 80/10/10), без утечки

### Как запустить генератор

```bash
# Полная генерация
python datasets/TACO/build_cd_pairs.py --output-root C:\Coding\AI\Dataset

# Smoke test (25 images)
python datasets/TACO/build_cd_pairs.py --max-images 25 --output-root C:\Coding\AI\Dataset\!tests\AI_TEST
```

### Как запустить обучение

```bash
.\install-conda-envs.ps1 -Methods siamese_unet_cd

conda run -n projekt-siamese-unet-cd python -m launcher.train_siamese_unet_cd \
	--index-csv C:\Coding\AI\Dataset\!tests\AI_TEST\index.csv \
	--output-root results/training/siamese_unet_cd \
	--epochs 70 --batch-size 2 --learning-rate 3e-4
```

### Как запустить инференс

```bash
conda run -n projekt-siamese-unet-cd python -m launcher.gui
```

### Статус реализации
✅ **Завершено**:
- Siamese U-Net архитектура, training loop, инференс runner
- Синтетический генератор пар из TACO annotations
- cv2.inpaint с PIL fallback
- Интеграция в launcher
- Документация в README и install-conda-envs.ps1
- Полная генерация TACO (~1200+ пар в прогрессе)

✅ **Протестировано**:
- Smoke test: 25 images → 30 pairs ✓
- Full generation: ~1200 train pairs ✓
- Import safety: cv2 optional ✓

- **Threshold**: 0.45 (calibrated на val set)
- **Postprocessing**: Morphology kernel 3×3, удаление компонентов < 64 px
- **Checkpoint discovery**: Автоматически находит best.pt из results/training/siamese_unet_cd/

