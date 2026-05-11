# Пакет подробных планов по алгоритмам

В этой папке собраны отдельные планы по каждому методу из финального списка. Каждый файл описывает цель, данные, подготовку, этапы реализации, метрики, риски и ожидаемые артефакты.

## Состав
1. [SIFT + RANSAC + difference map](01_sift_ransac_difference_map.md)
2. [ORB + RANSAC + difference map](02_orb_ransac_difference_map.md)
3. [YOLOv8-detection](03_yolov8_detection.md)
4. [Faster R-CNN](04_faster_rcnn.md)
5. [Mask R-CNN](05_mask_rcnn.md)
6. [YOLOv8-seg](06_yolov8_seg.md)
7. [U-Net segmentation](07_unet_segmentation.md)
8. [DeepLabV3+ segmentation](08_deeplabv3plus_segmentation.md)
9. [SegFormer segmentation](09_segformer_segmentation.md)
10. [Siamese U-Net change detection](10_siamese_unet_change_detection.md)
11. [BIT-like change detection model](11_bit_like_change_detection.md)
12. [ChangeFormer](12_changeformer.md)
13. [Open-CD baseline models](13_open_cd_baselines.md)
14. [ResNet classifier для чисто/грязно](14_resnet_classifier_clean_dirty.md)
15. [EfficientNet classifier для чисто/грязно](15_efficientnet_classifier_clean_dirty.md)
16. [Гибридный score](16_hybrid_score.md)
17. [DINOv2 cleanup degree 0-100](17_dinov2.md)

## Как пользоваться
1. Сначала пройтись по baseline-методам.
2. Потом сравнить детекцию и сегментацию.
3. Затем проверить change detection.
4. После этого обучить вспомогательные классификаторы.
5. Завершить сборкой гибридного score.

## Conda окружения
Для каждого метода есть отдельный `environment.yml` в папке [`conda_envs/`](../conda_envs). Название окружения и файл уже привязаны к выбору метода в `launcher`.
