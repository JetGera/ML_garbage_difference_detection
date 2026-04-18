# Conda environments

В этой папке лежат отдельные `environment.yml`-файлы для каждого метода.

Пример создания окружения:

```powershell
conda env create -f conda_envs\sift_ransac.yml
```

Обновление существующего окружения:

```powershell
conda env update -f conda_envs\sift_ransac.yml --prune
```

Сопоставление метода и файла задается в `launcher/runners.py`.