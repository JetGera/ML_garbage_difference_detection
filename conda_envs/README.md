# Conda environments

В этой папке лежат отдельные `environment.yml`-файлы для каждого метода.

Для быстрой установки из корня репозитория можно использовать [install-conda-envs.ps1](../install-conda-envs.ps1).

Пример создания окружения:

```powershell
conda env create -f conda_envs\sift_ransac.yml
```

Обновление существующего окружения:

```powershell
conda env update -f conda_envs\sift_ransac.yml --prune
```

Сопоставление метода и файла задается в `launcher/runners.py`.