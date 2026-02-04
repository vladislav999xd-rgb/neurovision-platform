# MLOps - Machine Learning Operations для NeuroVision Platform

## Обзор MLOps архитектуры

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                              MLOps PIPELINE                                        │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│   DATA LAYER                    ML LAYER                    SERVING LAYER         │
│   ┌─────────────┐              ┌─────────────┐              ┌─────────────┐       │
│   │             │              │             │              │             │       │
│   │  Data       │              │  Training   │              │  Model      │       │
│   │  Pipeline   │─────────────▶│  Pipeline   │─────────────▶│  Serving    │       │
│   │             │              │             │              │             │       │
│   └──────┬──────┘              └──────┬──────┘              └──────┬──────┘       │
│          │                            │                            │             │
│          ▼                            ▼                            ▼             │
│   ┌─────────────┐              ┌─────────────┐              ┌─────────────┐       │
│   │  DVC        │              │  MLflow     │              │  Triton     │       │
│   │  (Versions) │              │  (Tracking) │              │  (Inference)│       │
│   └─────────────┘              └─────────────┘              └─────────────┘       │
│                                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────────┐   │
│   │                         ORCHESTRATION (Airflow/Kubeflow)                   │   │
│   └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────────┐   │
│   │                         MONITORING (Prometheus + Grafana)                   │   │
│   └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Версионирование данных (DVC)

### Структура данных

```
data/
├── raw/                          # Исходные данные
│   ├── videos/                   # Видео операций
│   │   ├── operation_001.mp4
│   │   └── operation_002.mp4
│   ├── annotations/              # Разметка
│   │   ├── phases/              # Фазы операций
│   │   ├── instruments/         # Инструменты
│   │   ├── anatomy/             # Анатомия
│   │   └── events/              # События
│   └── metadata/                 # Метаданные
│       └── operations.json
│
├── processed/                    # Обработанные данные
│   ├── frames/                   # Извлечённые кадры
│   ├── features/                 # Feature vectors
│   └── splits/                   # Train/Val/Test splits
│
└── external/                     # Внешние датасеты
    ├── cholec80/                 # Публичный датасет
    └── m2cai16/                  # Публичный датасет
```

### DVC конфигурация

```yaml
# dvc.yaml
stages:
  extract_frames:
    cmd: python src/data/extract_frames.py
    deps:
      - data/raw/videos
      - src/data/extract_frames.py
    params:
      - extract_frames.fps
      - extract_frames.resolution
    outs:
      - data/processed/frames

  create_annotations:
    cmd: python src/data/create_annotations.py
    deps:
      - data/raw/annotations
      - data/processed/frames
    outs:
      - data/processed/annotations.json

  create_splits:
    cmd: python src/data/create_splits.py
    deps:
      - data/processed/annotations.json
    params:
      - splits.train_ratio
      - splits.val_ratio
      - splits.test_ratio
      - splits.random_seed
    outs:
      - data/processed/splits/train.json
      - data/processed/splits/val.json
      - data/processed/splits/test.json

  extract_features:
    cmd: python src/data/extract_features.py
    deps:
      - data/processed/frames
      - data/processed/splits
    params:
      - features.model
      - features.batch_size
    outs:
      - data/processed/features
```

### Параметры (params.yaml)

```yaml
# params.yaml
extract_frames:
  fps: 2.0
  resolution: [640, 480]
  format: "jpg"
  quality: 95

splits:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42
  stratify_by: "operation_type"

features:
  model: "resnet50"
  batch_size: 32
  normalize: true

training:
  phase_segmentation:
    epochs: 100
    batch_size: 8
    learning_rate: 0.001
    weight_decay: 0.0001
    num_stages: 4
    num_layers: 10
  
  instrument_detection:
    epochs: 200
    batch_size: 16
    img_size: 640
    learning_rate: 0.01
    augmentation: true
  
  anatomy_segmentation:
    epochs: 150
    batch_size: 4
    learning_rate: 0.0001
    use_pretrained_sam: true
```

### Команды DVC

```bash
# Инициализация
dvc init

# Добавление remote storage (S3)
dvc remote add -d storage s3://neurovision-data/dvc
dvc remote modify storage access_key_id ${AWS_ACCESS_KEY_ID}
dvc remote modify storage secret_access_key ${AWS_SECRET_ACCESS_KEY}

# Добавление данных под контроль версий
dvc add data/raw/videos
dvc add data/raw/annotations

# Запуск пайплайна
dvc repro

# Получение данных
dvc pull

# Просмотр метрик
dvc metrics show
```

---

## 2. Model Registry (MLflow)

### Структура экспериментов

```
mlflow/
├── experiments/
│   ├── phase_segmentation/
│   │   ├── run_001/
│   │   ├── run_002/
│   │   └── run_003/
│   ├── instrument_detection/
│   └── anatomy_segmentation/
│
└── models/
    ├── phase_segmenter/
    │   ├── v1.0.0/
    │   ├── v1.1.0/
    │   └── production/  (alias)
    ├── instrument_detector/
    └── anatomy_segmenter/
```

### Пример интеграции

```python
"""
MLflow Training Integration
Интеграция MLflow для трекинга экспериментов
"""
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
from typing import Dict, Any

class MLflowTracker:
    """Трекер экспериментов на базе MLflow"""
    
    def __init__(
        self,
        tracking_uri: str = "http://mlflow-server:5000",
        experiment_name: str = "default"
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
    
    def start_run(
        self,
        run_name: str,
        tags: Dict[str, str] = None
    ):
        """Начинает новый run"""
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        return self.run
    
    def log_params(self, params: Dict[str, Any]):
        """Логирует параметры"""
        mlflow.log_params(params)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int = None
    ):
        """Логирует метрики"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str,
        registered_model_name: str = None
    ):
        """Логирует и регистрирует модель"""
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Логирует артефакт (файл)"""
        mlflow.log_artifact(local_path, artifact_path)
    
    def end_run(self):
        """Завершает run"""
        mlflow.end_run()
    
    def promote_model(
        self,
        model_name: str,
        version: int,
        stage: str = "Production"
    ):
        """Переводит модель на следующий stage"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True
        )
    
    def get_production_model(self, model_name: str):
        """Получает production версию модели"""
        model_uri = f"models:/{model_name}/Production"
        return mlflow.pytorch.load_model(model_uri)


# Пример использования
def train_phase_segmenter(config: Dict):
    """Обучение модели с MLflow трекингом"""
    
    tracker = MLflowTracker(
        experiment_name="phase_segmentation"
    )
    
    with tracker.start_run(run_name=f"train_{config['run_id']}"):
        # Логируем параметры
        tracker.log_params({
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "num_stages": config["num_stages"],
            "num_layers": config["num_layers"]
        })
        
        # Создаём модель и датасет
        model = create_model(config)
        train_loader, val_loader = create_dataloaders(config)
        
        # Обучение
        for epoch in range(config["epochs"]):
            train_loss = train_epoch(model, train_loader)
            val_metrics = validate(model, val_loader)
            
            # Логируем метрики
            tracker.log_metrics({
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_temporal_iou": val_metrics["temporal_iou"]
            }, step=epoch)
            
            # Сохраняем лучшую модель
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                tracker.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name="phase_segmenter"
                )
        
        # Логируем финальные артефакты
        tracker.log_artifact("confusion_matrix.png")
        tracker.log_artifact("training_curves.png")
```

### MLflow Model Registry

```python
"""
Model Registry Operations
Операции с реестром моделей
"""
from mlflow.tracking import MlflowClient

def register_and_promote_model(
    run_id: str,
    model_name: str,
    metrics_threshold: Dict[str, float]
):
    """
    Регистрирует модель и переводит в Production,
    если метрики выше порога
    """
    client = MlflowClient()
    
    # Получаем метрики run
    run = client.get_run(run_id)
    metrics = run.data.metrics
    
    # Проверяем пороги
    all_passed = all(
        metrics.get(metric, 0) >= threshold
        for metric, threshold in metrics_threshold.items()
    )
    
    if all_passed:
        # Регистрируем модель
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        
        # Переводим в Staging
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging"
        )
        
        # Автоматические тесты в Staging
        if run_staging_tests(model_name, result.version):
            # Переводим в Production
            client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage="Production",
                archive_existing_versions=True
            )
            
            return {
                "status": "promoted",
                "version": result.version,
                "stage": "Production"
            }
    
    return {"status": "rejected", "reason": "metrics_below_threshold"}
```

---

## 3. Training Pipeline

### Архитектура пайплайна обучения

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │   Data   │   │  Data    │   │ Feature  │   │ Model    │   │ Model    │  │
│  │  Ingest  │──▶│ Validate │──▶│ Extract  │──▶│ Train    │──▶│ Evaluate │  │
│  │          │   │          │   │          │   │          │   │          │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│       │              │              │              │              │         │
│       ▼              ▼              ▼              ▼              ▼         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │   DVC    │   │ Great    │   │  MLflow  │   │  MLflow  │   │  MLflow  │  │
│  │          │   │ Expect.  │   │ Artifact │   │ Tracking │   │ Registry │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Kubeflow Pipeline

```python
"""
Kubeflow Training Pipeline
Пайплайн обучения для Kubeflow
"""
from kfp import dsl
from kfp.components import create_component_from_func

@dsl.component(
    base_image="neurovision/training:latest",
    packages_to_install=["dvc", "mlflow"]
)
def data_ingestion(
    dvc_remote: str,
    data_version: str,
    output_path: dsl.OutputPath()
):
    """Загрузка данных из DVC"""
    import subprocess
    
    subprocess.run([
        "dvc", "pull", "-r", dvc_remote,
        "--rev", data_version
    ])
    
    # Копируем данные в output
    import shutil
    shutil.copytree("data/processed", output_path)


@dsl.component(
    base_image="neurovision/training:latest"
)
def data_validation(
    data_path: dsl.InputPath(),
    validation_report: dsl.OutputPath()
):
    """Валидация данных с Great Expectations"""
    import great_expectations as ge
    import json
    
    # Загружаем датасет
    context = ge.get_context()
    
    # Запускаем валидацию
    results = context.run_checkpoint(
        checkpoint_name="surgical_data_checkpoint"
    )
    
    # Сохраняем отчёт
    with open(validation_report, 'w') as f:
        json.dump(results.to_json_dict(), f)
    
    if not results.success:
        raise ValueError("Data validation failed")


@dsl.component(
    base_image="neurovision/training:latest",
    accelerator_type="nvidia-tesla-v100",
    accelerator_count=1
)
def train_model(
    data_path: dsl.InputPath(),
    model_type: str,
    hyperparameters: dict,
    mlflow_tracking_uri: str,
    model_artifact: dsl.OutputPath()
):
    """Обучение модели"""
    import mlflow
    import torch
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    with mlflow.start_run():
        mlflow.log_params(hyperparameters)
        
        # Обучение (зависит от model_type)
        if model_type == "phase_segmentation":
            model = train_phase_segmenter(data_path, hyperparameters)
        elif model_type == "instrument_detection":
            model = train_instrument_detector(data_path, hyperparameters)
        elif model_type == "anatomy_segmentation":
            model = train_anatomy_segmenter(data_path, hyperparameters)
        
        # Сохраняем модель
        torch.save(model.state_dict(), model_artifact)
        mlflow.pytorch.log_model(model, "model")


@dsl.component(
    base_image="neurovision/training:latest",
    accelerator_type="nvidia-tesla-v100",
    accelerator_count=1
)
def evaluate_model(
    model_artifact: dsl.InputPath(),
    test_data_path: dsl.InputPath(),
    model_type: str,
    metrics_output: dsl.OutputPath()
):
    """Оценка модели на тестовых данных"""
    import torch
    import json
    
    # Загружаем модель
    model = load_model(model_artifact, model_type)
    
    # Оценка
    metrics = evaluate(model, test_data_path, model_type)
    
    # Сохраняем метрики
    with open(metrics_output, 'w') as f:
        json.dump(metrics, f)


@dsl.component(
    base_image="neurovision/training:latest"
)
def register_model(
    model_artifact: dsl.InputPath(),
    metrics_path: dsl.InputPath(),
    model_name: str,
    mlflow_tracking_uri: str,
    metrics_thresholds: dict
):
    """Регистрация модели в MLflow Registry"""
    import mlflow
    import json
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Проверяем пороги
    passed = all(
        metrics.get(m, 0) >= t 
        for m, t in metrics_thresholds.items()
    )
    
    if passed:
        # Регистрируем модель
        mlflow.pytorch.log_model(
            pytorch_model=torch.load(model_artifact),
            artifact_path="model",
            registered_model_name=model_name
        )


@dsl.pipeline(
    name="NeuroVision Training Pipeline",
    description="End-to-end ML training pipeline"
)
def training_pipeline(
    dvc_remote: str = "s3://neurovision-data/dvc",
    data_version: str = "v1.0",
    model_type: str = "phase_segmentation",
    hyperparameters: dict = None,
    mlflow_tracking_uri: str = "http://mlflow:5000"
):
    """Основной пайплайн обучения"""
    
    # Шаг 1: Загрузка данных
    ingest_task = data_ingestion(
        dvc_remote=dvc_remote,
        data_version=data_version
    )
    
    # Шаг 2: Валидация данных
    validate_task = data_validation(
        data_path=ingest_task.outputs["output_path"]
    )
    
    # Шаг 3: Обучение модели
    train_task = train_model(
        data_path=ingest_task.outputs["output_path"],
        model_type=model_type,
        hyperparameters=hyperparameters or {},
        mlflow_tracking_uri=mlflow_tracking_uri
    )
    train_task.after(validate_task)
    
    # Шаг 4: Оценка модели
    eval_task = evaluate_model(
        model_artifact=train_task.outputs["model_artifact"],
        test_data_path=ingest_task.outputs["output_path"],
        model_type=model_type
    )
    
    # Шаг 5: Регистрация модели
    register_task = register_model(
        model_artifact=train_task.outputs["model_artifact"],
        metrics_path=eval_task.outputs["metrics_output"],
        model_name=f"neurovision_{model_type}",
        mlflow_tracking_uri=mlflow_tracking_uri,
        metrics_thresholds={"accuracy": 0.85, "f1": 0.80}
    )
```

### Cross-Validation

```python
"""
Cross-Validation Strategy
Стратегия кросс-валидации для хирургических видео
"""
from sklearn.model_selection import GroupKFold
import numpy as np
from typing import List, Tuple, Dict

class SurgicalCrossValidator:
    """
    Кросс-валидатор для хирургических операций.
    Гарантирует, что операции одного хирурга или типа
    не попадают одновременно в train и test.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        group_by: str = "surgeon"  # "surgeon" или "operation_type"
    ):
        self.n_splits = n_splits
        self.group_by = group_by
    
    def split(
        self,
        operations: List[Dict]
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Разбивает операции на фолды
        
        Args:
            operations: Список операций с метаданными
            
        Returns:
            splits: Список (train_indices, test_indices) для каждого фолда
        """
        # Группируем по выбранному признаку
        groups = np.array([op[self.group_by] for op in operations])
        X = np.arange(len(operations))
        
        kfold = GroupKFold(n_splits=self.n_splits)
        
        splits = []
        for train_idx, test_idx in kfold.split(X, groups=groups):
            splits.append((train_idx.tolist(), test_idx.tolist()))
        
        return splits
    
    def get_stratified_splits(
        self,
        operations: List[Dict],
        stratify_by: str = "operation_type"
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Стратифицированное разбиение с балансировкой по типу операции
        """
        from collections import defaultdict
        
        # Группируем по типу операции
        by_type = defaultdict(list)
        for i, op in enumerate(operations):
            by_type[op[stratify_by]].append(i)
        
        # Для каждого типа делаем KFold
        all_splits = [[] for _ in range(self.n_splits)]
        
        for op_type, indices in by_type.items():
            # Shuffle
            np.random.shuffle(indices)
            
            # Распределяем по фолдам
            fold_size = len(indices) // self.n_splits
            for fold in range(self.n_splits):
                start = fold * fold_size
                end = start + fold_size if fold < self.n_splits - 1 else len(indices)
                
                test_idx = indices[start:end]
                train_idx = indices[:start] + indices[end:]
                
                all_splits[fold].append((train_idx, test_idx))
        
        # Объединяем splits
        final_splits = []
        for fold_splits in all_splits:
            train = []
            test = []
            for t_train, t_test in fold_splits:
                train.extend(t_train)
                test.extend(t_test)
            final_splits.append((train, test))
        
        return final_splits
```

---

## 4. Docker Containerization

### Dockerfile для обучения

```dockerfile
# Dockerfile.training
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements-training.txt /app/
WORKDIR /app

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements-training.txt

# Копируем код
COPY src/ /app/src/
COPY configs/ /app/configs/

# Устанавливаем пакет
COPY setup.py /app/
RUN pip install -e .

# Точка входа
ENTRYPOINT ["python", "-m", "src.training.train"]
```

### Dockerfile для инференса

```dockerfile
# Dockerfile.inference
FROM nvcr.io/nvidia/tritonserver:23.04-py3

# Копируем модели
COPY models/ /models/

# Копируем конфигурацию
COPY triton_config/ /models/

# Запускаем Triton
CMD ["tritonserver", "--model-repository=/models"]
```

### Docker Compose для разработки

```yaml
# docker-compose.yml
version: '3.8'

services:
  # MLflow Tracking Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.5.0
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
      - MLFLOW_ARTIFACT_ROOT=s3://neurovision-artifacts/
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    command: mlflow server --host 0.0.0.0 --port 5000
    depends_on:
      - postgres
  
  # PostgreSQL для MLflow
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflow
      - POSTGRES_DB=mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  # MinIO (S3-compatible storage)
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
  
  # Redis для очередей
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  # Triton Inference Server
  triton:
    image: nvcr.io/nvidia/tritonserver:23.04-py3
    ports:
      - "8000:8000"  # HTTP
      - "8001:8001"  # gRPC
      - "8002:8002"  # Metrics
    volumes:
      - ./models:/models
    command: tritonserver --model-repository=/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  # Worker для обработки видео
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - REDIS_URL=redis://redis:6379
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - TRITON_URL=triton:8001
    depends_on:
      - redis
      - mlflow
      - triton
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  # API Gateway
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://app:app@postgres:5432/neurovision
      - REDIS_URL=redis://redis:6379
      - TRITON_URL=triton:8001
    depends_on:
      - postgres
      - redis
      - triton

volumes:
  postgres_data:
  minio_data:
```

---

## 5. Inference Service

### Triton Model Configuration

```protobuf
# models/phase_segmenter/config.pbtxt
name: "phase_segmenter"
platform: "pytorch_libtorch"
max_batch_size: 8

input [
  {
    name: "features"
    data_type: TYPE_FP32
    dims: [ -1, 2048 ]  # Variable sequence length
  }
]

output [
  {
    name: "predictions"
    data_type: TYPE_FP32
    dims: [ -1, 11 ]  # 11 phases
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}

# Model versioning
version_policy: { latest { num_versions: 2 } }
```

### gRPC/REST Client

```python
"""
Triton Inference Client
Клиент для инференса через Triton
"""
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from typing import List, Dict, Union
import asyncio

class TritonInferenceClient:
    """Клиент для Triton Inference Server"""
    
    def __init__(
        self,
        url: str = "localhost:8001",
        use_grpc: bool = True,
        verbose: bool = False
    ):
        self.url = url
        self.use_grpc = use_grpc
        
        if use_grpc:
            self.client = grpcclient.InferenceServerClient(
                url=url, verbose=verbose
            )
        else:
            self.client = httpclient.InferenceServerClient(
                url=url, verbose=verbose
            )
    
    def is_server_ready(self) -> bool:
        """Проверяет готовность сервера"""
        return self.client.is_server_ready()
    
    def is_model_ready(self, model_name: str) -> bool:
        """Проверяет готовность модели"""
        return self.client.is_model_ready(model_name)
    
    def infer_phase_segmentation(
        self,
        features: np.ndarray,  # Shape: (seq_len, 2048)
        model_name: str = "phase_segmenter"
    ) -> np.ndarray:
        """
        Инференс для сегментации фаз
        
        Args:
            features: Feature vectors для последовательности кадров
            model_name: Имя модели в Triton
            
        Returns:
            predictions: Предсказания фаз (seq_len, num_phases)
        """
        if self.use_grpc:
            inputs = [
                grpcclient.InferInput(
                    "features", features.shape, "FP32"
                )
            ]
            outputs = [
                grpcclient.InferRequestedOutput("predictions")
            ]
        else:
            inputs = [
                httpclient.InferInput(
                    "features", features.shape, "FP32"
                )
            ]
            outputs = [
                httpclient.InferRequestedOutput("predictions")
            ]
        
        inputs[0].set_data_from_numpy(features.astype(np.float32))
        
        result = self.client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        return result.as_numpy("predictions")
    
    def infer_instrument_detection(
        self,
        images: np.ndarray,  # Shape: (N, 640, 640, 3)
        model_name: str = "instrument_detector"
    ) -> List[Dict]:
        """
        Инференс для детекции инструментов
        
        Returns:
            detections: Список детекций для каждого изображения
        """
        if self.use_grpc:
            inputs = [
                grpcclient.InferInput("images", images.shape, "FP32")
            ]
            outputs = [
                grpcclient.InferRequestedOutput("boxes"),
                grpcclient.InferRequestedOutput("scores"),
                grpcclient.InferRequestedOutput("classes")
            ]
        else:
            inputs = [
                httpclient.InferInput("images", images.shape, "FP32")
            ]
            outputs = [
                httpclient.InferRequestedOutput("boxes"),
                httpclient.InferRequestedOutput("scores"),
                httpclient.InferRequestedOutput("classes")
            ]
        
        # Normalize и transpose
        images_normalized = images.astype(np.float32) / 255.0
        images_transposed = np.transpose(images_normalized, (0, 3, 1, 2))
        inputs[0].set_data_from_numpy(images_transposed)
        
        result = self.client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        boxes = result.as_numpy("boxes")
        scores = result.as_numpy("scores")
        classes = result.as_numpy("classes")
        
        return self._parse_detections(boxes, scores, classes)
    
    def _parse_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray
    ) -> List[Dict]:
        """Парсит результаты детекции"""
        detections = []
        
        for i in range(len(boxes)):
            if scores[i] > 0.5:
                detections.append({
                    "bbox": boxes[i].tolist(),
                    "score": float(scores[i]),
                    "class_id": int(classes[i])
                })
        
        return detections
    
    async def infer_async(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Асинхронный инференс"""
        # Используем asyncio для async inference
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._sync_infer,
            model_name,
            inputs
        )
```

---

## 6. Queue System

### Redis Queue для видео обработки

```python
"""
Video Processing Queue
Очередь для обработки видео
"""
import redis
from rq import Queue, Worker
from rq.job import Job
import json
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class VideoProcessingJob:
    """Задача обработки видео"""
    video_id: str
    video_url: str
    operation_id: str
    processing_options: Dict
    callback_url: Optional[str] = None

class VideoProcessingQueue:
    """Очередь для обработки видео"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        queue_name: str = "video_processing"
    ):
        self.redis = redis.from_url(redis_url)
        self.queue = Queue(queue_name, connection=self.redis)
        self.high_priority_queue = Queue(
            f"{queue_name}_high",
            connection=self.redis
        )
    
    def enqueue(
        self,
        job: VideoProcessingJob,
        priority: str = "normal"
    ) -> str:
        """
        Добавляет задачу в очередь
        
        Returns:
            job_id: ID созданной задачи
        """
        queue = (
            self.high_priority_queue 
            if priority == "high" 
            else self.queue
        )
        
        rq_job = queue.enqueue(
            process_video,
            args=(asdict(job),),
            job_timeout="2h",
            result_ttl=86400,  # 24 часа
            failure_ttl=86400
        )
        
        # Сохраняем статус
        self._set_job_status(rq_job.id, JobStatus.PENDING)
        
        return rq_job.id
    
    def get_job_status(self, job_id: str) -> Dict:
        """Получает статус задачи"""
        job = Job.fetch(job_id, connection=self.redis)
        
        return {
            "job_id": job_id,
            "status": job.get_status(),
            "progress": self._get_job_progress(job_id),
            "result": job.result if job.is_finished else None,
            "error": str(job.exc_info) if job.is_failed else None
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Отменяет задачу"""
        try:
            job = Job.fetch(job_id, connection=self.redis)
            job.cancel()
            return True
        except Exception:
            return False
    
    def get_queue_stats(self) -> Dict:
        """Статистика очереди"""
        return {
            "queued": len(self.queue),
            "high_priority_queued": len(self.high_priority_queue),
            "workers": Worker.count(connection=self.redis),
            "failed": self.queue.failed_job_registry.count
        }
    
    def _set_job_status(self, job_id: str, status: JobStatus):
        """Устанавливает статус задачи"""
        self.redis.hset(f"job:{job_id}", "status", status.value)
    
    def _get_job_progress(self, job_id: str) -> Optional[Dict]:
        """Получает прогресс задачи"""
        progress = self.redis.hget(f"job:{job_id}", "progress")
        return json.loads(progress) if progress else None
    
    def update_progress(
        self,
        job_id: str,
        stage: str,
        progress: float,
        message: str = ""
    ):
        """Обновляет прогресс задачи"""
        progress_data = {
            "stage": stage,
            "progress": progress,
            "message": message
        }
        self.redis.hset(
            f"job:{job_id}",
            "progress",
            json.dumps(progress_data)
        )


def process_video(job_data: Dict):
    """
    Worker функция для обработки видео
    
    Stages:
    1. Download (0-10%)
    2. Frame extraction (10-20%)
    3. Phase segmentation (20-40%)
    4. Instrument detection (40-60%)
    5. Anatomy segmentation (60-80%)
    6. Event detection (80-90%)
    7. Indexing (90-100%)
    """
    from rq import get_current_job
    
    job = get_current_job()
    queue = VideoProcessingQueue()
    
    try:
        # Stage 1: Download
        queue.update_progress(job.id, "download", 0, "Загрузка видео...")
        video_path = download_video(job_data["video_url"])
        queue.update_progress(job.id, "download", 10, "Видео загружено")
        
        # Stage 2: Frame extraction
        queue.update_progress(job.id, "frames", 10, "Извлечение кадров...")
        frames, timestamps = extract_frames(video_path)
        queue.update_progress(job.id, "frames", 20, f"Извлечено {len(frames)} кадров")
        
        # Stage 3: Phase segmentation
        queue.update_progress(job.id, "phases", 20, "Сегментация фаз...")
        phases = segment_phases(frames, timestamps)
        queue.update_progress(job.id, "phases", 40, f"Найдено {len(phases)} фаз")
        
        # Stage 4: Instrument detection
        queue.update_progress(job.id, "instruments", 40, "Детекция инструментов...")
        instruments = detect_instruments(frames, timestamps)
        queue.update_progress(job.id, "instruments", 60, f"Найдено {len(instruments)} инструментов")
        
        # Stage 5: Anatomy segmentation
        queue.update_progress(job.id, "anatomy", 60, "Сегментация анатомии...")
        anatomy = segment_anatomy(frames, timestamps)
        queue.update_progress(job.id, "anatomy", 80, f"Найдено {len(anatomy)} структур")
        
        # Stage 6: Event detection
        queue.update_progress(job.id, "events", 80, "Детекция событий...")
        events = detect_events(frames, timestamps, phases, instruments)
        queue.update_progress(job.id, "events", 90, f"Найдено {len(events)} событий")
        
        # Stage 7: Indexing
        queue.update_progress(job.id, "indexing", 90, "Индексация результатов...")
        index_results(job_data["operation_id"], phases, instruments, anatomy, events)
        queue.update_progress(job.id, "indexing", 100, "Готово!")
        
        # Callback
        if job_data.get("callback_url"):
            send_callback(job_data["callback_url"], {
                "status": "completed",
                "operation_id": job_data["operation_id"]
            })
        
        return {
            "status": "completed",
            "phases_count": len(phases),
            "instruments_count": len(instruments),
            "anatomy_count": len(anatomy),
            "events_count": len(events)
        }
        
    except Exception as e:
        queue.update_progress(job.id, "error", -1, str(e))
        raise
```

---

## 7. Monitoring & Observability

### Prometheus Metrics

```python
"""
ML Monitoring Metrics
Метрики для мониторинга ML системы
"""
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest
)
import time
from functools import wraps

# Registry
REGISTRY = CollectorRegistry()

# Inference metrics
INFERENCE_REQUESTS = Counter(
    'ml_inference_requests_total',
    'Total inference requests',
    ['model', 'status'],
    registry=REGISTRY
)

INFERENCE_LATENCY = Histogram(
    'ml_inference_latency_seconds',
    'Inference latency in seconds',
    ['model'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

INFERENCE_BATCH_SIZE = Summary(
    'ml_inference_batch_size',
    'Batch size for inference',
    ['model'],
    registry=REGISTRY
)

# Model performance metrics
MODEL_CONFIDENCE = Histogram(
    'ml_model_confidence',
    'Model prediction confidence',
    ['model', 'class'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    registry=REGISTRY
)

# Queue metrics
QUEUE_SIZE = Gauge(
    'ml_queue_size',
    'Number of jobs in queue',
    ['queue_name'],
    registry=REGISTRY
)

QUEUE_PROCESSING_TIME = Histogram(
    'ml_queue_processing_seconds',
    'Video processing time',
    buckets=[60, 300, 600, 1800, 3600, 7200],
    registry=REGISTRY
)

# GPU metrics
GPU_UTILIZATION = Gauge(
    'ml_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id'],
    registry=REGISTRY
)

GPU_MEMORY_USED = Gauge(
    'ml_gpu_memory_used_bytes',
    'GPU memory used',
    ['gpu_id'],
    registry=REGISTRY
)

# Data drift metrics
DATA_DRIFT_SCORE = Gauge(
    'ml_data_drift_score',
    'Data drift score (0-1)',
    ['feature'],
    registry=REGISTRY
)


def track_inference(model_name: str):
    """Декоратор для трекинга метрик инференса"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                INFERENCE_REQUESTS.labels(
                    model=model_name, 
                    status='success'
                ).inc()
                return result
            except Exception as e:
                INFERENCE_REQUESTS.labels(
                    model=model_name, 
                    status='error'
                ).inc()
                raise
            finally:
                latency = time.time() - start_time
                INFERENCE_LATENCY.labels(model=model_name).observe(latency)
        
        return wrapper
    return decorator


class ModelMonitor:
    """Мониторинг производительности модели"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.predictions = []
        self.ground_truth = []
    
    def record_prediction(
        self,
        prediction: dict,
        confidence: float,
        class_name: str
    ):
        """Записывает предсказание"""
        MODEL_CONFIDENCE.labels(
            model=self.model_name,
            class_=class_name
        ).observe(confidence)
        
        self.predictions.append(prediction)
    
    def record_ground_truth(self, ground_truth: dict):
        """Записывает ground truth (для отложенной оценки)"""
        self.ground_truth.append(ground_truth)
    
    def calculate_drift(self, reference_data: list, current_data: list) -> float:
        """Вычисляет data drift"""
        from scipy import stats
        
        # KS test для числовых признаков
        stat, p_value = stats.ks_2samp(reference_data, current_data)
        
        # drift = 1 - p_value (чем выше, тем больше дрифт)
        return 1 - p_value
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "NeuroVision ML Monitoring",
    "panels": [
      {
        "title": "Inference Requests/sec",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_inference_requests_total[5m])",
            "legendFormat": "{{model}} - {{status}}"
          }
        ]
      },
      {
        "title": "Inference Latency (p95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_inference_latency_seconds_bucket[5m]))",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "title": "Model Confidence Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(ml_model_confidence_bucket[5m])",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "title": "Queue Size",
        "type": "gauge",
        "targets": [
          {
            "expr": "ml_queue_size",
            "legendFormat": "{{queue_name}}"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_gpu_utilization_percent",
            "legendFormat": "GPU {{gpu_id}}"
          }
        ]
      },
      {
        "title": "Data Drift Alert",
        "type": "stat",
        "targets": [
          {
            "expr": "max(ml_data_drift_score) > 0.3"
          }
        ],
        "thresholds": {
          "mode": "absolute",
          "steps": [
            {"color": "green", "value": null},
            {"color": "yellow", "value": 0.2},
            {"color": "red", "value": 0.3}
          ]
        }
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# prometheus_alerts.yml
groups:
  - name: ml_alerts
    rules:
      # High latency alert
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, rate(ml_inference_latency_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "95th percentile latency is {{ $value }}s for model {{ $labels.model }}"
      
      # High error rate
      - alert: HighErrorRate
        expr: rate(ml_inference_requests_total{status="error"}[5m]) / rate(ml_inference_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in inference"
          description: "Error rate is {{ $value | humanizePercentage }} for model {{ $labels.model }}"
      
      # Data drift detected
      - alert: DataDriftDetected
        expr: ml_data_drift_score > 0.3
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected"
          description: "Feature {{ $labels.feature }} has drift score {{ $value }}"
      
      # GPU memory pressure
      - alert: GPUMemoryPressure
        expr: ml_gpu_memory_used_bytes / (16 * 1024 * 1024 * 1024) > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage is high"
          description: "GPU {{ $labels.gpu_id }} memory usage is {{ $value | humanizePercentage }}"
      
      # Queue backlog
      - alert: QueueBacklog
        expr: ml_queue_size > 100
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Large queue backlog"
          description: "Queue {{ $labels.queue_name }} has {{ $value }} pending jobs"
```

---

## 8. Security

### Защита персональных данных (PII)

```python
"""
PII Protection Module
Модуль защиты персональных данных
"""
import hashlib
import re
from typing import Dict, Any
from cryptography.fernet import Fernet
import os

class PIIProtector:
    """Защита персональных данных пациентов"""
    
    def __init__(self, encryption_key: bytes = None):
        if encryption_key is None:
            encryption_key = os.environ.get('PII_ENCRYPTION_KEY')
            if encryption_key:
                encryption_key = encryption_key.encode()
            else:
                encryption_key = Fernet.generate_key()
        
        self.cipher = Fernet(encryption_key)
    
    def anonymize_patient_id(self, patient_id: str, salt: str = None) -> str:
        """
        Анонимизирует ID пациента
        
        Использует SHA-256 хеширование с солью
        """
        if salt is None:
            salt = os.environ.get('ANONYMIZATION_SALT', 'default_salt')
        
        salted = f"{patient_id}{salt}".encode()
        return hashlib.sha256(salted).hexdigest()[:16]
    
    def encrypt_metadata(self, metadata: Dict[str, Any]) -> bytes:
        """Шифрует метаданные пациента"""
        import json
        json_data = json.dumps(metadata).encode()
        return self.cipher.encrypt(json_data)
    
    def decrypt_metadata(self, encrypted: bytes) -> Dict[str, Any]:
        """Дешифрует метаданные"""
        import json
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())
    
    def redact_video_frame(
        self,
        frame,
        face_detector,
        blur_strength: int = 50
    ):
        """
        Размывает лица в кадре видео
        """
        import cv2
        
        # Детектируем лица
        faces = face_detector.detect(frame)
        
        # Размываем каждое лицо
        for (x, y, w, h) in faces:
            # Расширяем область
            x = max(0, x - 10)
            y = max(0, y - 10)
            w = min(frame.shape[1] - x, w + 20)
            h = min(frame.shape[0] - y, h + 20)
            
            # Применяем размытие
            roi = frame[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
            frame[y:y+h, x:x+w] = blurred
        
        return frame
    
    def remove_embedded_metadata(self, video_path: str, output_path: str):
        """Удаляет метаданные из видео файла"""
        import subprocess
        
        subprocess.run([
            'ffmpeg', '-i', video_path,
            '-map_metadata', '-1',
            '-c:v', 'copy',
            '-c:a', 'copy',
            output_path
        ], check=True)
    
    def sanitize_text(self, text: str) -> str:
        """
        Удаляет PII из текста (имена, даты рождения, ИНН и т.д.)
        """
        # Паттерны для PII
        patterns = {
            'phone': r'\+?[78]?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'passport': r'\d{2}\s?\d{2}\s?\d{6}',
            'snils': r'\d{3}[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{2}',
            'inn': r'\d{10,12}',
            'date_of_birth': r'\d{2}[./]\d{2}[./]\d{4}',
        }
        
        sanitized = text
        for pii_type, pattern in patterns.items():
            sanitized = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]', sanitized)
        
        return sanitized


class AuditLogger:
    """Логирование доступа к данным для аудита"""
    
    def __init__(self, log_path: str = "/var/log/neurovision/audit.log"):
        import logging
        
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        ip_address: str = None
    ):
        """Логирует доступ к ресурсу"""
        self.logger.info(
            f"ACCESS | user={user_id} | resource={resource_type}:{resource_id} | "
            f"action={action} | ip={ip_address}"
        )
    
    def log_export(
        self,
        user_id: str,
        data_type: str,
        record_count: int,
        purpose: str
    ):
        """Логирует экспорт данных"""
        self.logger.info(
            f"EXPORT | user={user_id} | type={data_type} | "
            f"records={record_count} | purpose={purpose}"
        )
```

### RBAC (Role-Based Access Control)

```python
"""
Role-Based Access Control
Ролевая модель доступа
"""
from enum import Enum
from typing import Set, Dict, Optional
from functools import wraps

class Role(Enum):
    ADMIN = "admin"
    SURGEON = "surgeon"
    RESIDENT = "resident"
    RESEARCHER = "researcher"
    TECHNICIAN = "technician"
    VIEWER = "viewer"

class Permission(Enum):
    # Video permissions
    VIDEO_VIEW = "video:view"
    VIDEO_UPLOAD = "video:upload"
    VIDEO_DELETE = "video:delete"
    VIDEO_EXPORT = "video:export"
    
    # Annotation permissions
    ANNOTATION_VIEW = "annotation:view"
    ANNOTATION_EDIT = "annotation:edit"
    ANNOTATION_DELETE = "annotation:delete"
    
    # Model permissions
    MODEL_INFERENCE = "model:inference"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    
    # Admin permissions
    USER_MANAGE = "user:manage"
    SYSTEM_CONFIG = "system:config"
    AUDIT_VIEW = "audit:view"

# Матрица разрешений
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # Все права
    
    Role.SURGEON: {
        Permission.VIDEO_VIEW,
        Permission.VIDEO_UPLOAD,
        Permission.VIDEO_EXPORT,
        Permission.ANNOTATION_VIEW,
        Permission.ANNOTATION_EDIT,
        Permission.MODEL_INFERENCE,
    },
    
    Role.RESIDENT: {
        Permission.VIDEO_VIEW,
        Permission.ANNOTATION_VIEW,
        Permission.MODEL_INFERENCE,
    },
    
    Role.RESEARCHER: {
        Permission.VIDEO_VIEW,
        Permission.VIDEO_EXPORT,
        Permission.ANNOTATION_VIEW,
        Permission.MODEL_INFERENCE,
        Permission.MODEL_TRAIN,
    },
    
    Role.TECHNICIAN: {
        Permission.VIDEO_VIEW,
        Permission.VIDEO_UPLOAD,
        Permission.VIDEO_DELETE,
        Permission.MODEL_INFERENCE,
        Permission.MODEL_DEPLOY,
    },
    
    Role.VIEWER: {
        Permission.VIDEO_VIEW,
        Permission.ANNOTATION_VIEW,
    },
}


class AccessControl:
    """Контроль доступа"""
    
    def __init__(self, user_service):
        self.user_service = user_service
    
    def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource_id: str = None
    ) -> bool:
        """Проверяет разрешение"""
        user = self.user_service.get_user(user_id)
        if not user:
            return False
        
        user_role = Role(user.role)
        allowed_permissions = ROLE_PERMISSIONS.get(user_role, set())
        
        if permission not in allowed_permissions:
            return False
        
        # Дополнительные проверки для конкретных ресурсов
        if resource_id and not self._check_resource_access(user, resource_id):
            return False
        
        return True
    
    def _check_resource_access(self, user, resource_id: str) -> bool:
        """Проверяет доступ к конкретному ресурсу"""
        # Например, хирург может видеть только свои операции
        # или операции своего отделения
        return True  # Simplified


def require_permission(permission: Permission):
    """Декоратор для проверки разрешений"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Получаем user_id из контекста
            from flask import g, abort
            
            if not hasattr(g, 'user_id'):
                abort(401)
            
            ac = AccessControl(g.user_service)
            if not ac.check_permission(g.user_id, permission):
                abort(403)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

## 9. CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main, develop]
    paths:
      - 'src/**'
      - 'configs/**'
      - 'tests/**'
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      
      - name: Lint with ruff
        run: ruff check src/
      
      - name: Type check with mypy
        run: mypy src/
      
      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build-training-image:
    needs: lint-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push training image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: Dockerfile.training
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/training:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  build-inference-image:
    needs: lint-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build and push inference image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: Dockerfile.inference
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/inference:${{ github.sha }}

  run-integration-tests:
    needs: [build-training-image, build-inference-image]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Start services
        run: docker-compose -f docker-compose.test.yml up -d
      
      - name: Wait for services
        run: sleep 30
      
      - name: Run integration tests
        run: |
          docker-compose -f docker-compose.test.yml exec -T api \
            pytest tests/integration/ -v
      
      - name: Cleanup
        run: docker-compose -f docker-compose.test.yml down -v

  deploy-staging:
    needs: run-integration-tests
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to staging
        run: |
          kubectl config use-context staging
          kubectl set image deployment/inference \
            inference=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/inference:${{ github.sha }}
          kubectl rollout status deployment/inference

  deploy-production:
    needs: run-integration-tests
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to production
        run: |
          kubectl config use-context production
          kubectl set image deployment/inference \
            inference=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/inference:${{ github.sha }}
          kubectl rollout status deployment/inference
      
      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          fields: repo,message,commit,author
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 10. Compliance (GDPR/HIPAA Principles)

### Checklist соответствия

```markdown
# GDPR/HIPAA Compliance Checklist

## Обработка данных
- [x] Минимизация данных - собираем только необходимое
- [x] Ограничение цели - данные используются только для заявленных целей
- [x] Ограничение хранения - автоматическое удаление после истечения срока
- [x] Анонимизация/псевдонимизация данных пациентов

## Права субъектов данных
- [x] Право на доступ - пациент может запросить свои данные
- [x] Право на удаление - возможность удаления данных по запросу
- [x] Право на переносимость - экспорт данных в стандартном формате
- [x] Право на ограничение обработки

## Безопасность
- [x] Шифрование данных в покое (AES-256)
- [x] Шифрование данных при передаче (TLS 1.3)
- [x] Контроль доступа (RBAC)
- [x] Аудит доступа к данным
- [x] Регулярное резервное копирование

## Организационные меры
- [ ] Назначен DPO (Data Protection Officer)
- [ ] Проведена DPIA (Data Protection Impact Assessment)
- [ ] Заключены соглашения об обработке данных с поставщиками
- [ ] Обучение персонала по защите данных
- [ ] План реагирования на инциденты

## Технические меры
- [x] Размытие лиц на видео
- [x] Удаление метаданных из файлов
- [x] Хеширование идентификаторов пациентов
- [x] Разделение идентифицирующих и медицинских данных
- [x] Автоматическое удаление устаревших данных
```

### Data Retention Policy

```python
"""
Data Retention Policy Implementation
Реализация политики хранения данных
"""
from datetime import datetime, timedelta
from typing import List
import logging

logger = logging.getLogger(__name__)

class DataRetentionPolicy:
    """Политика хранения данных"""
    
    # Сроки хранения по типам данных
    RETENTION_PERIODS = {
        'raw_video': timedelta(days=365),      # 1 год
        'processed_data': timedelta(days=730),  # 2 года
        'anonymized_data': timedelta(days=1825), # 5 лет
        'audit_logs': timedelta(days=2555),     # 7 лет
        'ml_models': timedelta(days=1825),      # 5 лет
        'temp_files': timedelta(days=7),        # 1 неделя
    }
    
    def __init__(self, storage_service, db_service):
        self.storage = storage_service
        self.db = db_service
    
    def enforce_retention(self, data_type: str) -> dict:
        """Применяет политику хранения для типа данных"""
        retention_period = self.RETENTION_PERIODS.get(data_type)
        if not retention_period:
            raise ValueError(f"Unknown data type: {data_type}")
        
        cutoff_date = datetime.utcnow() - retention_period
        
        # Находим данные для удаления
        items_to_delete = self.db.find_items(
            data_type=data_type,
            created_before=cutoff_date
        )
        
        deleted_count = 0
        errors = []
        
        for item in items_to_delete:
            try:
                # Удаляем из хранилища
                self.storage.delete(item['storage_path'])
                # Удаляем из БД
                self.db.delete_item(item['id'])
                deleted_count += 1
                
                logger.info(
                    f"Deleted {data_type} item {item['id']} "
                    f"(created: {item['created_at']})"
                )
            except Exception as e:
                errors.append({
                    'item_id': item['id'],
                    'error': str(e)
                })
                logger.error(f"Failed to delete {item['id']}: {e}")
        
        return {
            'data_type': data_type,
            'retention_period_days': retention_period.days,
            'cutoff_date': cutoff_date.isoformat(),
            'items_found': len(items_to_delete),
            'items_deleted': deleted_count,
            'errors': errors
        }
    
    def schedule_deletion(
        self,
        item_id: str,
        deletion_date: datetime,
        reason: str
    ):
        """Планирует удаление конкретного элемента"""
        self.db.create_deletion_schedule({
            'item_id': item_id,
            'scheduled_for': deletion_date,
            'reason': reason,
            'created_at': datetime.utcnow()
        })
    
    def process_deletion_requests(self) -> List[dict]:
        """Обрабатывает запланированные удаления"""
        scheduled = self.db.find_scheduled_deletions(
            due_before=datetime.utcnow()
        )
        
        results = []
        for schedule in scheduled:
            result = self._delete_item(schedule['item_id'])
            result['reason'] = schedule['reason']
            results.append(result)
            
            # Отмечаем как выполненное
            self.db.mark_deletion_complete(schedule['id'])
        
        return results
```

---

## Заключение

Данный документ описывает полный MLOps стек для платформы NeuroVision:

1. **Data Management** - DVC для версионирования данных и экспериментов
2. **Model Registry** - MLflow для трекинга и управления моделями
3. **Training Pipeline** - Kubeflow для оркестрации обучения
4. **Containerization** - Docker для изоляции и воспроизводимости
5. **Inference** - Triton для high-performance serving
6. **Queue System** - Redis/RQ для асинхронной обработки
7. **Monitoring** - Prometheus + Grafana для наблюдаемости
8. **Security** - Шифрование, RBAC, аудит
9. **CI/CD** - GitHub Actions для автоматизации
10. **Compliance** - GDPR/HIPAA принципы защиты данных
