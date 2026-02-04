# CV Pipeline - Компьютерное зрение для анализа нейрохирургических операций

## Обзор пайплайна

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           CV PROCESSING PIPELINE                                   │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│   │  Video   │    │  Frame   │    │  Scene   │    │  Object  │    │  Event   │   │
│   │  Ingest  │───▶│ Extract  │───▶│ Segment  │───▶│ Detect   │───▶│ Detect   │   │
│   │          │    │          │    │          │    │          │    │          │   │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│        │              │               │               │               │          │
│        ▼              ▼               ▼               ▼               ▼          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│   │ Metadata │    │  Frames  │    │ Segments │    │ Bounding │    │  Events  │   │
│   │  + Audio │    │   + KF   │    │ + Phases │    │  Boxes   │    │   List   │   │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│                                                                                    │
│                                        │                                          │
│                                        ▼                                          │
│                              ┌──────────────────┐                                 │
│                              │    Instance      │                                 │
│                              │  Segmentation    │                                 │
│                              │   (Anatomy)      │                                 │
│                              └────────┬─────────┘                                 │
│                                       │                                           │
│                                       ▼                                           │
│                              ┌──────────────────┐                                 │
│                              │    Indexer &     │                                 │
│                              │   Aggregator     │                                 │
│                              └────────┬─────────┘                                 │
│                                       │                                           │
│                                       ▼                                           │
│                              ┌──────────────────┐                                 │
│                              │   Searchable     │                                 │
│                              │     Index        │                                 │
│                              └──────────────────┘                                 │
│                                                                                    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Video Ingestion Module

### Назначение
Прием и первичная обработка видео файлов хирургических операций.

### Входные данные
| Параметр | Формат | Описание |
|----------|--------|----------|
| video_file | MP4, MOV, MKV, AVI | Исходное видео операции |
| metadata | JSON | Информация об операции |
| surgeon_id | UUID | Идентификатор хирурга |
| patient_id | UUID (анонимизирован) | Идентификатор пациента |

### Выходные данные
| Параметр | Формат | Описание |
|----------|--------|----------|
| video_id | UUID | Уникальный ID видео |
| video_path | S3 URI | Путь к сохраненному видео |
| audio_path | S3 URI | Извлеченная аудиодорожка |
| duration | Float (seconds) | Длительность видео |
| resolution | Tuple (w, h) | Разрешение |
| fps | Float | Частота кадров |
| codec | String | Кодек видео |

### Реализация

```python
"""
Video Ingestion Module
Модуль приема и первичной обработки видео
"""
import ffmpeg
import boto3
from dataclasses import dataclass
from typing import Optional, Tuple
import hashlib

@dataclass
class VideoMetadata:
    video_id: str
    duration: float
    resolution: Tuple[int, int]
    fps: float
    codec: str
    file_hash: str
    
class VideoIngestionService:
    def __init__(self, s3_client, bucket_name: str):
        self.s3 = s3_client
        self.bucket = bucket_name
    
    def ingest(self, video_path: str, metadata: dict) -> VideoMetadata:
        """
        Принимает видео и сохраняет в хранилище
        
        Args:
            video_path: Путь к локальному видео файлу
            metadata: Метаданные операции
            
        Returns:
            VideoMetadata: Информация о загруженном видео
        """
        # Извлекаем метаданные через FFprobe
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            s for s in probe['streams'] if s['codec_type'] == 'video'
        )
        
        # Вычисляем хеш для дедупликации
        file_hash = self._compute_hash(video_path)
        video_id = self._generate_video_id(file_hash)
        
        # Загружаем в S3
        s3_key = f"videos/{video_id}/original.mp4"
        self.s3.upload_file(video_path, self.bucket, s3_key)
        
        # Извлекаем аудио
        audio_key = f"videos/{video_id}/audio.wav"
        self._extract_audio(video_path, f"/tmp/{video_id}_audio.wav")
        self.s3.upload_file(f"/tmp/{video_id}_audio.wav", self.bucket, audio_key)
        
        return VideoMetadata(
            video_id=video_id,
            duration=float(probe['format']['duration']),
            resolution=(
                int(video_stream['width']),
                int(video_stream['height'])
            ),
            fps=eval(video_stream['r_frame_rate']),
            codec=video_stream['codec_name'],
            file_hash=file_hash
        )
    
    def _extract_audio(self, video_path: str, output_path: str):
        """Извлекает аудиодорожку из видео"""
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='pcm_s16le', ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )
    
    def _compute_hash(self, file_path: str) -> str:
        """Вычисляет SHA256 хеш файла"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _generate_video_id(self, file_hash: str) -> str:
        """Генерирует уникальный ID видео"""
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, file_hash))
```

---

## 2. Frame Extraction Module

### Назначение
Извлечение кадров из видео с оптимальной частотой для ML-обработки.

### Входные данные
| Параметр | Формат | Описание |
|----------|--------|----------|
| video_path | S3 URI | Путь к видео |
| sample_rate | Float | Частота сэмплирования (кадров/сек) |
| keyframe_only | Boolean | Только ключевые кадры |

### Выходные данные
| Параметр | Формат | Описание |
|----------|--------|----------|
| frames | List[np.ndarray] | Массив кадров |
| timestamps | List[float] | Временные метки кадров |
| keyframes | List[int] | Индексы ключевых кадров |

### Реализация

```python
"""
Frame Extraction Module
Модуль извлечения кадров из видео
"""
import cv2
import numpy as np
from typing import List, Tuple, Generator
from dataclasses import dataclass

@dataclass
class FrameBatch:
    frames: np.ndarray  # Shape: (N, H, W, 3)
    timestamps: List[float]
    frame_indices: List[int]
    is_keyframe: List[bool]

class FrameExtractor:
    def __init__(
        self, 
        target_fps: float = 2.0,  # 2 кадра в секунду по умолчанию
        resize_to: Tuple[int, int] = (640, 480),
        batch_size: int = 32
    ):
        self.target_fps = target_fps
        self.resize_to = resize_to
        self.batch_size = batch_size
    
    def extract_frames(
        self, 
        video_path: str
    ) -> Generator[FrameBatch, None, None]:
        """
        Генератор батчей кадров из видео
        
        Args:
            video_path: Путь к видео файлу
            
        Yields:
            FrameBatch: Батч кадров с метаданными
        """
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / self.target_fps)
        
        frames_buffer = []
        timestamps_buffer = []
        indices_buffer = []
        keyframe_buffer = []
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Resize кадра
                frame_resized = cv2.resize(frame, self.resize_to)
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                timestamp = frame_idx / original_fps
                is_keyframe = self._is_keyframe(frame_idx, cap)
                
                frames_buffer.append(frame_rgb)
                timestamps_buffer.append(timestamp)
                indices_buffer.append(frame_idx)
                keyframe_buffer.append(is_keyframe)
                
                if len(frames_buffer) >= self.batch_size:
                    yield FrameBatch(
                        frames=np.array(frames_buffer),
                        timestamps=timestamps_buffer.copy(),
                        frame_indices=indices_buffer.copy(),
                        is_keyframe=keyframe_buffer.copy()
                    )
                    frames_buffer.clear()
                    timestamps_buffer.clear()
                    indices_buffer.clear()
                    keyframe_buffer.clear()
            
            frame_idx += 1
        
        cap.release()
        
        # Возвращаем оставшиеся кадры
        if frames_buffer:
            yield FrameBatch(
                frames=np.array(frames_buffer),
                timestamps=timestamps_buffer,
                frame_indices=indices_buffer,
                is_keyframe=keyframe_buffer
            )
    
    def _is_keyframe(self, frame_idx: int, cap) -> bool:
        """Определяет, является ли кадр ключевым (I-frame)"""
        # Упрощенная логика - реальная реализация требует
        # анализа GOP структуры
        return frame_idx % 30 == 0
    
    def extract_keyframes_only(self, video_path: str) -> FrameBatch:
        """Извлекает только ключевые кадры (I-frames)"""
        # Использует FFmpeg для быстрого извлечения I-frames
        import subprocess
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Извлекаем только I-frames через FFmpeg
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', 'select=eq(pict_type\\,I)',
                '-vsync', 'vfr',
                '-frame_pts', '1',
                f'{tmpdir}/frame_%06d.jpg'
            ]
            subprocess.run(cmd, capture_output=True)
            
            frames = []
            timestamps = []
            for fname in sorted(os.listdir(tmpdir)):
                frame = cv2.imread(os.path.join(tmpdir, fname))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.resize_to)
                frames.append(frame)
                # Извлекаем timestamp из имени файла
                # (требует дополнительной обработки)
            
            return FrameBatch(
                frames=np.array(frames),
                timestamps=timestamps,
                frame_indices=list(range(len(frames))),
                is_keyframe=[True] * len(frames)
            )
```

---

## 3. Scene Segmentation Module

### Назначение
Определение этапов/фаз хирургической операции на основе визуального анализа.

### Этапы нейрохирургической операции

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ФАЗЫ НЕЙРОХИРУРГИЧЕСКОЙ ОПЕРАЦИИ                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. ПОДГОТОВКА          2. ДОСТУП               3. ОСНОВНОЙ ЭТАП            │
│  ┌───────────────┐      ┌───────────────┐       ┌───────────────┐           │
│  │ • Укладка     │      │ • Разрез кожи │       │ • Работа с    │           │
│  │ • Обработка   │─────▶│ • Трепанация  │──────▶│   патологией  │           │
│  │ • Разметка    │      │ • Вскрытие ТМО│       │ • Резекция    │           │
│  └───────────────┘      └───────────────┘       └───────────────┘           │
│                                                         │                    │
│                                                         ▼                    │
│  6. ЗАВЕРШЕНИЕ          5. ГЕМОСТАЗ             4. КОНТРОЛЬ                 │
│  ┌───────────────┐      ┌───────────────┐       ┌───────────────┐           │
│  │ • Ушивание    │      │ • Остановка   │       │ • Визуальный  │           │
│  │ • Повязка     │◀─────│   кровотечения│◀──────│   осмотр      │           │
│  │ • Документация│      │ • Коагуляция  │       │ • Эндоскопия  │           │
│  └───────────────┘      └───────────────┘       └───────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Модель: Temporal Action Segmentation

**Архитектура:** MS-TCN++ (Multi-Stage Temporal Convolutional Network)

```
Input: Frame Features (T x D)
         │
         ▼
┌─────────────────────────┐
│   Feature Encoder       │  <- ResNet50 / I3D pretrained
│   (Spatial Features)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Temporal Conv Layer 1  │  <- Dilated convolutions
│  (Capture local context)│
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Temporal Conv Layer 2  │  <- Multi-scale temporal
│  (Medium-range context) │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Temporal Conv Layer 3  │
│  (Long-range context)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    Refinement Stage     │  <- Smooth predictions
│    (MS-TCN++)           │
└───────────┬─────────────┘
            │
            ▼
Output: Phase predictions (T x num_phases)
```

### Реализация

```python
"""
Scene Segmentation Module
Модуль сегментации видео на фазы операции
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import numpy as np

class DilatedResidualLayer(nn.Module):
    """Слой с dilated convolution для temporal modeling"""
    
    def __init__(self, dilation: int, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3,
            padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out

class MSTCN(nn.Module):
    """Multi-Stage Temporal Convolutional Network для сегментации фаз"""
    
    def __init__(
        self,
        num_stages: int = 4,
        num_layers: int = 10,
        num_features: int = 64,
        num_classes: int = 10,  # Количество фаз операции
        input_dim: int = 2048   # Размерность feature вектора
    ):
        super().__init__()
        self.num_stages = num_stages
        
        # Первая стадия
        self.stage1 = SingleStageModel(
            num_layers, num_features, num_classes, input_dim
        )
        
        # Стадии уточнения
        self.stages = nn.ModuleList([
            SingleStageModel(num_layers, num_features, num_classes, num_classes)
            for _ in range(num_stages - 1)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Input features (B, input_dim, T)
            
        Returns:
            outputs: List of predictions from each stage
        """
        outputs = []
        
        out = self.stage1(x)
        outputs.append(out)
        
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)
        
        return outputs

class SingleStageModel(nn.Module):
    """Одна стадия MS-TCN"""
    
    def __init__(
        self, 
        num_layers: int,
        num_features: int,
        num_classes: int,
        input_dim: int
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(input_dim, num_features, 1)
        
        self.layers = nn.ModuleList([
            DilatedResidualLayer(2 ** i, num_features, num_features)
            for i in range(num_layers)
        ])
        
        self.conv_out = nn.Conv1d(num_features, num_classes, 1)
    
    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)

class SurgicalPhaseSegmenter:
    """Сервис сегментации хирургических фаз"""
    
    PHASES = [
        "preparation",           # Подготовка
        "patient_positioning",   # Укладка пациента
        "draping",              # Обкладывание
        "incision",             # Разрез
        "craniotomy",           # Трепанация
        "dura_opening",         # Вскрытие ТМО
        "tumor_resection",      # Резекция опухоли
        "hemostasis",           # Гемостаз
        "dura_closure",         # Пластика ТМО
        "bone_flap_replacement", # Установка костного лоскута
        "wound_closure"         # Ушивание раны
    ]
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.feature_extractor = self._load_feature_extractor()
    
    def _load_model(self, model_path: str) -> MSTCN:
        model = MSTCN(
            num_classes=len(self.PHASES),
            input_dim=2048
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def _load_feature_extractor(self):
        """Загружает ResNet50 для извлечения признаков"""
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True)
        # Убираем последний FC слой
        modules = list(resnet.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        feature_extractor.to(self.device)
        feature_extractor.eval()
        return feature_extractor
    
    def segment(
        self, 
        frames: np.ndarray,
        timestamps: List[float]
    ) -> List[Dict]:
        """
        Сегментирует видео на фазы
        
        Args:
            frames: Массив кадров (N, H, W, C)
            timestamps: Временные метки кадров
            
        Returns:
            segments: Список сегментов с фазами
        """
        # Извлекаем признаки
        features = self._extract_features(frames)
        
        # Предсказание фаз
        with torch.no_grad():
            # (1, D, T)
            features_tensor = torch.tensor(features).permute(1, 0).unsqueeze(0)
            features_tensor = features_tensor.to(self.device).float()
            
            outputs = self.model(features_tensor)
            predictions = outputs[-1]  # Берем последнюю стадию
            
            # (T, num_classes) -> (T,)
            phase_indices = predictions.argmax(dim=1).squeeze().cpu().numpy()
            confidences = F.softmax(predictions, dim=1).max(dim=1)[0]
            confidences = confidences.squeeze().cpu().numpy()
        
        # Конвертируем в сегменты
        segments = self._predictions_to_segments(
            phase_indices, confidences, timestamps
        )
        
        return segments
    
    def _extract_features(self, frames: np.ndarray) -> np.ndarray:
        """Извлекает feature векторы из кадров"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        features = []
        batch_size = 32
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            batch_tensor = torch.stack([transform(f) for f in batch])
            batch_tensor = batch_tensor.to(self.device)
            
            with torch.no_grad():
                feat = self.feature_extractor(batch_tensor)
                feat = feat.squeeze(-1).squeeze(-1)  # (B, 2048)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def _predictions_to_segments(
        self,
        phase_indices: np.ndarray,
        confidences: np.ndarray,
        timestamps: List[float]
    ) -> List[Dict]:
        """Конвертирует покадровые предсказания в сегменты"""
        segments = []
        current_phase = phase_indices[0]
        segment_start = 0
        segment_confidences = [confidences[0]]
        
        for i, (phase_idx, conf) in enumerate(zip(phase_indices[1:], confidences[1:]), 1):
            if phase_idx != current_phase:
                # Завершаем текущий сегмент
                segments.append({
                    'type': 'phase',
                    'name': self.PHASES[current_phase],
                    'name_ru': self._get_phase_name_ru(current_phase),
                    'start_time': timestamps[segment_start],
                    'end_time': timestamps[i - 1],
                    'confidence': float(np.mean(segment_confidences)),
                    'phase_index': int(current_phase)
                })
                
                # Начинаем новый сегмент
                current_phase = phase_idx
                segment_start = i
                segment_confidences = [conf]
            else:
                segment_confidences.append(conf)
        
        # Добавляем последний сегмент
        segments.append({
            'type': 'phase',
            'name': self.PHASES[current_phase],
            'name_ru': self._get_phase_name_ru(current_phase),
            'start_time': timestamps[segment_start],
            'end_time': timestamps[-1],
            'confidence': float(np.mean(segment_confidences)),
            'phase_index': int(current_phase)
        })
        
        return segments
    
    def _get_phase_name_ru(self, phase_index: int) -> str:
        """Возвращает русское название фазы"""
        names_ru = [
            "Подготовка",
            "Укладка пациента",
            "Обкладывание",
            "Разрез",
            "Трепанация",
            "Вскрытие твёрдой мозговой оболочки",
            "Резекция опухоли",
            "Гемостаз",
            "Пластика ТМО",
            "Установка костного лоскута",
            "Ушивание раны"
        ]
        return names_ru[phase_index] if phase_index < len(names_ru) else "Неизвестно"
```

---

## 4. Object Detection Module

### Назначение
Обнаружение хирургических инструментов в кадрах видео.

### Классы инструментов

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ХИРУРГИЧЕСКИЕ ИНСТРУМЕНТЫ                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  РЕЖУЩИЕ              ЗАХВАТЫВАЮЩИЕ          ОТСАСЫВАЮЩИЕ                   │
│  ┌───────────────┐    ┌───────────────┐      ┌───────────────┐              │
│  │ • Скальпель   │    │ • Пинцет      │      │ • Аспиратор   │              │
│  │ • Ножницы     │    │ • Зажим       │      │ • CUSA        │              │
│  │ • Кусачки     │    │ • Диссектор   │      │               │              │
│  └───────────────┘    └───────────────┘      └───────────────┘              │
│                                                                              │
│  КОАГУЛЯЦИОННЫЕ       РЕТРАКТОРЫ             СПЕЦИАЛЬНЫЕ                    │
│  ┌───────────────┐    ┌───────────────┐      ┌───────────────┐              │
│  │ • Биполярный  │    │ • Шпатель     │      │ • Эндоскоп    │              │
│  │   коагулятор  │    │ • Ретрактор   │      │ • Микроскоп   │              │
│  │ • Монополярный│    │   мозга       │      │ • Навигация   │              │
│  └───────────────┘    └───────────────┘      └───────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Модель: YOLOv8 для нейрохирургических инструментов

```
Input Image (640x640)
        │
        ▼
┌─────────────────────┐
│     Backbone        │
│   (CSPDarknet53)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│       Neck          │
│   (PANet + SPPF)    │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌───────┐   ┌───────┐
│ P3/8  │   │ P4/16 │   <- Multi-scale detection
└───┬───┘   └───┬───┘
    │           │
    ▼           ▼
┌─────────────────────┐
│   Detection Heads   │
│  (class + box + conf)│
└─────────────────────┘
          │
          ▼
   Detected Instruments
   with bounding boxes
```

### Реализация

```python
"""
Surgical Instrument Detection Module
Модуль детекции хирургических инструментов
"""
from ultralytics import YOLO
import torch
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Detection:
    """Результат детекции одного объекта"""
    class_id: int
    class_name: str
    class_name_ru: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    timestamp: float
    frame_index: int

class SurgicalInstrumentDetector:
    """Детектор хирургических инструментов на базе YOLOv8"""
    
    INSTRUMENT_CLASSES = {
        0: ("scalpel", "Скальпель"),
        1: ("scissors", "Ножницы"),
        2: ("forceps", "Пинцет"),
        3: ("bipolar", "Биполярный коагулятор"),
        4: ("monopolar", "Монополярный коагулятор"),
        5: ("suction", "Аспиратор"),
        6: ("retractor", "Ретрактор"),
        7: ("dissector", "Диссектор"),
        8: ("drill", "Краниотом"),
        9: ("cusa", "CUSA"),
        10: ("clip_applier", "Клипсонакладыватель"),
        11: ("needle_holder", "Иглодержатель"),
        12: ("spatula", "Шпатель"),
        13: ("endoscope", "Эндоскоп"),
        14: ("microscope", "Микроскоп")
    }
    
    def __init__(
        self, 
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'cuda'
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
    
    def detect(
        self,
        frames: np.ndarray,
        timestamps: List[float]
    ) -> List[Detection]:
        """
        Детектирует инструменты в кадрах
        
        Args:
            frames: Массив кадров (N, H, W, C)
            timestamps: Временные метки кадров
            
        Returns:
            detections: Список детекций
        """
        all_detections = []
        
        # Batch inference
        results = self.model.predict(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        for frame_idx, result in enumerate(results):
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    bbox = box.xyxyn[0].tolist()  # Normalized coords
                    
                    class_info = self.INSTRUMENT_CLASSES.get(
                        class_id, 
                        ("unknown", "Неизвестно")
                    )
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_info[0],
                        class_name_ru=class_info[1],
                        confidence=confidence,
                        bbox=tuple(bbox),
                        timestamp=timestamps[frame_idx],
                        frame_index=frame_idx
                    )
                    all_detections.append(detection)
        
        return all_detections
    
    def detect_with_tracking(
        self,
        frames: np.ndarray,
        timestamps: List[float]
    ) -> Dict[int, List[Detection]]:
        """
        Детекция с трекингом объектов между кадрами
        
        Returns:
            tracks: Словарь {track_id: [detections]}
        """
        tracks = {}
        
        # Используем встроенный трекер YOLOv8
        results = self.model.track(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            persist=True,
            verbose=False
        )
        
        for frame_idx, result in enumerate(results):
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Track ID может быть None если трекинг не сработал
                    track_id = int(box.id[0].item()) if box.id is not None else -1
                    class_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    bbox = box.xyxyn[0].tolist()
                    
                    class_info = self.INSTRUMENT_CLASSES.get(
                        class_id,
                        ("unknown", "Неизвестно")
                    )
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_info[0],
                        class_name_ru=class_info[1],
                        confidence=confidence,
                        bbox=tuple(bbox),
                        timestamp=timestamps[frame_idx],
                        frame_index=frame_idx
                    )
                    
                    if track_id not in tracks:
                        tracks[track_id] = []
                    tracks[track_id].append(detection)
        
        return tracks
    
    def aggregate_detections(
        self,
        detections: List[Detection]
    ) -> List[Dict]:
        """
        Агрегирует детекции по временным интервалам использования
        
        Returns:
            aggregated: Список инструментов с временными интервалами
        """
        from collections import defaultdict
        
        # Группируем по классу
        by_class = defaultdict(list)
        for det in detections:
            by_class[det.class_id].append(det)
        
        aggregated = []
        
        for class_id, class_detections in by_class.items():
            # Сортируем по времени
            class_detections.sort(key=lambda x: x.timestamp)
            
            # Объединяем близкие детекции в интервалы
            intervals = []
            current_interval = None
            gap_threshold = 2.0  # секунды
            
            for det in class_detections:
                if current_interval is None:
                    current_interval = {
                        'start': det.timestamp,
                        'end': det.timestamp,
                        'confidences': [det.confidence]
                    }
                elif det.timestamp - current_interval['end'] <= gap_threshold:
                    current_interval['end'] = det.timestamp
                    current_interval['confidences'].append(det.confidence)
                else:
                    intervals.append(current_interval)
                    current_interval = {
                        'start': det.timestamp,
                        'end': det.timestamp,
                        'confidences': [det.confidence]
                    }
            
            if current_interval:
                intervals.append(current_interval)
            
            class_info = self.INSTRUMENT_CLASSES.get(
                class_id,
                ("unknown", "Неизвестно")
            )
            
            aggregated.append({
                'type': 'instrument',
                'class_id': class_id,
                'name': class_info[0],
                'name_ru': class_info[1],
                'time_ranges': [
                    [interval['start'], interval['end']]
                    for interval in intervals
                ],
                'total_duration': sum(
                    interval['end'] - interval['start']
                    for interval in intervals
                ),
                'avg_confidence': float(np.mean([
                    c for interval in intervals
                    for c in interval['confidences']
                ]))
            })
        
        return aggregated
```

---

## 5. Instance Segmentation Module (Anatomy)

### Назначение
Попиксельная сегментация анатомических структур мозга.

### Анатомические структуры

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    АНАТОМИЧЕСКИЕ СТРУКТУРЫ МОЗГА                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ОБОЛОЧКИ                 СОСУДЫ                  ТКАНИ                     │
│  ┌───────────────┐        ┌───────────────┐       ┌───────────────┐         │
│  │ • Твёрдая     │        │ • Артерии     │       │ • Белое       │         │
│  │   мозговая    │        │ • Вены        │       │   вещество    │         │
│  │   оболочка    │        │ • Синусы      │       │ • Серое       │         │
│  │ • Мягкая      │        │               │       │   вещество    │         │
│  │   оболочка    │        │               │       │ • Опухоль     │         │
│  └───────────────┘        └───────────────┘       └───────────────┘         │
│                                                                              │
│  СТРУКТУРЫ                ЖЕЛУДОЧКИ               НЕРВЫ                     │
│  ┌───────────────┐        ┌───────────────┐       ┌───────────────┐         │
│  │ • Мозолистое  │        │ • Боковой     │       │ • Зрительный  │         │
│  │   тело        │        │   желудочек   │       │ • Обонятельн. │         │
│  │ • Гипофиз     │        │ • III         │       │ • Лицевой     │         │
│  │ • Гипоталамус │        │   желудочек   │       │               │         │
│  └───────────────┘        └───────────────┘       └───────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Модель: Mask R-CNN / Segment Anything (SAM)

### Реализация

```python
"""
Anatomical Structure Segmentation Module
Модуль сегментации анатомических структур
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2

@dataclass
class SegmentationResult:
    """Результат сегментации анатомической структуры"""
    class_id: int
    class_name: str
    class_name_ru: str
    mask: np.ndarray  # Binary mask (H, W)
    confidence: float
    bbox: Tuple[float, float, float, float]
    area_pixels: int
    centroid: Tuple[float, float]
    timestamp: float
    frame_index: int

class AnatomySegmenter:
    """Сегментатор анатомических структур на базе SAM + Classification head"""
    
    ANATOMY_CLASSES = {
        0: ("dura_mater", "Твёрдая мозговая оболочка"),
        1: ("pia_mater", "Мягкая мозговая оболочка"),
        2: ("cerebral_cortex", "Кора головного мозга"),
        3: ("white_matter", "Белое вещество"),
        4: ("artery", "Артерия"),
        5: ("vein", "Вена"),
        6: ("venous_sinus", "Венозный синус"),
        7: ("tumor", "Опухоль"),
        8: ("cyst", "Киста"),
        9: ("hematoma", "Гематома"),
        10: ("ventricle", "Желудочек"),
        11: ("corpus_callosum", "Мозолистое тело"),
        12: ("brainstem", "Ствол мозга"),
        13: ("cerebellum", "Мозжечок"),
        14: ("pituitary", "Гипофиз"),
        15: ("optic_nerve", "Зрительный нерв"),
        16: ("carotid_artery", "Внутренняя сонная артерия"),
        17: ("aneurysm", "Аневризма")
    }
    
    def __init__(
        self,
        sam_checkpoint: str,
        classifier_checkpoint: str,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sam = self._load_sam(sam_checkpoint)
        self.classifier = self._load_classifier(classifier_checkpoint)
    
    def _load_sam(self, checkpoint: str):
        """Загружает Segment Anything Model"""
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        
        model_type = "vit_h"  # или vit_l, vit_b
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(self.device)
        
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100
        )
        
        return mask_generator
    
    def _load_classifier(self, checkpoint: str):
        """Загружает классификатор анатомических структур"""
        import torchvision.models as models
        
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(self.ANATOMY_CLASSES))
        model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
    def segment(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_index: int
    ) -> List[SegmentationResult]:
        """
        Сегментирует анатомические структуры в кадре
        
        Args:
            frame: Кадр (H, W, C) в RGB
            timestamp: Временная метка
            frame_index: Индекс кадра
            
        Returns:
            results: Список сегментированных структур
        """
        # Генерируем маски с помощью SAM
        masks = self.sam.generate(frame)
        
        results = []
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # (x, y, w, h)
            
            # Вырезаем ROI для классификации
            x, y, w, h = [int(v) for v in bbox]
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
            
            # Классифицируем
            class_id, confidence = self._classify_roi(roi, mask[y:y+h, x:x+w])
            
            if confidence < 0.5:  # Порог уверенности
                continue
            
            class_info = self.ANATOMY_CLASSES.get(
                class_id,
                ("unknown", "Неизвестно")
            )
            
            # Вычисляем центроид
            moments = cv2.moments(mask.astype(np.uint8))
            if moments["m00"] > 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            else:
                cx, cy = x + w/2, y + h/2
            
            # Нормализуем bbox
            h_img, w_img = frame.shape[:2]
            bbox_normalized = (
                x / w_img,
                y / h_img,
                (x + w) / w_img,
                (y + h) / h_img
            )
            
            result = SegmentationResult(
                class_id=class_id,
                class_name=class_info[0],
                class_name_ru=class_info[1],
                mask=mask,
                confidence=confidence,
                bbox=bbox_normalized,
                area_pixels=int(mask.sum()),
                centroid=(cx / w_img, cy / h_img),
                timestamp=timestamp,
                frame_index=frame_index
            )
            results.append(result)
        
        return results
    
    def _classify_roi(
        self,
        roi: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[int, float]:
        """Классифицирует ROI как анатомическую структуру"""
        from torchvision import transforms
        
        # Применяем маску
        roi_masked = roi.copy()
        roi_masked[~mask] = 0
        
        # Предобработка
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        roi_tensor = transform(roi_masked).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(roi_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, class_id = probs.max(dim=1)
        
        return int(class_id.item()), float(confidence.item())
    
    def aggregate_anatomy(
        self,
        segmentations: List[SegmentationResult]
    ) -> List[Dict]:
        """Агрегирует сегментации по временным интервалам"""
        from collections import defaultdict
        
        by_class = defaultdict(list)
        for seg in segmentations:
            by_class[seg.class_id].append(seg)
        
        aggregated = []
        
        for class_id, class_segs in by_class.items():
            class_segs.sort(key=lambda x: x.timestamp)
            
            # Объединяем в интервалы
            intervals = []
            current_interval = None
            gap_threshold = 3.0
            
            for seg in class_segs:
                if current_interval is None:
                    current_interval = {
                        'start': seg.timestamp,
                        'end': seg.timestamp,
                        'confidences': [seg.confidence]
                    }
                elif seg.timestamp - current_interval['end'] <= gap_threshold:
                    current_interval['end'] = seg.timestamp
                    current_interval['confidences'].append(seg.confidence)
                else:
                    intervals.append(current_interval)
                    current_interval = {
                        'start': seg.timestamp,
                        'end': seg.timestamp,
                        'confidences': [seg.confidence]
                    }
            
            if current_interval:
                intervals.append(current_interval)
            
            class_info = self.ANATOMY_CLASSES.get(
                class_id,
                ("unknown", "Неизвестно")
            )
            
            aggregated.append({
                'type': 'anatomy',
                'class_id': class_id,
                'name': class_info[0],
                'name_ru': class_info[1],
                'time_ranges': [
                    [interval['start'], interval['end']]
                    for interval in intervals
                ],
                'avg_confidence': float(np.mean([
                    c for interval in intervals
                    for c in interval['confidences']
                ]))
            })
        
        return aggregated
```

---

## 6. Event Detection Module

### Назначение
Обнаружение критических событий и осложнений во время операции.

### Типы событий

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ХИРУРГИЧЕСКИЕ СОБЫТИЯ                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  КРИТИЧЕСКИЕ            НОРМАЛЬНЫЕ              ТЕХНИЧЕСКИЕ                 │
│  ┌───────────────┐      ┌───────────────┐       ┌───────────────┐           │
│  │ ⚠ Кровотечение│      │ • Коагуляция  │       │ • Смена       │           │
│  │ ⚠ Перфорация  │      │ • Ирригация   │       │   инструмента │           │
│  │ ⚠ Повреждение │      │ • Разрез      │       │ • Пауза       │           │
│  │   нерва       │      │ • Удаление    │       │ • Zoom        │           │
│  │ ⚠ Гипоксия    │      │   ткани       │       │   микроскопа  │           │
│  └───────────────┘      └───────────────┘       └───────────────┘           │
│                                                                              │
│  МАРКЕРЫ                АНАТОМИЧЕСКИЕ                                       │
│  ┌───────────────┐      ┌───────────────┐                                   │
│  │ • Начало      │      │ • Идентифик.  │                                   │
│  │   резекции    │      │   структуры   │                                   │
│  │ • Клипирование│      │ • Границы     │                                   │
│  │ • Установка   │      │   резекции    │                                   │
│  │   импланта    │      │               │                                   │
│  └───────────────┘      └───────────────┘                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Реализация

```python
"""
Surgical Event Detection Module
Модуль детекции хирургических событий
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class EventSeverity(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class SurgicalEvent:
    """Хирургическое событие"""
    event_type: str
    event_name_ru: str
    severity: EventSeverity
    start_time: float
    end_time: float
    confidence: float
    description: str
    requires_attention: bool

class EventDetector:
    """Детектор хирургических событий"""
    
    EVENT_TYPES = {
        # Критические события
        "bleeding": ("Кровотечение", EventSeverity.CRITICAL),
        "vessel_injury": ("Повреждение сосуда", EventSeverity.CRITICAL),
        "nerve_injury": ("Повреждение нерва", EventSeverity.CRITICAL),
        "perforation": ("Перфорация", EventSeverity.CRITICAL),
        "unplanned_opening": ("Непреднамеренное вскрытие", EventSeverity.CRITICAL),
        
        # Предупреждения
        "minor_bleeding": ("Незначительное кровотечение", EventSeverity.WARNING),
        "adhesion": ("Спаечный процесс", EventSeverity.WARNING),
        "difficult_dissection": ("Сложная диссекция", EventSeverity.WARNING),
        
        # Нормальные события
        "coagulation": ("Коагуляция", EventSeverity.NORMAL),
        "irrigation": ("Ирригация", EventSeverity.NORMAL),
        "suction": ("Аспирация", EventSeverity.NORMAL),
        "clip_application": ("Клипирование", EventSeverity.NORMAL),
        "tissue_removal": ("Удаление ткани", EventSeverity.NORMAL),
        "instrument_change": ("Смена инструмента", EventSeverity.NORMAL),
        "microscope_focus": ("Фокусировка микроскопа", EventSeverity.NORMAL),
        "pause": ("Пауза", EventSeverity.NORMAL)
    }
    
    def __init__(
        self,
        video_model_path: str,
        audio_model_path: str,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.video_model = self._load_video_model(video_model_path)
        self.audio_model = self._load_audio_model(audio_model_path)
    
    def _load_video_model(self, path: str):
        """
        Загружает модель для анализа видео
        Используем TimeSformer или Video Swin Transformer
        """
        # Placeholder - в продакшне используем pretrained модель
        from transformers import VideoMAEForVideoClassification
        
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base",
            num_labels=len(self.EVENT_TYPES)
        )
        
        if path:
            # Загружаем fine-tuned веса
            model.load_state_dict(
                torch.load(path, map_location=self.device),
                strict=False
            )
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_audio_model(self, path: str):
        """
        Загружает модель для анализа аудио
        Детектирует звуки: коагуляция, аспиратор, CUSA и т.д.
        """
        # Placeholder для аудио модели
        pass
    
    def detect_events(
        self,
        frames: np.ndarray,
        timestamps: List[float],
        audio: np.ndarray = None,
        detections: List[Dict] = None,
        phases: List[Dict] = None
    ) -> List[SurgicalEvent]:
        """
        Детектирует события в видео
        
        Args:
            frames: Массив кадров (N, H, W, C)
            timestamps: Временные метки
            audio: Аудио данные (опционально)
            detections: Результаты детекции инструментов
            phases: Текущие фазы операции
            
        Returns:
            events: Список обнаруженных событий
        """
        events = []
        
        # 1. Детекция событий по видео
        video_events = self._detect_video_events(frames, timestamps)
        events.extend(video_events)
        
        # 2. Детекция по аудио (если доступно)
        if audio is not None:
            audio_events = self._detect_audio_events(audio, timestamps)
            events.extend(audio_events)
        
        # 3. Rule-based детекция на основе детекций
        if detections:
            rule_events = self._detect_rule_based_events(
                detections, timestamps
            )
            events.extend(rule_events)
        
        # 4. Объединяем дублирующиеся события
        events = self._merge_events(events)
        
        # 5. Фильтруем по контексту фазы
        if phases:
            events = self._filter_by_phase_context(events, phases)
        
        return events
    
    def _detect_video_events(
        self,
        frames: np.ndarray,
        timestamps: List[float]
    ) -> List[SurgicalEvent]:
        """Детекция событий с помощью видео модели"""
        from transformers import VideoMAEImageProcessor
        
        processor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base"
        )
        
        events = []
        window_size = 16  # Количество кадров для анализа
        stride = 8  # Шаг скольжения
        
        for i in range(0, len(frames) - window_size, stride):
            window_frames = frames[i:i + window_size]
            window_timestamps = timestamps[i:i + window_size]
            
            # Предобработка
            inputs = processor(
                list(window_frames),
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.video_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                
                # Находим события с confidence > threshold
                for event_idx, prob in enumerate(probs):
                    if prob > 0.5:
                        event_type = list(self.EVENT_TYPES.keys())[event_idx]
                        event_info = self.EVENT_TYPES[event_type]
                        
                        events.append(SurgicalEvent(
                            event_type=event_type,
                            event_name_ru=event_info[0],
                            severity=event_info[1],
                            start_time=window_timestamps[0],
                            end_time=window_timestamps[-1],
                            confidence=float(prob),
                            description=f"Обнаружено: {event_info[0]}",
                            requires_attention=event_info[1] != EventSeverity.NORMAL
                        ))
        
        return events
    
    def _detect_audio_events(
        self,
        audio: np.ndarray,
        timestamps: List[float]
    ) -> List[SurgicalEvent]:
        """Детекция событий по аудио"""
        # Placeholder - анализ звуков хирургического оборудования
        events = []
        # Реализация аудио анализа
        return events
    
    def _detect_rule_based_events(
        self,
        detections: List[Dict],
        timestamps: List[float]
    ) -> List[SurgicalEvent]:
        """
        Rule-based детекция на основе паттернов детекций
        
        Правила:
        - Появление крови без коагулятора -> кровотечение
        - Быстрые движения инструментов -> сложная диссекция
        - Многократная смена инструментов -> технические сложности
        """
        events = []
        
        # Пример: детекция кровотечения
        for det in detections:
            if det.get('name') == 'blood' and det.get('area_ratio', 0) > 0.1:
                # Проверяем наличие коагулятора рядом
                has_coagulator = any(
                    d.get('name') in ['bipolar', 'monopolar']
                    and abs(d.get('timestamp', 0) - det.get('timestamp', 0)) < 2
                    for d in detections
                )
                
                if not has_coagulator:
                    events.append(SurgicalEvent(
                        event_type="bleeding",
                        event_name_ru="Кровотечение",
                        severity=EventSeverity.CRITICAL,
                        start_time=det.get('timestamp', 0),
                        end_time=det.get('timestamp', 0) + 2,
                        confidence=0.8,
                        description="Обнаружена кровь без активной коагуляции",
                        requires_attention=True
                    ))
        
        return events
    
    def _merge_events(
        self,
        events: List[SurgicalEvent]
    ) -> List[SurgicalEvent]:
        """Объединяет дублирующиеся/перекрывающиеся события"""
        if not events:
            return []
        
        # Сортируем по времени и типу
        events.sort(key=lambda x: (x.event_type, x.start_time))
        
        merged = []
        current = events[0]
        
        for event in events[1:]:
            if (event.event_type == current.event_type and 
                event.start_time <= current.end_time + 1):
                # Объединяем
                current = SurgicalEvent(
                    event_type=current.event_type,
                    event_name_ru=current.event_name_ru,
                    severity=max(current.severity, event.severity, 
                                key=lambda x: x.value),
                    start_time=current.start_time,
                    end_time=max(current.end_time, event.end_time),
                    confidence=max(current.confidence, event.confidence),
                    description=current.description,
                    requires_attention=current.requires_attention or event.requires_attention
                )
            else:
                merged.append(current)
                current = event
        
        merged.append(current)
        return merged
    
    def _filter_by_phase_context(
        self,
        events: List[SurgicalEvent],
        phases: List[Dict]
    ) -> List[SurgicalEvent]:
        """Фильтрует события по контексту фазы"""
        # Например, "кровотечение" менее критично во время гемостаза
        filtered = []
        
        for event in events:
            current_phase = self._get_phase_at_time(
                event.start_time, phases
            )
            
            if current_phase == 'hemostasis' and event.event_type == 'bleeding':
                # Понижаем severity во время гемостаза
                event = SurgicalEvent(
                    **{**event.__dict__, 'severity': EventSeverity.WARNING}
                )
            
            filtered.append(event)
        
        return filtered
    
    def _get_phase_at_time(
        self,
        timestamp: float,
        phases: List[Dict]
    ) -> str:
        """Возвращает фазу операции в указанный момент времени"""
        for phase in phases:
            if phase['start_time'] <= timestamp <= phase['end_time']:
                return phase['name']
        return 'unknown'
```

---

## 7. Метрики качества моделей

### 7.1 Scene Segmentation

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| **Frame-wise Accuracy** | Доля правильно классифицированных кадров | > 85% |
| **Segment-wise F1** | F1-score на уровне сегментов | > 0.80 |
| **Edit Distance** | Расстояние Левенштейна между предсказанием и GT | < 15% |
| **Temporal IoU** | Пересечение по времени | > 0.75 |

```python
def calculate_temporal_iou(pred_segments, gt_segments):
    """
    Вычисляет Temporal IoU между предсказанными и GT сегментами
    """
    total_iou = 0
    count = 0
    
    for gt in gt_segments:
        best_iou = 0
        for pred in pred_segments:
            if pred['name'] == gt['name']:
                # Вычисляем IoU по времени
                intersection_start = max(pred['start_time'], gt['start_time'])
                intersection_end = min(pred['end_time'], gt['end_time'])
                intersection = max(0, intersection_end - intersection_start)
                
                union_start = min(pred['start_time'], gt['start_time'])
                union_end = max(pred['end_time'], gt['end_time'])
                union = union_end - union_start
                
                iou = intersection / union if union > 0 else 0
                best_iou = max(best_iou, iou)
        
        total_iou += best_iou
        count += 1
    
    return total_iou / count if count > 0 else 0
```

### 7.2 Object Detection

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| **mAP@0.5** | Mean Average Precision при IoU=0.5 | > 0.75 |
| **mAP@0.5:0.95** | mAP при разных порогах IoU | > 0.55 |
| **Recall** | Полнота детекции | > 0.85 |
| **FPS** | Скорость инференса | > 30 fps |

### 7.3 Instance Segmentation

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| **Mask mAP** | mAP для масок | > 0.65 |
| **Boundary F1** | F1 на границах масок | > 0.70 |
| **Pixel Accuracy** | Точность по пикселям | > 90% |
| **Dice Score** | Коэффициент Дайса | > 0.80 |

### 7.4 Event Detection

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| **Event F1** | F1-score для событий | > 0.75 |
| **Temporal Overlap** | Перекрытие по времени | > 0.70 |
| **False Alarm Rate** | Ложные срабатывания | < 5% |
| **Critical Event Recall** | Полнота критических событий | > 95% |

---

## 8. Форматы данных

### 8.1 Входной формат видео

```yaml
supported_formats:
  video:
    - MP4 (H.264, H.265)
    - MOV
    - MKV
    - AVI
  resolution:
    min: 720p
    recommended: 1080p
    max: 4K
  framerate:
    min: 24 fps
    recommended: 30 fps
  bitrate:
    min: 5 Mbps
    recommended: 15-25 Mbps
```

### 8.2 Выходной формат детекций

```json
{
  "video_id": "uuid",
  "processed_at": "2024-01-15T10:30:00Z",
  "duration": 7200.0,
  "fps": 30.0,
  
  "phases": [
    {
      "id": "phase-001",
      "name": "preparation",
      "name_ru": "Подготовка",
      "start_time": 0.0,
      "end_time": 300.0,
      "confidence": 0.95
    }
  ],
  
  "instruments": [
    {
      "id": "inst-001",
      "name": "bipolar",
      "name_ru": "Биполярный коагулятор",
      "time_ranges": [[120.5, 180.0], [250.0, 320.5]],
      "total_duration": 130.0,
      "avg_confidence": 0.88
    }
  ],
  
  "anatomy": [
    {
      "id": "anat-001",
      "name": "dura_mater",
      "name_ru": "Твёрдая мозговая оболочка",
      "time_ranges": [[300.0, 600.0]],
      "avg_confidence": 0.82
    }
  ],
  
  "events": [
    {
      "id": "event-001",
      "type": "bleeding",
      "name_ru": "Кровотечение",
      "severity": "critical",
      "start_time": 450.0,
      "end_time": 480.0,
      "confidence": 0.91,
      "requires_attention": true
    }
  ]
}
```

---

## 9. Требования к оборудованию

### Inference Server

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| GPU | NVIDIA RTX 3080 (10GB) | NVIDIA A100 (40GB) |
| CPU | 8 cores | 32 cores |
| RAM | 32 GB | 128 GB |
| Storage | SSD 500GB | NVMe 2TB |
| Network | 1 Gbps | 10 Gbps |

### Throughput

| Режим | Время обработки | Параллельность |
|-------|-----------------|----------------|
| Real-time | 1x (real-time) | 1 видео |
| Batch | 0.3x длительности | 4 видео |
| High-quality | 0.5x длительности | 2 видео |
