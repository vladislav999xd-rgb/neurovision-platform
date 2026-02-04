# Архитектура NeuroVision Platform

## Общая схема системы

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              NeuroVision Platform                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                           FRONTEND (Browser)                                │ │
│  │  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │ │
│  │  │  Login   │ │ Dashboard │ │  Upload  │ │  Viewer  │ │ Archive/Search   │ │ │
│  │  └────┬─────┘ └─────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘ │ │
│  │       │             │            │            │                │           │ │
│  │       └─────────────┴────────────┴────────────┴────────────────┘           │ │
│  │                                   │                                         │ │
│  │                            REST API Calls                                   │ │
│  └───────────────────────────────────┼─────────────────────────────────────────┘ │
│                                      │                                           │
│  ┌───────────────────────────────────▼─────────────────────────────────────────┐ │
│  │                        API GATEWAY / Load Balancer                          │ │
│  └───────────────────────────────────┬─────────────────────────────────────────┘ │
│                                      │                                           │
│  ┌───────────────────────────────────▼─────────────────────────────────────────┐ │
│  │                           BACKEND SERVICES                                   │ │
│  │                                                                              │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────────┐   │ │
│  │  │   Auth Service  │  │  Video Service  │  │     Search Service         │   │ │
│  │  │                 │  │                 │  │                            │   │ │
│  │  │ • JWT tokens    │  │ • Upload        │  │ • Elasticsearch/OpenSearch │   │ │
│  │  │ • RBAC          │  │ • Streaming     │  │ • Combined queries         │   │ │
│  │  │ • Sessions      │  │ • Transcoding   │  │ • Faceted search           │   │ │
│  │  └────────┬────────┘  └────────┬────────┘  └─────────────┬──────────────┘   │ │
│  │           │                    │                         │                   │ │
│  └───────────┴────────────────────┴─────────────────────────┴───────────────────┘ │
│                                   │                                              │
│  ┌────────────────────────────────▼────────────────────────────────────────────┐ │
│  │                          MESSAGE QUEUE (RabbitMQ/Redis)                      │ │
│  └────────────────────────────────┬────────────────────────────────────────────┘ │
│                                   │                                              │
│  ┌────────────────────────────────▼────────────────────────────────────────────┐ │
│  │                         ML PROCESSING PIPELINE                               │ │
│  │                                                                              │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │ │
│  │  │   Ingestion  │  │    Scene     │  │   Object     │  │   Event          │ │ │
│  │  │   Worker     │──▶│  Segmenter   │──▶│  Detector    │──▶│   Detector       │ │ │
│  │  │              │  │              │  │              │  │                  │ │ │
│  │  │ • FFmpeg     │  │ • Phase det. │  │ • Instruments│  │ • Bleeding       │ │ │
│  │  │ • Frames     │  │ • Temporal   │  │ • Anatomy    │  │ • Complications  │ │ │
│  │  │ • Audio      │  │   modeling   │  │ • Tracking   │  │ • Anomalies      │ │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │ │
│  │         │                 │                 │                   │           │ │
│  │         └─────────────────┴─────────────────┴───────────────────┘           │ │
│  │                                   │                                         │ │
│  │                        ┌──────────▼──────────┐                              │ │
│  │                        │   Indexer Service   │                              │ │
│  │                        │                     │                              │ │
│  │                        │ • Aggregate results │                              │ │
│  │                        │ • Build search idx  │                              │ │
│  │                        │ • Store metadata    │                              │ │
│  │                        └──────────┬──────────┘                              │ │
│  │                                   │                                         │ │
│  └───────────────────────────────────┼─────────────────────────────────────────┘ │
│                                      │                                           │
│  ┌───────────────────────────────────▼─────────────────────────────────────────┐ │
│  │                            DATA LAYER                                        │ │
│  │                                                                              │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────────┐ │ │
│  │  │   PostgreSQL   │  │  S3 / MinIO    │  │      Elasticsearch             │ │ │
│  │  │                │  │                │  │                                │ │ │
│  │  │ • Operations   │  │ • Raw videos   │  │ • Full-text search             │ │ │
│  │  │ • Segments     │  │ • Thumbnails   │  │ • Semantic search              │ │ │
│  │  │ • Detections   │  │ • Frames       │  │ • Aggregations                 │ │ │
│  │  │ • Users        │  │ • Models       │  │ • Analytics                    │ │ │
│  │  └────────────────┘  └────────────────┘  └────────────────────────────────┘ │ │
│  │                                                                              │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## Компоненты системы

### 1. Frontend Layer

| Компонент | Описание | Технологии |
|-----------|----------|------------|
| Login | Аутентификация пользователей | HTML/JS, JWT |
| Dashboard | Список операций, статистика | HTML/JS, REST API |
| Upload | Загрузка и метаданные видео | Drag&Drop, FormData |
| Viewer | Видеоплеер, таймлайн, фильтры | Custom player, Canvas |
| Archive | Глобальный поиск, фильтры | Elasticsearch queries |

### 2. Backend Services

| Сервис | Ответственность | Порт |
|--------|-----------------|------|
| API Gateway | Роутинг, rate limiting, auth | 3000 |
| Auth Service | JWT, RBAC, сессии | 3001 |
| Video Service | Upload, streaming, transcoding | 3002 |
| Search Service | Индексация, поиск | 3003 |
| ML Orchestrator | Управление ML пайплайном | 3004 |

### 3. ML Processing Workers

| Worker | Функция | GPU |
|--------|---------|-----|
| Ingestion | FFmpeg, извлечение кадров | Нет |
| Scene Segmenter | Определение этапов | Да |
| Object Detector | Инструменты, анатомия | Да |
| Event Detector | Кровотечения, осложнения | Да |
| Speech-to-Text | Субтитры | Да |
| Indexer | Агрегация и индексация | Нет |

### 4. Data Storage

| Хранилище | Данные | Backup |
|-----------|--------|--------|
| PostgreSQL | Метаданные, связи | Ежедневно |
| S3/MinIO | Видео, изображения | Репликация |
| Elasticsearch | Поисковый индекс | Snapshots |
| Redis | Кэш, очереди | AOF |

---

## API Endpoints

### Operations API

```
GET    /api/operations              # Список операций
GET    /api/operations/:id          # Детали операции
POST   /api/operations              # Создать операцию
PUT    /api/operations/:id          # Обновить операцию
DELETE /api/operations/:id          # Удалить операцию

GET    /api/operations/:id/segments   # Сегменты операции
GET    /api/operations/:id/detections # Детекции
GET    /api/operations/:id/subtitles  # Субтитры
```

### Search API

```
POST   /api/search                  # Комбинированный поиск
GET    /api/search/suggest          # Автодополнение
GET    /api/search/facets           # Фасеты для фильтров
```

### Tags API

```
GET    /api/tags                    # Все справочники
GET    /api/tags/:type              # Справочник по типу
POST   /api/tags/:type              # Добавить тег
```

### Video API

```
POST   /api/videos/upload           # Загрузка видео
GET    /api/videos/:id/stream       # Стриминг
GET    /api/videos/:id/thumbnail    # Превью
POST   /api/videos/:id/process      # Запуск обработки
```

---

## JSON Schemas

### Operation

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["id", "title", "surgeon", "date", "type", "status"],
  "properties": {
    "id": { "type": "string", "format": "uuid" },
    "title": { "type": "string", "maxLength": 500 },
    "patientId": { "type": "string" },
    "surgeon": { "type": "string" },
    "date": { "type": "string", "format": "date-time" },
    "type": { "type": "string" },
    "duration": { "type": "integer", "minimum": 0 },
    "videoUrl": { "type": "string", "format": "uri" },
    "thumbnailUrl": { "type": "string", "format": "uri" },
    "status": { 
      "type": "string", 
      "enum": ["pending", "processing", "completed", "failed"] 
    },
    "description": { "type": "string" },
    "segments": { "type": "array", "items": { "$ref": "#/definitions/Segment" } },
    "detections": { "type": "array", "items": { "$ref": "#/definitions/Detection" } },
    "subtitles": { "type": "array", "items": { "$ref": "#/definitions/Subtitle" } }
  }
}
```

### Segment

```json
{
  "type": "object",
  "required": ["id", "type", "name", "startTime", "endTime", "confidence"],
  "properties": {
    "id": { "type": "string", "format": "uuid" },
    "operationId": { "type": "string", "format": "uuid" },
    "type": { "type": "string", "enum": ["phase", "event"] },
    "name": { "type": "string" },
    "startTime": { "type": "number", "minimum": 0 },
    "endTime": { "type": "number", "minimum": 0 },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
    "thumbnailUrl": { "type": "string" },
    "tags": { "type": "array", "items": { "type": "string" } },
    "metadata": { "type": "object" }
  }
}
```

### Detection

```json
{
  "type": "object",
  "required": ["id", "type", "name", "timestamp", "confidence"],
  "properties": {
    "id": { "type": "string", "format": "uuid" },
    "operationId": { "type": "string", "format": "uuid" },
    "type": { "type": "string", "enum": ["instrument", "anatomy", "event"] },
    "name": { "type": "string" },
    "timestamp": { "type": "number", "minimum": 0 },
    "duration": { "type": "number", "minimum": 0 },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
    "bbox": {
      "type": "object",
      "properties": {
        "x": { "type": "number" },
        "y": { "type": "number" },
        "width": { "type": "number" },
        "height": { "type": "number" }
      }
    },
    "mask": { "type": "object" },
    "timeRanges": { 
      "type": "array", 
      "items": { 
        "type": "array", 
        "items": { "type": "number" },
        "minItems": 2,
        "maxItems": 2
      } 
    }
  }
}
```

---

## Миграция с MVP на Production

### Этап 1: Инфраструктура (2-4 недели)
- [ ] Развертывание Kubernetes кластера
- [ ] Настройка PostgreSQL (managed или self-hosted)
- [ ] Развертывание MinIO/S3 для хранения видео
- [ ] Установка Elasticsearch кластера
- [ ] Настройка RabbitMQ/Redis для очередей

### Этап 2: Backend Services (4-6 недель)
- [ ] Декомпозиция монолита на микросервисы
- [ ] Реализация Auth Service с JWT
- [ ] Video Service с HLS стримингом
- [ ] Search Service с Elasticsearch
- [ ] API Gateway с rate limiting

### Этап 3: ML Pipeline (6-8 недель)
- [ ] Обучение моделей на реальных данных
- [ ] Развертывание GPU-воркеров
- [ ] Интеграция с MLflow для трекинга
- [ ] A/B тестирование моделей

### Этап 4: Продакшн (2-4 недели)
- [ ] Нагрузочное тестирование
- [ ] Настройка мониторинга (Prometheus/Grafana)
- [ ] CI/CD пайплайны
- [ ] Документация и обучение
