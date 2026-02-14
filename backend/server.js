/**
 * NeuroVision Platform - Backend Server
 * Интеллектуальная видеоплатформа для анализа нейрохирургических операций
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../frontend')));

// Multer для загрузки файлов
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, path.join(__dirname, 'uploads'));
    },
    filename: (req, file, cb) => {
        cb(null, `${uuidv4()}-${file.originalname}`);
    }
});
const upload = multer({ storage });

// ==================== IN-MEMORY DATA STORE ====================

// Загрузка демо-данных
const fs = require('fs');

let operations = [];
let tags = { instruments: [], anatomy: [], events: [], phases: [] };
let annotations = {}; // { operationId: [annotation, ...] }

// Загрузка данных при старте
try {
    const operationsData = fs.readFileSync(path.join(__dirname, 'data/operations.json'), 'utf8');
    operations = JSON.parse(operationsData);
    console.log(`✓ Загружено ${operations.length} операций`);
} catch (e) {
    console.log('⚠ Файл operations.json не найден, используются пустые данные');
}

try {
    const tagsData = fs.readFileSync(path.join(__dirname, 'data/tags.json'), 'utf8');
    tags = JSON.parse(tagsData);
    console.log(`✓ Загружены справочники тегов`);
} catch (e) {
    console.log('⚠ Файл tags.json не найден');
}

// Загрузка аннотаций
try {
    const annotationsData = fs.readFileSync(path.join(__dirname, 'data/annotations.json'), 'utf8');
    annotations = JSON.parse(annotationsData);
    const totalAnnotations = Object.values(annotations).reduce((sum, arr) => sum + arr.length, 0);
    console.log(`✓ Загружено ${totalAnnotations} аннотаций`);
} catch (e) {
    console.log('⚠ Файл annotations.json не найден, создаётся пустой');
    annotations = {};
}

function saveAnnotations() {
    try {
        fs.writeFileSync(
            path.join(__dirname, 'data/annotations.json'),
            JSON.stringify(annotations, null, 4),
            'utf8'
        );
    } catch (e) {
        console.error('Ошибка сохранения аннотаций:', e);
    }
}

// ==================== JSON SCHEMAS (для документации) ====================

/**
 * @typedef {Object} Operation
 * @property {string} id - UUID операции
 * @property {string} title - Название операции
 * @property {string} patientId - ID пациента (анонимизированный)
 * @property {string} surgeon - Имя хирурга
 * @property {string} date - Дата операции (ISO 8601)
 * @property {string} type - Тип операции
 * @property {number} duration - Длительность в секундах
 * @property {string} videoUrl - URL видео
 * @property {string} thumbnailUrl - URL превью
 * @property {string} status - Статус обработки: pending|processing|completed|failed
 * @property {Segment[]} segments - Массив сегментов
 * @property {Detection[]} detections - Массив детекций
 * @property {Subtitle[]} subtitles - Субтитры
 */

/**
 * @typedef {Object} Segment
 * @property {string} id - UUID сегмента
 * @property {string} operationId - ID операции
 * @property {string} type - Тип: phase|event
 * @property {string} name - Название этапа/события
 * @property {number} startTime - Начало в секундах
 * @property {number} endTime - Конец в секундах
 * @property {number} confidence - Уверенность модели (0-1)
 * @property {string} thumbnailUrl - Ключевой кадр
 * @property {string[]} tags - Связанные теги
 * @property {Object} metadata - Дополнительные данные
 */

/**
 * @typedef {Object} Detection
 * @property {string} id - UUID детекции
 * @property {string} operationId - ID операции
 * @property {string} type - Тип: instrument|anatomy|event
 * @property {string} name - Название объекта
 * @property {number} timestamp - Время в секундах
 * @property {number} duration - Длительность присутствия
 * @property {number} confidence - Уверенность модели
 * @property {Object} bbox - Bounding box {x, y, width, height}
 * @property {Object} mask - Маска сегментации (если есть)
 * @property {number[]} timeRanges - Массив диапазонов времени присутствия
 */

// ==================== API ROUTES ====================

// Главная страница - отдаём frontend
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// ---------- OPERATIONS ----------

/**
 * GET /api/operations
 * Получить список всех операций с пагинацией и фильтрами
 */
app.get('/api/operations', (req, res) => {
    const { page = 1, limit = 20, surgeon, type, status, dateFrom, dateTo } = req.query;
    
    let filtered = [...operations];
    
    // Фильтрация
    if (surgeon) {
        filtered = filtered.filter(op => op.surgeon.toLowerCase().includes(surgeon.toLowerCase()));
    }
    if (type) {
        filtered = filtered.filter(op => op.type === type);
    }
    if (status) {
        filtered = filtered.filter(op => op.status === status);
    }
    if (dateFrom) {
        filtered = filtered.filter(op => new Date(op.date) >= new Date(dateFrom));
    }
    if (dateTo) {
        filtered = filtered.filter(op => new Date(op.date) <= new Date(dateTo));
    }
    
    // Пагинация
    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + parseInt(limit);
    const paginated = filtered.slice(startIndex, endIndex);
    
    res.json({
        success: true,
        data: paginated,
        pagination: {
            page: parseInt(page),
            limit: parseInt(limit),
            total: filtered.length,
            totalPages: Math.ceil(filtered.length / limit)
        }
    });
});

/**
 * GET /api/operations/:id
 * Получить детали операции по ID
 */
app.get('/api/operations/:id', (req, res) => {
    const operation = operations.find(op => op.id === req.params.id);
    
    if (!operation) {
        return res.status(404).json({
            success: false,
            error: 'Операция не найдена'
        });
    }
    
    res.json({
        success: true,
        data: operation
    });
});

/**
 * POST /api/operations/:id/process
 * Обработать операцию AI - сохранить результаты анализа
 */
app.post('/api/operations/:id/process', (req, res) => {
    const operationIndex = operations.findIndex(op => op.id === req.params.id);
    
    if (operationIndex === -1) {
        return res.status(404).json({
            success: false,
            error: 'Операция не найдена'
        });
    }
    
    const { status, segments, detections, subtitles } = req.body;
    
    // Обновляем операцию
    operations[operationIndex] = {
        ...operations[operationIndex],
        status: status || 'completed',
        segments: segments || operations[operationIndex].segments,
        detections: detections || operations[operationIndex].detections,
        subtitles: subtitles || operations[operationIndex].subtitles,
        processedAt: new Date().toISOString()
    };
    
    // Сохраняем в файл
    try {
        fs.writeFileSync(
            path.join(__dirname, 'data/operations.json'),
            JSON.stringify(operations, null, 4),
            'utf8'
        );
        console.log(`✓ Операция ${req.params.id} обработана и сохранена`);
    } catch (e) {
        console.error('Ошибка сохранения:', e);
    }
    
    res.json({
        success: true,
        data: operations[operationIndex],
        message: 'AI обработка завершена успешно'
    });
});

/**
 * GET /api/operations/:id/segments
 * Получить сегменты операции с фильтрами
 */
app.get('/api/operations/:id/segments', (req, res) => {
    const operation = operations.find(op => op.id === req.params.id);
    
    if (!operation) {
        return res.status(404).json({
            success: false,
            error: 'Операция не найдена'
        });
    }
    
    const { type, minConfidence = 0 } = req.query;
    let segments = operation.segments || [];
    
    if (type) {
        segments = segments.filter(seg => seg.type === type);
    }
    segments = segments.filter(seg => seg.confidence >= parseFloat(minConfidence));
    
    res.json({
        success: true,
        data: segments
    });
});

/**
 * GET /api/operations/:id/detections
 * Получить детекции (инструменты, анатомия, события)
 */
app.get('/api/operations/:id/detections', (req, res) => {
    const operation = operations.find(op => op.id === req.params.id);
    
    if (!operation) {
        return res.status(404).json({
            success: false,
            error: 'Операция не найдена'
        });
    }
    
    const { type, name, minConfidence = 0 } = req.query;
    let detections = operation.detections || [];
    
    if (type) {
        detections = detections.filter(det => det.type === type);
    }
    if (name) {
        detections = detections.filter(det => det.name.toLowerCase().includes(name.toLowerCase()));
    }
    detections = detections.filter(det => det.confidence >= parseFloat(minConfidence));
    
    res.json({
        success: true,
        data: detections
    });
});

/**
 * POST /api/videos/upload
 * Загрузить новое видео операции
 */
app.post('/api/videos/upload', upload.single('video'), (req, res) => {
    const { title, surgeon, date, type, patientId } = req.body;
    
    const newOperation = {
        id: uuidv4(),
        title: title || 'Новая операция',
        patientId: patientId || `P-${Date.now()}`,
        surgeon: surgeon || 'Не указан',
        date: date || new Date().toISOString(),
        type: type || 'Не указан',
        duration: 0,
        videoUrl: req.file ? `/uploads/${req.file.filename}` : null,
        thumbnailUrl: '/assets/placeholder-thumb.jpg',
        status: 'pending', // pending → processing → completed
        segments: [],
        detections: [],
        subtitles: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
    };
    
    operations.push(newOperation);
    
    // Симуляция начала обработки AI
    setTimeout(() => {
        const op = operations.find(o => o.id === newOperation.id);
        if (op) {
            op.status = 'processing';
            // В реальной системе здесь запускается ML пайплайн
        }
    }, 1000);
    
    res.status(201).json({
        success: true,
        data: newOperation,
        message: 'Видео загружено и поставлено в очередь на обработку'
    });
});

// ---------- SEARCH ----------

/**
 * POST /api/search
 * Комбинированный поиск по всем операциям
 */
app.post('/api/search', (req, res) => {
    const {
        query,           // Текстовый запрос
        phases,          // Массив этапов
        instruments,     // Массив инструментов
        anatomy,         // Массив анатомических структур
        events,          // Массив событий
        surgeons,        // Массив хирургов
        operationTypes,  // Массив типов операций
        dateFrom,        // Дата начала
        dateTo,          // Дата конца
        minConfidence = 0.5,
        page = 1,
        limit = 50
    } = req.body;
    
    const results = [];
    
    for (const operation of operations) {
        // Фильтр по дате
        if (dateFrom && new Date(operation.date) < new Date(dateFrom)) continue;
        if (dateTo && new Date(operation.date) > new Date(dateTo)) continue;
        
        // Фильтр по хирургу
        if (surgeons && surgeons.length > 0) {
            if (!surgeons.some(s => operation.surgeon.toLowerCase().includes(s.toLowerCase()))) continue;
        }
        
        // Фильтр по типу операции
        if (operationTypes && operationTypes.length > 0) {
            if (!operationTypes.includes(operation.type)) continue;
        }
        
        // Поиск по сегментам и детекциям
        const matchingSegments = [];
        
        // Фильтр по этапам
        if (phases && phases.length > 0) {
            const phaseSegments = (operation.segments || []).filter(seg => 
                seg.type === 'phase' && 
                phases.some(p => seg.name.toLowerCase().includes(p.toLowerCase())) &&
                seg.confidence >= minConfidence
            );
            matchingSegments.push(...phaseSegments);
        }
        
        // Фильтр по инструментам
        if (instruments && instruments.length > 0) {
            const instrumentDetections = (operation.detections || []).filter(det =>
                det.type === 'instrument' &&
                instruments.some(i => det.name.toLowerCase().includes(i.toLowerCase())) &&
                det.confidence >= minConfidence
            );
            // Преобразуем детекции в сегменты для унифицированного результата
            instrumentDetections.forEach(det => {
                matchingSegments.push({
                    id: det.id,
                    operationId: operation.id,
                    type: 'instrument',
                    name: det.name,
                    startTime: det.timestamp,
                    endTime: det.timestamp + (det.duration || 1),
                    confidence: det.confidence,
                    thumbnailUrl: operation.thumbnailUrl
                });
            });
        }
        
        // Фильтр по анатомии
        if (anatomy && anatomy.length > 0) {
            const anatomyDetections = (operation.detections || []).filter(det =>
                det.type === 'anatomy' &&
                anatomy.some(a => det.name.toLowerCase().includes(a.toLowerCase())) &&
                det.confidence >= minConfidence
            );
            anatomyDetections.forEach(det => {
                matchingSegments.push({
                    id: det.id,
                    operationId: operation.id,
                    type: 'anatomy',
                    name: det.name,
                    startTime: det.timestamp,
                    endTime: det.timestamp + (det.duration || 1),
                    confidence: det.confidence,
                    thumbnailUrl: operation.thumbnailUrl
                });
            });
        }
        
        // Фильтр по событиям
        if (events && events.length > 0) {
            const eventSegments = (operation.segments || []).filter(seg =>
                seg.type === 'event' &&
                events.some(e => seg.name.toLowerCase().includes(e.toLowerCase())) &&
                seg.confidence >= minConfidence
            );
            matchingSegments.push(...eventSegments);
        }
        
        // Текстовый поиск
        if (query) {
            const queryLower = query.toLowerCase();
            const textMatches = [
                ...(operation.segments || []).filter(seg => 
                    seg.name.toLowerCase().includes(queryLower) &&
                    seg.confidence >= minConfidence
                ),
                ...(operation.detections || []).filter(det =>
                    det.name.toLowerCase().includes(queryLower) &&
                    det.confidence >= minConfidence
                ).map(det => ({
                    id: det.id,
                    operationId: operation.id,
                    type: det.type,
                    name: det.name,
                    startTime: det.timestamp,
                    endTime: det.timestamp + (det.duration || 1),
                    confidence: det.confidence,
                    thumbnailUrl: operation.thumbnailUrl
                }))
            ];
            matchingSegments.push(...textMatches);
        }
        
        // Если нет фильтров, показываем все сегменты операции
        if (!phases && !instruments && !anatomy && !events && !query) {
            matchingSegments.push(...(operation.segments || []));
        }
        
        // Удаляем дубликаты
        const uniqueSegments = matchingSegments.filter((seg, index, self) =>
            index === self.findIndex(s => s.id === seg.id)
        );
        
        if (uniqueSegments.length > 0) {
            results.push({
                operation: {
                    id: operation.id,
                    title: operation.title,
                    surgeon: operation.surgeon,
                    date: operation.date,
                    type: operation.type,
                    thumbnailUrl: operation.thumbnailUrl,
                    videoUrl: operation.videoUrl,
                    duration: operation.duration
                },
                segments: uniqueSegments.sort((a, b) => a.startTime - b.startTime)
            });
        }
    }
    
    // Пагинация результатов
    const totalResults = results.reduce((sum, r) => sum + r.segments.length, 0);
    const startIndex = (page - 1) * limit;
    const paginatedResults = results.slice(startIndex, startIndex + limit);
    
    res.json({
        success: true,
        data: paginatedResults,
        pagination: {
            page: parseInt(page),
            limit: parseInt(limit),
            totalOperations: results.length,
            totalSegments: totalResults
        }
    });
});

// ---------- TAGS (Справочники) ----------

/**
 * GET /api/tags
 * Получить все справочники тегов
 */
app.get('/api/tags', (req, res) => {
    res.json({
        success: true,
        data: tags
    });
});

/**
 * GET /api/tags/:type
 * Получить справочник по типу: instruments, anatomy, events, phases
 */
app.get('/api/tags/:type', (req, res) => {
    const { type } = req.params;
    
    if (!tags[type]) {
        return res.status(404).json({
            success: false,
            error: `Справочник "${type}" не найден`
        });
    }
    
    res.json({
        success: true,
        data: tags[type]
    });
});

// ---------- STATISTICS ----------

/**
 * GET /api/stats
 * Общая статистика платформы
 */
app.get('/api/stats', (req, res) => {
    const totalOperations = operations.length;
    const completedOperations = operations.filter(op => op.status === 'completed').length;
    const totalDuration = operations.reduce((sum, op) => sum + (op.duration || 0), 0);
    const totalSegments = operations.reduce((sum, op) => sum + (op.segments?.length || 0), 0);
    const totalDetections = operations.reduce((sum, op) => sum + (op.detections?.length || 0), 0);
    
    // Статистика по хирургам
    const surgeonStats = {};
    operations.forEach(op => {
        surgeonStats[op.surgeon] = (surgeonStats[op.surgeon] || 0) + 1;
    });
    
    // Статистика по типам операций
    const typeStats = {};
    operations.forEach(op => {
        typeStats[op.type] = (typeStats[op.type] || 0) + 1;
    });
    
    res.json({
        success: true,
        data: {
            totalOperations,
            completedOperations,
            totalDuration,
            totalSegments,
            totalDetections,
            surgeonStats,
            typeStats,
            lastUpdated: new Date().toISOString()
        }
    });
});

// ---------- SUBTITLES ----------

/**
 * GET /api/operations/:id/subtitles
 * Получить субтитры операции
 */
app.get('/api/operations/:id/subtitles', (req, res) => {
    const operation = operations.find(op => op.id === req.params.id);
    
    if (!operation) {
        return res.status(404).json({
            success: false,
            error: 'Операция не найдена'
        });
    }
    
    res.json({
        success: true,
        data: operation.subtitles || []
    });
});

// ==================== ANNOTATIONS (Аннотации видео) ====================

/**
 * GET /api/operations/:id/annotations
 * Получить все аннотации для операции
 */
app.get('/api/operations/:id/annotations', (req, res) => {
    const operation = operations.find(op => op.id === req.params.id);
    if (!operation) {
        return res.status(404).json({ success: false, error: 'Операция не найдена' });
    }
    
    const opAnnotations = annotations[req.params.id] || [];
    
    // Сортировка по времени
    const sorted = [...opAnnotations].sort((a, b) => a.timestamp - b.timestamp);
    
    res.json({
        success: true,
        data: sorted,
        total: sorted.length
    });
});

/**
 * POST /api/operations/:id/annotations
 * Добавить новую аннотацию
 */
app.post('/api/operations/:id/annotations', (req, res) => {
    const operation = operations.find(op => op.id === req.params.id);
    if (!operation) {
        return res.status(404).json({ success: false, error: 'Операция не найдена' });
    }
    
    const { timestamp, endTimestamp, text, author, type, color, phase } = req.body;
    
    if (timestamp === undefined || !text || !text.trim()) {
        return res.status(400).json({ 
            success: false, 
            error: 'Необходимы поля: timestamp и text' 
        });
    }
    
    const annotation = {
        id: uuidv4(),
        operationId: req.params.id,
        timestamp: parseFloat(timestamp),
        endTimestamp: endTimestamp ? parseFloat(endTimestamp) : null,
        text: text.trim(),
        author: author || 'Аноним',
        type: type || 'comment', // comment | note | question | important
        color: color || '#4285f4',
        phase: phase || null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        replies: []
    };
    
    if (!annotations[req.params.id]) {
        annotations[req.params.id] = [];
    }
    annotations[req.params.id].push(annotation);
    
    saveAnnotations();
    console.log(`✓ Новая аннотация для ${req.params.id} в ${Utils_formatTime(timestamp)}`);
    
    res.status(201).json({
        success: true,
        data: annotation,
        message: 'Аннотация добавлена'
    });
});

/**
 * PUT /api/operations/:id/annotations/:annotationId
 * Обновить аннотацию
 */
app.put('/api/operations/:id/annotations/:annotationId', (req, res) => {
    const opAnnotations = annotations[req.params.id];
    if (!opAnnotations) {
        return res.status(404).json({ success: false, error: 'Аннотации не найдены' });
    }
    
    const index = opAnnotations.findIndex(a => a.id === req.params.annotationId);
    if (index === -1) {
        return res.status(404).json({ success: false, error: 'Аннотация не найдена' });
    }
    
    const { text, type, color, endTimestamp } = req.body;
    
    if (text !== undefined) opAnnotations[index].text = text.trim();
    if (type !== undefined) opAnnotations[index].type = type;
    if (color !== undefined) opAnnotations[index].color = color;
    if (endTimestamp !== undefined) opAnnotations[index].endTimestamp = parseFloat(endTimestamp);
    opAnnotations[index].updatedAt = new Date().toISOString();
    
    saveAnnotations();
    
    res.json({
        success: true,
        data: opAnnotations[index],
        message: 'Аннотация обновлена'
    });
});

/**
 * DELETE /api/operations/:id/annotations/:annotationId
 * Удалить аннотацию
 */
app.delete('/api/operations/:id/annotations/:annotationId', (req, res) => {
    const opAnnotations = annotations[req.params.id];
    if (!opAnnotations) {
        return res.status(404).json({ success: false, error: 'Аннотации не найдены' });
    }
    
    const index = opAnnotations.findIndex(a => a.id === req.params.annotationId);
    if (index === -1) {
        return res.status(404).json({ success: false, error: 'Аннотация не найдена' });
    }
    
    const deleted = opAnnotations.splice(index, 1)[0];
    saveAnnotations();
    
    res.json({
        success: true,
        data: deleted,
        message: 'Аннотация удалена'
    });
});

/**
 * POST /api/operations/:id/annotations/:annotationId/replies
 * Ответить на аннотацию
 */
app.post('/api/operations/:id/annotations/:annotationId/replies', (req, res) => {
    const opAnnotations = annotations[req.params.id];
    if (!opAnnotations) {
        return res.status(404).json({ success: false, error: 'Аннотации не найдены' });
    }
    
    const annotation = opAnnotations.find(a => a.id === req.params.annotationId);
    if (!annotation) {
        return res.status(404).json({ success: false, error: 'Аннотация не найдена' });
    }
    
    const { text, author } = req.body;
    if (!text || !text.trim()) {
        return res.status(400).json({ success: false, error: 'Необходимо поле text' });
    }
    
    const reply = {
        id: uuidv4(),
        text: text.trim(),
        author: author || 'Аноним',
        createdAt: new Date().toISOString()
    };
    
    annotation.replies.push(reply);
    annotation.updatedAt = new Date().toISOString();
    saveAnnotations();
    
    res.status(201).json({
        success: true,
        data: reply,
        message: 'Ответ добавлен'
    });
});

// Вспомогательная функция форматирования времени для лога
function Utils_formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// ==================== STATIC FILES ====================

// Создаём папку uploads если не существует
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
}

app.use('/uploads', express.static(uploadsDir));
app.use('/assets', express.static(path.join(__dirname, 'assets')));

// ==================== ERROR HANDLING ====================

app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(500).json({
        success: false,
        error: 'Внутренняя ошибка сервера',
        message: process.env.NODE_ENV === 'development' ? err.message : undefined
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        success: false,
        error: 'Endpoint не найден'
    });
});

// ==================== START SERVER ====================

app.listen(PORT, () => {
    console.log(`
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   🧠 NeuroVision Platform                                      ║
║   Интеллектуальная видеоплатформа для анализа                  ║
║   нейрохирургических операций                                  ║
║                                                                ║
║   Server: http://localhost:${PORT}                               ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    `);
});

module.exports = app;
