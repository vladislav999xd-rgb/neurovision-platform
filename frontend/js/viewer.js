/**
 * NeuroVision Platform - Viewer
 * Страница просмотра операции с видеоплеером и таймлайном
 */

(function() {
    'use strict';
    
    let operation = null;
    let currentTime = 0;
    let duration = 0;
    let isPlaying = false;
    let subtitlesEnabled = true;
    let activeFilters = {
        instruments: [],
        anatomy: []
    };
    
    document.addEventListener('DOMContentLoaded', init);
    
    async function init() {
        const operationId = Utils.getUrlParam('id');
        
        if (!operationId) {
            window.location.href = 'dashboard.html';
            return;
        }
        
        try {
            await loadOperation(operationId);
            setupEventListeners();
            startTimeSimulation();
        } catch (error) {
            console.error('Init error:', error);
        }
    }
    
    async function loadOperation(id) {
        try {
            const response = await API.getOperation(id);
            operation = response.data;
            
            // Обновить UI
            renderOperationInfo();
            renderTimeline();
            renderPhases();
            renderInstruments();
            renderAnatomy();
            renderEvents();
            renderSubtitles();
            loadRelatedOperations();
            
            // Установить видео
            const videoPlayer = document.getElementById('videoPlayer');
            if (videoPlayer && operation.videoUrl) {
                videoPlayer.src = operation.videoUrl;
                
                // Обработчики для HTML5 video
                videoPlayer.addEventListener('loadedmetadata', () => {
                    // Если длительность не задана, берем из видео
                    if (!duration) {
                        duration = videoPlayer.duration;
                        document.getElementById('totalTime').textContent = Utils.formatTime(duration);
                    }
                });
                
                videoPlayer.addEventListener('timeupdate', () => {
                    currentTime = videoPlayer.currentTime;
                    updateTimeDisplay();
                    updateSubtitle();
                });
                
                videoPlayer.addEventListener('play', () => {
                    isPlaying = true;
                    document.getElementById('playPauseBtn').innerHTML = '<i class="fas fa-pause"></i>';
                });
                
                videoPlayer.addEventListener('pause', () => {
                    isPlaying = false;
                    document.getElementById('playPauseBtn').innerHTML = '<i class="fas fa-play"></i>';
                });
            }
            
            // Установить длительность
            duration = operation.duration || 0;
            document.getElementById('totalTime').textContent = Utils.formatTime(duration);
            
            // Таймлайн метки времени
            document.getElementById('timelineQuarter1').textContent = Utils.formatTime(duration * 0.25);
            document.getElementById('timelineHalf').textContent = Utils.formatTime(duration * 0.5);
            document.getElementById('timelineQuarter3').textContent = Utils.formatTime(duration * 0.75);
            document.getElementById('timelineEnd').textContent = Utils.formatTime(duration);
            
        } catch (error) {
            console.error('Load operation error:', error);
            alert('Ошибка загрузки операции');
        }
    }
    
    function renderOperationInfo() {
        document.title = `NeuroVision - ${operation.title}`;
        document.getElementById('operationTitle').textContent = operation.title;
        document.getElementById('operationSurgeon').textContent = operation.surgeon;
        document.getElementById('operationDate').textContent = Utils.formatDate(operation.date);
        document.getElementById('operationDuration').textContent = Utils.formatDuration(operation.duration);
        document.getElementById('operationType').textContent = operation.type;
        document.getElementById('operationDescription').textContent = operation.description || 'Описание не указано';
        
        // Статус
        const statusBadge = document.getElementById('statusBadge');
        const processBtn = document.getElementById('processVideoBtn');
        
        if (operation.status === 'completed') {
            statusBadge.textContent = 'Обработано';
            statusBadge.className = 'tag phase';
            if (processBtn) processBtn.classList.add('hidden');
        } else {
            statusBadge.textContent = 'Ожидает обработки';
            statusBadge.className = 'tag event';
            if (processBtn) {
                processBtn.classList.remove('hidden');
                processBtn.addEventListener('click', startAIProcessing);
            }
        }
    }
    
    function renderTimeline() {
        const segmentsContainer = document.getElementById('timelineSegments');
        const eventsContainer = document.getElementById('timelineEvents');
        
        // Если нет segments или duration, ждём загрузки видео
        if (!operation.segments || operation.segments.length === 0) {
            console.log('No segments to render on timeline');
            return;
        }
        
        // Используем duration из операции или из видео
        const totalDuration = duration || operation.duration || 180;
        
        // Фазы на таймлайне
        const phases = operation.segments.filter(s => s.type === 'phase');
        console.log('Rendering phases:', phases.length);
        
        segmentsContainer.innerHTML = phases.map((phase, index) => {
            const left = (phase.startTime / totalDuration) * 100;
            const width = ((phase.endTime - phase.startTime) / totalDuration) * 100;
            // Цвета для разных фаз
            const colors = ['#4285f4', '#34a853', '#fbbc04', '#ea4335', '#9c27b0', '#00bcd4'];
            const color = colors[index % colors.length];
            return `
                <div class="timeline-segment phase" 
                     style="left: ${left}%; width: ${width}%; background-color: ${color};"
                     data-start="${phase.startTime}"
                     data-name="${phase.name}"
                     title="${phase.name} (${Utils.formatTime(phase.startTime)} - ${Utils.formatTime(phase.endTime)})">
                </div>
            `;
        }).join('');
        
        // События на таймлайне
        const events = operation.segments.filter(s => s.type === 'event');
        eventsContainer.innerHTML = events.map(event => {
            const left = (event.startTime / totalDuration) * 100;
            return `
                <div class="timeline-marker" 
                     style="left: ${left}%;"
                     data-start="${event.startTime}"
                     data-tooltip="${event.name}">
                </div>
            `;
        }).join('');
    }
    
    function renderPhases() {
        const container = document.getElementById('phasesList');
        const phases = (operation.segments || []).filter(s => s.type === 'phase');
        
        if (phases.length === 0) {
            container.innerHTML = '<div class="text-muted text-center p-3">Этапы не определены</div>';
            return;
        }
        
        container.innerHTML = phases.map(phase => `
            <div class="segment-item" data-start="${phase.startTime}" onclick="seekTo(${phase.startTime})">
                <div class="segment-thumb">
                    <img src="${phase.thumbnailUrl || operation.thumbnailUrl}" alt="${phase.name}">
                </div>
                <div class="segment-info">
                    <div class="segment-name">${phase.name}</div>
                    <div class="segment-time">
                        ${Utils.formatTime(phase.startTime)} - ${Utils.formatTime(phase.endTime)}
                    </div>
                    <div class="segment-confidence">
                        <i class="fas fa-robot"></i> ${Utils.formatConfidence(phase.confidence)}
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    function renderInstruments() {
        const filterContainer = document.getElementById('instrumentFilters');
        const listContainer = document.getElementById('instrumentsList');
        const instruments = (operation.detections || []).filter(d => d.type === 'instrument');
        
        // Уникальные инструменты для фильтров
        const uniqueInstruments = [...new Set(instruments.map(i => i.name))];
        
        filterContainer.innerHTML = uniqueInstruments.map(name => `
            <button class="quick-filter" data-instrument="${name}">${name}</button>
        `).join('');
        
        // Список детекций
        renderInstrumentList(instruments);
        
        // Обработчики фильтров
        filterContainer.querySelectorAll('.quick-filter').forEach(btn => {
            btn.addEventListener('click', () => {
                btn.classList.toggle('active');
                const name = btn.dataset.instrument;
                
                if (btn.classList.contains('active')) {
                    activeFilters.instruments.push(name);
                } else {
                    activeFilters.instruments = activeFilters.instruments.filter(n => n !== name);
                }
                
                const filtered = activeFilters.instruments.length > 0
                    ? instruments.filter(i => activeFilters.instruments.includes(i.name))
                    : instruments;
                    
                renderInstrumentList(filtered);
            });
        });
    }
    
    function renderInstrumentList(instruments) {
        const container = document.getElementById('instrumentsList');
        
        if (instruments.length === 0) {
            container.innerHTML = '<div class="text-muted text-center p-3">Инструменты не обнаружены</div>';
            return;
        }
        
        container.innerHTML = instruments.map(inst => `
            <div class="detection-item" onclick="seekTo(${inst.timestamp})">
                <div class="detection-icon instrument">
                    <i class="fas fa-tools"></i>
                </div>
                <div class="detection-info">
                    <div class="detection-name">${inst.name}</div>
                    <div class="detection-time">
                        ${Utils.formatTime(inst.timestamp)} • ${Utils.formatConfidence(inst.confidence)}
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    function renderAnatomy() {
        const filterContainer = document.getElementById('anatomyFilters');
        const listContainer = document.getElementById('anatomyList');
        const anatomy = (operation.detections || []).filter(d => d.type === 'anatomy');
        
        // Уникальные структуры для фильтров
        const uniqueAnatomy = [...new Set(anatomy.map(a => a.name))];
        
        filterContainer.innerHTML = uniqueAnatomy.map(name => `
            <button class="quick-filter" data-anatomy="${name}">${name}</button>
        `).join('');
        
        // Список детекций
        renderAnatomyList(anatomy);
        
        // Обработчики фильтров
        filterContainer.querySelectorAll('.quick-filter').forEach(btn => {
            btn.addEventListener('click', () => {
                btn.classList.toggle('active');
                const name = btn.dataset.anatomy;
                
                if (btn.classList.contains('active')) {
                    activeFilters.anatomy.push(name);
                } else {
                    activeFilters.anatomy = activeFilters.anatomy.filter(n => n !== name);
                }
                
                const filtered = activeFilters.anatomy.length > 0
                    ? anatomy.filter(a => activeFilters.anatomy.includes(a.name))
                    : anatomy;
                    
                renderAnatomyList(filtered);
            });
        });
    }
    
    function renderAnatomyList(anatomy) {
        const container = document.getElementById('anatomyList');
        
        if (anatomy.length === 0) {
            container.innerHTML = '<div class="text-muted text-center p-3">Анатомические структуры не обнаружены</div>';
            return;
        }
        
        container.innerHTML = anatomy.map(anat => `
            <div class="detection-item" onclick="seekTo(${anat.timestamp})">
                <div class="detection-icon anatomy">
                    <i class="fas fa-brain"></i>
                </div>
                <div class="detection-info">
                    <div class="detection-name">${anat.name}</div>
                    <div class="detection-time">
                        ${Utils.formatTime(anat.timestamp)} • ${Utils.formatConfidence(anat.confidence)}
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    function renderEvents() {
        const container = document.getElementById('eventsList');
        const events = (operation.segments || []).filter(s => s.type === 'event');
        
        if (events.length === 0) {
            container.innerHTML = '<div class="text-muted text-center p-3">События не обнаружены</div>';
            return;
        }
        
        container.innerHTML = events.map(event => `
            <div class="segment-item" data-start="${event.startTime}" onclick="seekTo(${event.startTime})">
                <div class="segment-thumb">
                    <img src="${event.thumbnailUrl || operation.thumbnailUrl}" alt="${event.name}">
                </div>
                <div class="segment-info">
                    <div class="segment-name">
                        <span class="tag event" style="padding: 2px 6px; margin-right: 4px;">
                            ${event.metadata?.severity === 'high' ? '⚠️' : '📍'}
                        </span>
                        ${event.name}
                    </div>
                    <div class="segment-time">
                        ${Utils.formatTime(event.startTime)}
                    </div>
                    <div class="segment-confidence">
                        <i class="fas fa-robot"></i> ${Utils.formatConfidence(event.confidence)}
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    function renderSubtitles() {
        const container = document.getElementById('subtitlesList');
        const subtitles = operation.subtitles || [];
        
        if (subtitles.length === 0) {
            container.innerHTML = '<div class="text-muted text-center p-3">Субтитры недоступны</div>';
            return;
        }
        
        container.innerHTML = subtitles.map(sub => `
            <div class="subtitle-item" data-start="${sub.startTime}" data-end="${sub.endTime}" 
                 onclick="seekTo(${sub.startTime})">
                <span class="subtitle-time">${Utils.formatTime(sub.startTime)}</span>
                <span class="subtitle-text">${sub.text}</span>
            </div>
        `).join('');
    }
    
    async function loadRelatedOperations() {
        try {
            const response = await API.getOperations({ type: operation.type, limit: 3 });
            const related = response.data.filter(op => op.id !== operation.id).slice(0, 3);
            
            const container = document.getElementById('relatedList');
            
            if (related.length === 0) {
                container.innerHTML = '<div class="text-muted text-center p-3">Нет похожих операций</div>';
                return;
            }
            
            container.innerHTML = related.map(op => `
                <div class="segment-item" onclick="window.location.href='viewer.html?id=${op.id}'">
                    <div class="segment-thumb">
                        <img src="${op.thumbnailUrl}" alt="${op.title}">
                    </div>
                    <div class="segment-info">
                        <div class="segment-name">${op.title}</div>
                        <div class="segment-time">${op.surgeon}</div>
                    </div>
                </div>
            `).join('');
            
        } catch (error) {
            console.error('Related operations error:', error);
        }
    }
    
    function setupEventListeners() {
        // Toggle Sidebar меню
        const menuBtn = document.getElementById('menuBtn');
        const sidebar = document.getElementById('sidebar');
        const pageContent = document.querySelector('.page-content');
        
        if (menuBtn && sidebar) {
            menuBtn.addEventListener('click', () => {
                sidebar.classList.toggle('collapsed');
                if (pageContent) {
                    pageContent.classList.toggle('sidebar-collapsed');
                }
            });
        }
        
        // Tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                // Убрать active со всех табов
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // Активировать выбранный
                tab.classList.add('active');
                const tabId = tab.dataset.tab + 'Tab';
                document.getElementById(tabId).classList.add('active');
            });
        });
        
        // Timeline click
        const timelineTrack = document.getElementById('timelineTrack');
        if (timelineTrack) {
            timelineTrack.addEventListener('click', (e) => {
                const rect = timelineTrack.getBoundingClientRect();
                const percent = (e.clientX - rect.left) / rect.width;
                const time = percent * duration;
                seekTo(time);
            });
        }
        
        // Progress bar click
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.addEventListener('click', (e) => {
                const rect = progressBar.getBoundingClientRect();
                const percent = (e.clientX - rect.left) / rect.width;
                const time = percent * duration;
                seekTo(time);
            });
        }
        
        // Play/Pause
        const playPauseBtn = document.getElementById('playPauseBtn');
        if (playPauseBtn) {
            playPauseBtn.addEventListener('click', togglePlay);
        }
        
        // Subtitles toggle
        const subtitlesBtn = document.getElementById('subtitlesBtn');
        if (subtitlesBtn) {
            subtitlesBtn.addEventListener('click', () => {
                subtitlesEnabled = !subtitlesEnabled;
                subtitlesBtn.classList.toggle('active', subtitlesEnabled);
                document.getElementById('subtitlesOverlay').style.display = 
                    subtitlesEnabled ? 'block' : 'none';
            });
        }
        
        // Fullscreen
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        if (fullscreenBtn) {
            fullscreenBtn.addEventListener('click', toggleFullscreen);
        }
        
        // Favorite
        const favoriteBtn = document.getElementById('favoriteBtn');
        if (favoriteBtn) {
            favoriteBtn.addEventListener('click', () => {
                favoriteBtn.querySelector('i').classList.toggle('far');
                favoriteBtn.querySelector('i').classList.toggle('fas');
            });
        }
        
        // Search in video
        const searchBtn = document.getElementById('searchInVideoBtn');
        const searchInput = document.getElementById('searchInVideo');
        if (searchBtn && searchInput) {
            searchBtn.addEventListener('click', () => searchInVideo(searchInput.value));
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') searchInVideo(searchInput.value);
            });
        }
    }
    
    function startTimeSimulation() {
        // Симуляция воспроизведения для демо
        setInterval(() => {
            if (isPlaying && duration > 0) {
                currentTime = Math.min(currentTime + 1, duration);
                updateTimeDisplay();
                updateSubtitle();
            }
        }, 1000);
    }
    
    function updateTimeDisplay() {
        const percent = (currentTime / duration) * 100;
        
        document.getElementById('currentTime').textContent = Utils.formatTime(currentTime);
        document.getElementById('progressFill').style.width = percent + '%';
        document.getElementById('timelinePlayhead').style.left = percent + '%';
        
        // Highlight active segment
        document.querySelectorAll('.segment-item').forEach(item => {
            const start = parseFloat(item.dataset.start);
            const isActive = Math.abs(currentTime - start) < 30;
            item.classList.toggle('active', isActive);
        });
    }
    
    function updateSubtitle() {
        const subtitles = operation.subtitles || [];
        const currentSub = subtitles.find(sub => 
            currentTime >= sub.startTime && currentTime <= sub.endTime
        );
        
        const overlay = document.getElementById('currentSubtitle');
        if (currentSub && subtitlesEnabled) {
            overlay.textContent = currentSub.text;
            overlay.parentElement.style.display = 'block';
        } else {
            overlay.parentElement.style.display = 'none';
        }
        
        // Highlight in list
        document.querySelectorAll('.subtitle-item').forEach(item => {
            const start = parseFloat(item.dataset.start);
            const end = parseFloat(item.dataset.end);
            item.classList.toggle('active', currentTime >= start && currentTime <= end);
        });
        
        // Обновляем детекции на видео
        updateDetectionOverlay();
    }
    
    function updateDetectionOverlay() {
        const overlay = document.getElementById('detectionOverlay');
        if (!overlay) return;
        
        const detections = operation.detections || [];
        // Показываем детекции в пределах 5 секунд от текущего времени
        const activeDetections = detections.filter(d => 
            Math.abs(d.timestamp - currentTime) < 5
        );
        
        if (activeDetections.length === 0) {
            overlay.innerHTML = '';
            return;
        }
        
        overlay.innerHTML = activeDetections.map(det => {
            // Генерируем позицию если не указана
            const box = det.boundingBox || {
                x: 0.2 + Math.random() * 0.5,
                y: 0.2 + Math.random() * 0.4,
                width: 0.15 + Math.random() * 0.1,
                height: 0.12 + Math.random() * 0.08
            };
            
            const left = box.x * 100;
            const top = box.y * 100;
            const width = box.width * 100;
            const height = box.height * 100;
            
            return `
                <div class="detection-box ${det.type}" 
                     style="left: ${left}%; top: ${top}%; width: ${width}%; height: ${height}%;">
                    <span class="detection-box-label">
                        <i class="fas fa-${det.type === 'instrument' ? 'tools' : 'microscope'}"></i>
                        ${det.name}
                    </span>
                </div>
            `;
        }).join('');
    }
    
    function togglePlay() {
        const videoPlayer = document.getElementById('videoPlayer');
        if (videoPlayer) {
            if (videoPlayer.paused) {
                videoPlayer.play();
            } else {
                videoPlayer.pause();
            }
        } else {
            // Fallback для симуляции если видео не загружено
            isPlaying = !isPlaying;
            const btn = document.getElementById('playPauseBtn');
            btn.innerHTML = isPlaying ? '<i class="fas fa-pause"></i>' : '<i class="fas fa-play"></i>';
        }
    }
    
    function toggleFullscreen() {
        const playerWrapper = document.getElementById('playerWrapper');
        
        if (!document.fullscreenElement) {
            playerWrapper.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }
    
    function searchInVideo(query) {
        if (!query.trim()) return;
        
        query = query.toLowerCase();
        
        // Поиск в сегментах
        const segments = (operation.segments || []).filter(s => 
            s.name.toLowerCase().includes(query) ||
            (s.tags || []).some(t => t.toLowerCase().includes(query))
        );
        
        // Поиск в детекциях
        const detections = (operation.detections || []).filter(d =>
            d.name.toLowerCase().includes(query)
        );
        
        // Поиск в субтитрах
        const subtitles = (operation.subtitles || []).filter(s =>
            s.text.toLowerCase().includes(query)
        );
        
        if (segments.length > 0) {
            seekTo(segments[0].startTime);
        } else if (detections.length > 0) {
            seekTo(detections[0].timestamp);
        } else if (subtitles.length > 0) {
            seekTo(subtitles[0].startTime);
        } else {
            alert('Ничего не найдено');
        }
    }
    
    // Глобальная функция для перехода к времени
    window.seekTo = function(time) {
        currentTime = time;
        
        // Перемотка HTML5 video
        const videoPlayer = document.getElementById('videoPlayer');
        if (videoPlayer) {
            videoPlayer.currentTime = time;
        }
        
        updateTimeDisplay();
        updateSubtitle();
        console.log('Seeking to:', Utils.formatTime(time));
    };
    
    // AI Обработка видео
    async function startAIProcessing() {
        const modal = document.getElementById('processingModal');
        const progressBar = document.getElementById('processingProgress');
        const processingText = document.getElementById('processingText');
        
        modal.classList.add('active');
        
        const steps = [
            { id: 'step1', text: 'Генерация превью изображения...', duration: 2000 },
            { id: 'step2', text: 'Распознавание фаз операции...', duration: 3000 },
            { id: 'step3', text: 'Детекция хирургических инструментов...', duration: 3000 },
            { id: 'step4', text: 'Сегментация анатомических структур...', duration: 2500 },
            { id: 'step5', text: 'Генерация субтитров и описаний...', duration: 2500 }
        ];
        
        let progress = 0;
        const progressStep = 100 / steps.length;
        
        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            const stepEl = document.getElementById(step.id);
            
            // Активируем текущий шаг
            stepEl.classList.add('active');
            stepEl.querySelector('i').className = 'fas fa-spinner fa-spin';
            processingText.textContent = step.text;
            
            await new Promise(resolve => setTimeout(resolve, step.duration));
            
            // Завершаем шаг
            stepEl.classList.remove('active');
            stepEl.classList.add('completed');
            stepEl.querySelector('i').className = 'fas fa-check-circle';
            
            progress += progressStep;
            progressBar.style.width = progress + '%';
        }
        
        processingText.textContent = 'Обработка завершена! Обновление данных...';
        
        // Симулируем обновление данных
        await simulateAIResults();
        
        // Обновляем UI с новыми данными
        renderOperationInfo();
        renderTimeline();
        renderPhases();
        renderInstruments();
        renderAnatomy();
        renderEvents();
        renderSubtitles();
        
        setTimeout(() => {
            modal.classList.remove('active');
            // Показываем уведомление об успехе
            alert('✓ AI обработка завершена успешно! Данные обновлены.');
        }, 1000);
    }
    
    async function simulateAIResults() {
        // Генерируем AI результаты для операции
        const generatedData = {
            status: 'completed',
            segments: generatePhases(),
            detections: generateDetections(),
            subtitles: generateSubtitles()
        };
        
        // Отправляем на сервер
        try {
            const response = await API.processOperation(operation.id, generatedData);
            if (response.success) {
                // Обновляем локальный объект операции
                operation.status = 'completed';
                operation.segments = generatedData.segments;
                operation.detections = generatedData.detections;
                operation.subtitles = generatedData.subtitles;
                console.log('✓ AI обработка сохранена на сервере');
            }
        } catch (error) {
            console.log('Симуляция обработки завершена локально');
            // Обновляем локально даже если сервер недоступен
            operation.status = 'completed';
            operation.segments = generatedData.segments;
            operation.detections = generatedData.detections;
            operation.subtitles = generatedData.subtitles;
        }
    }
    
    function generatePhases() {
        const phaseNames = [
            'Подготовка и позиционирование',
            'Краниотомия / Доступ',
            'Идентификация структур',
            'Основной этап операции',
            'Гемостаз и контроль',
            'Закрытие'
        ];
        
        const phases = [];
        const phaseDuration = duration / phaseNames.length;
        
        phaseNames.forEach((name, i) => {
            phases.push({
                type: 'phase',
                name: name,
                startTime: Math.round(i * phaseDuration),
                endTime: Math.round((i + 1) * phaseDuration),
                confidence: 0.85 + Math.random() * 0.14,
                tags: ['AI-detected']
            });
        });
        
        return phases;
    }
    
    function generateDetections() {
        const instruments = ['Микроножницы', 'Биполярный пинцет', 'Аспиратор', 'Микрокрючок', 'Ретрактор'];
        const anatomy = ['Твердая мозговая оболочка', 'Нервная ткань', 'Сосуды', 'Арахноидальная оболочка'];
        
        const detections = [];
        const count = 15 + Math.floor(Math.random() * 10);
        
        for (let i = 0; i < count; i++) {
            const isInstrument = Math.random() > 0.4;
            const items = isInstrument ? instruments : anatomy;
            
            detections.push({
                type: isInstrument ? 'instrument' : 'anatomy',
                name: items[Math.floor(Math.random() * items.length)],
                timestamp: Math.floor(Math.random() * duration),
                confidence: 0.75 + Math.random() * 0.24,
                boundingBox: {
                    x: Math.random() * 0.6 + 0.1,
                    y: Math.random() * 0.6 + 0.1,
                    width: Math.random() * 0.2 + 0.1,
                    height: Math.random() * 0.2 + 0.1
                }
            });
        }
        
        return detections.sort((a, b) => a.timestamp - b.timestamp);
    }
    
    function generateSubtitles() {
        const descriptions = [
            'Хирург выполняет разрез кожи в области операционного доступа',
            'Проводится диссекция мягких тканей с использованием биполярной коагуляции',
            'Краниотомия: формирование костного окна для доступа к операционному полю',
            'Вскрытие твердой мозговой оболочки с сохранением арахноидальной оболочки',
            'Идентификация ключевых анатомических ориентиров под микроскопом',
            'Микрохирургическая диссекция в области поражения',
            'Применение ретрактора для улучшения визуализации',
            'Тщательный гемостаз операционного поля',
            'Контрольный осмотр зоны вмешательства',
            'Послойное закрытие операционной раны'
        ];
        
        const subtitles = [];
        const subtitleDuration = duration / descriptions.length;
        
        descriptions.forEach((text, i) => {
            subtitles.push({
                startTime: Math.round(i * subtitleDuration),
                endTime: Math.round((i + 0.9) * subtitleDuration),
                text: text
            });
        });
        
        return subtitles;
    }
    
})();
