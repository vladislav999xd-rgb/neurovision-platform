/**
 * NeuroVision Platform - Upload
 * Страница загрузки видео
 */

(function() {
    'use strict';
    
    let selectedFile = null;
    let tags = {};
    
    document.addEventListener('DOMContentLoaded', init);
    
    async function init() {
        try {
            await loadTags();
            setupEventListeners();
            setDefaultDate();
        } catch (error) {
            console.error('Init error:', error);
        }
    }
    
    async function loadTags() {
        try {
            const response = await API.getTags();
            tags = response.data;
            
            // Заполнить select хирургов
            const surgeonSelect = document.getElementById('surgeon');
            if (surgeonSelect && tags.surgeons) {
                tags.surgeons.forEach(surgeon => {
                    const option = document.createElement('option');
                    option.value = surgeon.name;
                    option.textContent = surgeon.name;
                    surgeonSelect.appendChild(option);
                });
            }
            
            // Заполнить select типов операций
            const typeSelect = document.getElementById('operationType');
            if (typeSelect && tags.operationTypes) {
                tags.operationTypes.forEach(type => {
                    const option = document.createElement('option');
                    option.value = type.name;
                    option.textContent = type.name;
                    typeSelect.appendChild(option);
                });
            }
            
        } catch (error) {
            console.error('Tags error:', error);
        }
    }
    
    function setDefaultDate() {
        const dateInput = document.getElementById('date');
        if (dateInput) {
            dateInput.value = new Date().toISOString().split('T')[0];
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
        
        const uploadZone = document.getElementById('uploadZone');
        const videoInput = document.getElementById('videoInput');
        const uploadForm = document.getElementById('uploadForm');
        const removeFileBtn = document.getElementById('removeFileBtn');
        
        // Drag & Drop
        if (uploadZone) {
            uploadZone.addEventListener('click', () => videoInput.click());
            
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });
            
            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('dragover');
            });
            
            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFileSelect(files[0]);
                }
            });
        }
        
        // File input change
        if (videoInput) {
            videoInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFileSelect(e.target.files[0]);
                }
            });
        }
        
        // Remove file
        if (removeFileBtn) {
            removeFileBtn.addEventListener('click', () => {
                selectedFile = null;
                document.getElementById('selectedFilePanel').classList.add('hidden');
                videoInput.value = '';
            });
        }
        
        // Form submit
        if (uploadForm) {
            uploadForm.addEventListener('submit', handleSubmit);
        }
    }
    
    function handleFileSelect(file) {
        // Проверка типа файла
        const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
            Utils.showNotification('Неподдерживаемый формат файла', 'error');
            return;
        }
        
        // Проверка размера (10 ГБ)
        const maxSize = 10 * 1024 * 1024 * 1024;
        if (file.size > maxSize) {
            Utils.showNotification('Файл слишком большой (максимум 10 ГБ)', 'error');
            return;
        }
        
        selectedFile = file;
        
        // Показать информацию о файле
        document.getElementById('selectedFilePanel').classList.remove('hidden');
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = Utils.formatFileSize(file.size);
        
        // Попробовать извлечь название из имени файла
        const titleInput = document.getElementById('title');
        if (titleInput && !titleInput.value) {
            const nameWithoutExt = file.name.replace(/\.[^/.]+$/, '').replace(/[_-]/g, ' ');
            titleInput.value = nameWithoutExt;
        }
    }
    
    async function handleSubmit(e) {
        e.preventDefault();
        
        const title = document.getElementById('title').value.trim();
        const surgeon = document.getElementById('surgeon').value;
        const operationType = document.getElementById('operationType').value;
        const date = document.getElementById('date').value;
        const patientId = document.getElementById('patientId').value.trim();
        const description = document.getElementById('description').value.trim();
        
        // Валидация
        if (!title || !surgeon || !operationType || !date) {
            Utils.showNotification('Заполните все обязательные поля', 'error');
            return;
        }
        
        // Показать прогресс
        document.getElementById('progressPanel').classList.remove('hidden');
        document.getElementById('submitBtn').disabled = true;
        
        try {
            // Создаем FormData
            const formData = new FormData();
            formData.append('title', title);
            formData.append('surgeon', surgeon);
            formData.append('type', operationType);
            formData.append('date', date);
            formData.append('patientId', patientId || `P-${Date.now()}`);
            formData.append('description', description);
            
            if (selectedFile) {
                formData.append('video', selectedFile);
            }
            
            // Симуляция загрузки для демо
            await simulateUpload();
            
            // Отправка на сервер
            const response = await API.uploadVideo(formData);
            
            Utils.showNotification('Видео успешно загружено!', 'success');
            
            // Редирект на дашборд через 2 секунды
            setTimeout(() => {
                window.location.href = 'dashboard.html';
            }, 2000);
            
        } catch (error) {
            console.error('Upload error:', error);
            Utils.showNotification('Ошибка загрузки: ' + error.message, 'error');
            document.getElementById('progressPanel').classList.add('hidden');
            document.getElementById('submitBtn').disabled = false;
        }
    }
    
    function simulateUpload() {
        return new Promise((resolve) => {
            const progressFill = document.getElementById('progressFill');
            const progressPercent = document.getElementById('progressPercent');
            const progressStatus = document.getElementById('progressStatus');
            const steps = document.querySelectorAll('#processingSteps .tag');
            
            let progress = 0;
            const stages = [
                { progress: 25, status: 'Загрузка видео...', step: 0 },
                { progress: 50, status: 'Извлечение кадров...', step: 1 },
                { progress: 75, status: 'AI анализ...', step: 2 },
                { progress: 100, status: 'Индексация...', step: 3 }
            ];
            
            let stageIndex = 0;
            
            const interval = setInterval(() => {
                progress += 2;
                
                // Обновить прогресс
                progressFill.style.width = progress + '%';
                progressPercent.textContent = progress + '%';
                
                // Проверить этапы
                if (stageIndex < stages.length && progress >= stages[stageIndex].progress) {
                    progressStatus.textContent = stages[stageIndex].status;
                    
                    // Обновить иконки этапов
                    steps.forEach((step, i) => {
                        if (i < stageIndex) {
                            step.innerHTML = '<i class="fas fa-check"></i> ' + step.textContent.trim();
                            step.classList.add('phase');
                        } else if (i === stageIndex) {
                            step.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ' + step.textContent.replace(/^[\s\S]*?(?=\w)/, '');
                        }
                    });
                    
                    stageIndex++;
                }
                
                if (progress >= 100) {
                    clearInterval(interval);
                    progressStatus.textContent = 'Готово!';
                    
                    // Отметить все этапы как завершённые
                    steps.forEach(step => {
                        step.innerHTML = '<i class="fas fa-check"></i> ' + step.textContent.replace(/^[\s\S]*?(?=\w)/, '');
                        step.classList.add('phase');
                    });
                    
                    resolve();
                }
            }, 50);
        });
    }
    
})();
