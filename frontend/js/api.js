/**
 * NeuroVision Platform - API Client
 * Клиент для взаимодействия с backend API
 */

const API_BASE = 'http://localhost:3000/api';

const API = {
    /**
     * Базовый метод для выполнения запросов
     */
    async request(endpoint, options = {}) {
        const url = `${API_BASE}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        try {
            const response = await fetch(url, { ...defaultOptions, ...options });
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Ошибка запроса');
            }
            
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },
    
    /**
     * Получить список всех операций
     */
    async getOperations(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/operations${query ? '?' + query : ''}`);
    },
    
    /**
     * Получить операцию по ID
     */
    async getOperation(id) {
        return this.request(`/operations/${id}`);
    },
    
    /**
     * Запустить AI обработку операции
     */
    async processOperation(id, data) {
        return this.request(`/operations/${id}/process`, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },
    
    /**
     * Получить сегменты операции
     */
    async getSegments(operationId, params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/operations/${operationId}/segments${query ? '?' + query : ''}`);
    },
    
    /**
     * Получить детекции операции
     */
    async getDetections(operationId, params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/operations/${operationId}/detections${query ? '?' + query : ''}`);
    },
    
    /**
     * Получить субтитры операции
     */
    async getSubtitles(operationId) {
        return this.request(`/operations/${operationId}/subtitles`);
    },
    
    /**
     * Загрузить видео
     */
    async uploadVideo(formData) {
        const url = `${API_BASE}/videos/upload`;
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData, // FormData - не JSON
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Ошибка загрузки');
            }
            
            return data;
        } catch (error) {
            console.error('Upload Error:', error);
            throw error;
        }
    },
    
    /**
     * Комбинированный поиск
     */
    async search(params) {
        return this.request('/search', {
            method: 'POST',
            body: JSON.stringify(params),
        });
    },
    
    /**
     * Получить справочники тегов
     */
    async getTags() {
        return this.request('/tags');
    },
    
    /**
     * Получить справочник по типу
     */
    async getTagsByType(type) {
        return this.request(`/tags/${type}`);
    },
    
    /**
     * Получить статистику
     */
    async getStats() {
        return this.request('/stats');
    },
    
    // ==================== ANNOTATIONS ====================
    
    /**
     * Получить аннотации операции
     */
    async getAnnotations(operationId) {
        return this.request(`/operations/${operationId}/annotations`);
    },
    
    /**
     * Создать аннотацию
     */
    async createAnnotation(operationId, data) {
        return this.request(`/operations/${operationId}/annotations`, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },
    
    /**
     * Обновить аннотацию
     */
    async updateAnnotation(operationId, annotationId, data) {
        return this.request(`/operations/${operationId}/annotations/${annotationId}`, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },
    
    /**
     * Удалить аннотацию
     */
    async deleteAnnotation(operationId, annotationId) {
        return this.request(`/operations/${operationId}/annotations/${annotationId}`, {
            method: 'DELETE'
        });
    },
    
    /**
     * Ответить на аннотацию
     */
    async replyToAnnotation(operationId, annotationId, data) {
        return this.request(`/operations/${operationId}/annotations/${annotationId}/replies`, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },
};

/**
 * Вспомогательные функции
 */
const Utils = {
    /**
     * Форматирование времени (секунды → MM:SS или HH:MM:SS)
     */
    formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return '0:00';
        
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hrs > 0) {
            return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    },
    
    /**
     * Форматирование длительности (секунды → "X ч Y мин")
     */
    formatDuration(seconds) {
        if (!seconds || isNaN(seconds)) return '-';
        
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        
        if (hrs > 0) {
            return `${hrs} ч ${mins} мин`;
        }
        return `${mins} мин`;
    },
    
    /**
     * Форматирование даты
     */
    formatDate(dateStr) {
        if (!dateStr) return '-';
        
        const date = new Date(dateStr);
        return date.toLocaleDateString('ru-RU', {
            day: 'numeric',
            month: 'long',
            year: 'numeric',
        });
    },
    
    /**
     * Форматирование даты (короткое)
     */
    formatDateShort(dateStr) {
        if (!dateStr) return '-';
        
        const date = new Date(dateStr);
        return date.toLocaleDateString('ru-RU', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
        });
    },
    
    /**
     * Форматирование уверенности модели
     */
    formatConfidence(confidence) {
        if (!confidence && confidence !== 0) return '-';
        return `${Math.round(confidence * 100)}%`;
    },
    
    /**
     * Получить цвет по типу
     */
    getTypeColor(type) {
        const colors = {
            phase: '#4285f4',
            instrument: '#34a853',
            anatomy: '#9c27b0',
            event: '#ea4335',
        };
        return colors[type] || '#5f6368';
    },
    
    /**
     * Debounce функция
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    /**
     * Показать уведомление
     */
    showNotification(message, type = 'success') {
        const notification = document.getElementById('notification');
        if (!notification) return;
        
        notification.className = `notification ${type} show`;
        notification.querySelector('#notificationText').textContent = message;
        
        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    },
    
    /**
     * Получить параметр из URL
     */
    getUrlParam(name) {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(name);
    },
    
    /**
     * Установить параметр в URL
     */
    setUrlParam(name, value) {
        const url = new URL(window.location);
        if (value) {
            url.searchParams.set(name, value);
        } else {
            url.searchParams.delete(name);
        }
        window.history.replaceState({}, '', url);
    },
    
    /**
     * Проверка авторизации
     */
    checkAuth() {
        const user = localStorage.getItem('neurovision_user');
        if (!user && !window.location.pathname.includes('index.html')) {
            window.location.href = 'index.html';
            return false;
        }
        return true;
    },
    
    /**
     * Получить текущего пользователя
     */
    getCurrentUser() {
        const user = localStorage.getItem('neurovision_user');
        return user ? JSON.parse(user) : null;
    },
    
    /**
     * Выход из системы
     */
    logout() {
        localStorage.removeItem('neurovision_user');
        window.location.href = 'index.html';
    },
    
    /**
     * Создание HTML элемента из строки
     */
    createElement(html) {
        const template = document.createElement('template');
        template.innerHTML = html.trim();
        return template.content.firstChild;
    },
    
    /**
     * Форматирование размера файла
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
};

// Глобальный доступ
window.API = API;
window.Utils = Utils;
