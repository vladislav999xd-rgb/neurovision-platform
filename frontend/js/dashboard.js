/**
 * NeuroVision Platform - Dashboard
 * Главная страница с списком операций
 */

(function() {
    'use strict';
    
    let allOperations = [];
    let currentFilter = {};
    let tags = {};
    
    // Инициализация при загрузке
    document.addEventListener('DOMContentLoaded', init);
    
    async function init() {
        try {
            // Загрузка данных
            await Promise.all([
                loadStats(),
                loadOperations(),
                loadTags()
            ]);
            
            // Установка обработчиков
            setupEventListeners();
            
        } catch (error) {
            console.error('Init error:', error);
        }
    }
    
    async function loadStats() {
        try {
            const response = await API.getStats();
            const stats = response.data;
            
            document.getElementById('statOperations').textContent = stats.totalOperations;
            document.getElementById('statCompleted').textContent = stats.completedOperations;
            document.getElementById('statSegments').textContent = stats.totalSegments;
            document.getElementById('statDuration').textContent = 
                Math.round(stats.totalDuration / 3600) + 'ч';
            
        } catch (error) {
            console.error('Stats error:', error);
        }
    }
    
    async function loadOperations() {
        const grid = document.getElementById('videoGrid');
        
        try {
            const response = await API.getOperations(currentFilter);
            allOperations = response.data;
            
            renderOperations(allOperations);
            
        } catch (error) {
            console.error('Operations error:', error);
            grid.innerHTML = '<div class="empty-state"><p>Ошибка загрузки данных</p></div>';
        }
    }
    
    async function loadTags() {
        try {
            const response = await API.getTags();
            tags = response.data;
            
            // Заполнить список хирургов в sidebar
            const surgeonsList = document.getElementById('surgeonsList');
            if (surgeonsList && tags.surgeons) {
                surgeonsList.innerHTML = tags.surgeons.map(surgeon => `
                    <a href="#" class="sidebar-item" data-surgeon="${surgeon.name}">
                        <i class="fas fa-user-md"></i>
                        <span>${surgeon.name}</span>
                    </a>
                `).join('');
            }
            
        } catch (error) {
            console.error('Tags error:', error);
        }
    }
    
    function renderOperations(operations) {
        const grid = document.getElementById('videoGrid');
        const emptyState = document.getElementById('emptyState');
        
        if (!operations || operations.length === 0) {
            grid.innerHTML = '';
            emptyState.classList.remove('hidden');
            return;
        }
        
        emptyState.classList.add('hidden');
        
        grid.innerHTML = operations.map(op => `
            <div class="video-card" data-id="${op.id}" onclick="openOperation('${op.id}')">
                <div class="video-thumbnail">
                    <img src="${op.thumbnailUrl || 'https://via.placeholder.com/320x180?text=No+Preview'}" 
                         alt="${op.title}"
                         onerror="this.src='https://via.placeholder.com/320x180?text=No+Preview'">
                    <span class="video-duration">${Utils.formatDuration(op.duration)}</span>
                    <span class="video-status ${op.status}">${getStatusText(op.status)}</span>
                </div>
                <div class="video-info">
                    <h3 class="video-title">${op.title}</h3>
                    <div class="video-meta">
                        <span class="video-meta-item">
                            <i class="fas fa-user-md"></i>
                            ${op.surgeon}
                        </span>
                        <span class="video-meta-item">
                            <i class="fas fa-calendar"></i>
                            ${Utils.formatDateShort(op.date)}
                        </span>
                    </div>
                    <div class="video-tags">
                        <span class="tag phase">${op.type}</span>
                        ${op.segments ? `<span class="tag">${op.segments.length} сегментов</span>` : ''}
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    function getStatusText(status) {
        const texts = {
            'completed': 'Обработано',
            'processing': 'Обработка',
            'pending': 'В очереди',
            'failed': 'Ошибка'
        };
        return texts[status] || status;
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
        
        // Поиск
        const searchBtn = document.getElementById('searchBtn');
        const searchInput = document.getElementById('globalSearch');
        
        if (searchBtn && searchInput) {
            searchBtn.addEventListener('click', performSearch);
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') performSearch();
            });
        }
        
        // Сортировка
        const sortSelect = document.getElementById('sortSelect');
        if (sortSelect) {
            sortSelect.addEventListener('change', (e) => {
                sortOperations(e.target.value);
            });
        }
        
        // Фильтры в sidebar
        document.querySelectorAll('[data-type]').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                filterByType(e.currentTarget.dataset.type);
            });
        });
        
        document.querySelectorAll('[data-surgeon]').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                filterBySurgeon(e.currentTarget.dataset.surgeon);
            });
        });
        
        document.querySelectorAll('[data-filter]').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                applyQuickFilter(e.currentTarget.dataset.filter);
            });
        });
    }
    
    function performSearch() {
        const query = document.getElementById('globalSearch').value.trim();
        
        if (!query) {
            renderOperations(allOperations);
            return;
        }
        
        const filtered = allOperations.filter(op => {
            const searchText = `${op.title} ${op.surgeon} ${op.type} ${op.description || ''}`.toLowerCase();
            return searchText.includes(query.toLowerCase());
        });
        
        document.getElementById('sectionTitle').textContent = `Результаты поиска: "${query}"`;
        renderOperations(filtered);
    }
    
    function sortOperations(sortBy) {
        let sorted = [...allOperations];
        
        switch (sortBy) {
            case 'date-desc':
                sorted.sort((a, b) => new Date(b.date) - new Date(a.date));
                break;
            case 'date-asc':
                sorted.sort((a, b) => new Date(a.date) - new Date(b.date));
                break;
            case 'duration-desc':
                sorted.sort((a, b) => (b.duration || 0) - (a.duration || 0));
                break;
            case 'duration-asc':
                sorted.sort((a, b) => (a.duration || 0) - (b.duration || 0));
                break;
        }
        
        renderOperations(sorted);
    }
    
    function filterByType(type) {
        const filtered = allOperations.filter(op => op.type === type);
        document.getElementById('sectionTitle').textContent = type;
        
        // Update active state in sidebar
        document.querySelectorAll('[data-type]').forEach(item => {
            item.classList.toggle('active', item.dataset.type === type);
        });
        
        renderOperations(filtered);
    }
    
    function filterBySurgeon(surgeon) {
        const filtered = allOperations.filter(op => op.surgeon === surgeon);
        document.getElementById('sectionTitle').textContent = `Операции: ${surgeon}`;
        
        // Update active state
        document.querySelectorAll('[data-surgeon]').forEach(item => {
            item.classList.toggle('active', item.dataset.surgeon === surgeon);
        });
        
        renderOperations(filtered);
    }
    
    function applyQuickFilter(filter) {
        let filtered = allOperations;
        let title = 'Все операции';
        
        switch (filter) {
            case 'recent':
                filtered = [...allOperations].sort((a, b) => 
                    new Date(b.updatedAt || b.date) - new Date(a.updatedAt || a.date)
                ).slice(0, 10);
                title = 'Недавние операции';
                break;
            case 'favorites':
                // В MVP просто показываем первые 3
                filtered = allOperations.slice(0, 3);
                title = 'Избранное';
                break;
            case 'processing':
                filtered = allOperations.filter(op => op.status === 'processing' || op.status === 'pending');
                title = 'В обработке';
                break;
        }
        
        document.getElementById('sectionTitle').textContent = title;
        
        // Update active state
        document.querySelectorAll('[data-filter]').forEach(item => {
            item.classList.toggle('active', item.dataset.filter === filter);
        });
        
        renderOperations(filtered);
    }
    
    // Глобальная функция для открытия операции
    window.openOperation = function(id) {
        window.location.href = `viewer.html?id=${id}`;
    };
    
})();
