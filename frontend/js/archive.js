/**
 * NeuroVision Platform - Archive/Search
 * Страница расширенного поиска и архива
 */

(function() {
    'use strict';
    
    let tags = {};
    let searchResults = [];
    
    document.addEventListener('DOMContentLoaded', init);
    
    async function init() {
        try {
            await loadTags();
            setupEventListeners();
            performSearch(); // Начальная загрузка всех операций
        } catch (error) {
            console.error('Init error:', error);
        }
    }
    
    async function loadTags() {
        try {
            const response = await API.getTags();
            tags = response.data;
            
            // Заполнить фильтры
            populateSelect('filterPhases', tags.phases, 'name');
            populateSelect('filterInstruments', tags.instruments, 'name');
            populateSelect('filterAnatomy', tags.anatomy, 'name');
            populateSelect('filterEvents', tags.events, 'name');
            populateSelect('filterSurgeon', tags.surgeons, 'name', true);
            populateSelect('filterType', tags.operationTypes, 'name', true);
            
        } catch (error) {
            console.error('Tags error:', error);
        }
    }
    
    function populateSelect(selectId, items, key, addEmpty = false) {
        const select = document.getElementById(selectId);
        if (!select || !items) return;
        
        if (!addEmpty) {
            select.innerHTML = '';
        }
        
        items.forEach(item => {
            const option = document.createElement('option');
            option.value = item[key];
            option.textContent = item[key];
            select.appendChild(option);
        });
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
        
        // Глобальный поиск
        const searchBtn = document.getElementById('searchBtn');
        const searchInput = document.getElementById('globalSearch');
        
        if (searchBtn && searchInput) {
            searchBtn.addEventListener('click', performSearch);
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') performSearch();
            });
        }
        
        // Применить фильтры
        const applyBtn = document.getElementById('applyFiltersBtn');
        if (applyBtn) {
            applyBtn.addEventListener('click', performSearch);
        }
        
        // Сбросить фильтры
        const resetBtn = document.getElementById('resetFiltersBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', resetFilters);
        }
        
        const clearSearchBtn = document.getElementById('clearSearchBtn');
        if (clearSearchBtn) {
            clearSearchBtn.addEventListener('click', resetFilters);
        }
        
        // Ползунок уверенности
        const confidenceSlider = document.getElementById('filterConfidence');
        const confidenceValue = document.getElementById('confidenceValue');
        if (confidenceSlider && confidenceValue) {
            confidenceSlider.addEventListener('input', (e) => {
                confidenceValue.textContent = e.target.value + '%';
            });
        }
        
        // Быстрые фильтры в sidebar
        document.querySelectorAll('[data-quick]').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                applyQuickFilter(e.currentTarget.dataset.quick);
            });
        });
    }
    
    async function performSearch() {
        const resultsContainer = document.getElementById('searchResults');
        const emptyState = document.getElementById('emptyState');
        const resultsCount = document.getElementById('resultsCount');
        
        // Показать загрузку
        resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
        emptyState.classList.add('hidden');
        
        try {
            // Собрать параметры поиска
            const params = {
                query: document.getElementById('globalSearch')?.value || '',
                phases: getSelectedValues('filterPhases'),
                instruments: getSelectedValues('filterInstruments'),
                anatomy: getSelectedValues('filterAnatomy'),
                events: getSelectedValues('filterEvents'),
                surgeons: getSelectedValue('filterSurgeon') ? [getSelectedValue('filterSurgeon')] : null,
                operationTypes: getSelectedValue('filterType') ? [getSelectedValue('filterType')] : null,
                dateFrom: document.getElementById('filterDateFrom')?.value || null,
                dateTo: document.getElementById('filterDateTo')?.value || null,
                minConfidence: (document.getElementById('filterConfidence')?.value || 50) / 100
            };
            
            // Удалить пустые параметры
            Object.keys(params).forEach(key => {
                if (!params[key] || (Array.isArray(params[key]) && params[key].length === 0)) {
                    delete params[key];
                }
            });
            
            // Выполнить поиск
            const response = await API.search(params);
            searchResults = response.data;
            
            // Обновить счётчик
            const totalSegments = response.pagination?.totalSegments || 0;
            const totalOps = response.pagination?.totalOperations || 0;
            resultsCount.textContent = `Найдено: ${totalSegments} сегментов в ${totalOps} операциях`;
            
            // Обновить активные фильтры
            updateActiveFilters(params);
            
            // Отобразить результаты
            renderResults(searchResults);
            
        } catch (error) {
            console.error('Search error:', error);
            resultsContainer.innerHTML = '<div class="empty-state"><p>Ошибка поиска</p></div>';
        }
    }
    
    function getSelectedValues(selectId) {
        const select = document.getElementById(selectId);
        if (!select) return [];
        
        return Array.from(select.selectedOptions).map(opt => opt.value);
    }
    
    function getSelectedValue(selectId) {
        const select = document.getElementById(selectId);
        return select?.value || '';
    }
    
    function updateActiveFilters(params) {
        const container = document.getElementById('activeFilters');
        const chipsContainer = document.getElementById('activeFilterChips');
        
        const chips = [];
        
        if (params.query) {
            chips.push({ label: `Поиск: "${params.query}"`, type: 'query' });
        }
        if (params.phases?.length) {
            params.phases.forEach(p => chips.push({ label: p, type: 'phase' }));
        }
        if (params.instruments?.length) {
            params.instruments.forEach(i => chips.push({ label: i, type: 'instrument' }));
        }
        if (params.anatomy?.length) {
            params.anatomy.forEach(a => chips.push({ label: a, type: 'anatomy' }));
        }
        if (params.events?.length) {
            params.events.forEach(e => chips.push({ label: e, type: 'event' }));
        }
        if (params.surgeons?.length) {
            chips.push({ label: params.surgeons[0], type: 'surgeon' });
        }
        if (params.operationTypes?.length) {
            chips.push({ label: params.operationTypes[0], type: 'type' });
        }
        
        if (chips.length === 0) {
            container.style.display = 'none';
            return;
        }
        
        container.style.display = 'flex';
        chipsContainer.innerHTML = chips.map(chip => `
            <span class="chip ${chip.type}">
                ${chip.label}
                <span class="remove" onclick="removeFilter('${chip.type}', '${chip.label}')">&times;</span>
            </span>
        `).join('');
    }
    
    function renderResults(results) {
        const container = document.getElementById('searchResults');
        const emptyState = document.getElementById('emptyState');
        
        if (!results || results.length === 0) {
            container.innerHTML = '';
            emptyState.classList.remove('hidden');
            return;
        }
        
        emptyState.classList.add('hidden');
        
        container.innerHTML = results.map(result => `
            <div class="result-group">
                <div class="result-group-header" onclick="window.location.href='viewer.html?id=${result.operation.id}'">
                    <div class="result-group-thumb">
                        <img src="${result.operation.thumbnailUrl || 'https://via.placeholder.com/120x68'}" 
                             alt="${result.operation.title}">
                    </div>
                    <div class="result-group-info">
                        <h3 class="result-group-title">${result.operation.title}</h3>
                        <div class="result-group-meta">
                            <span><i class="fas fa-user-md"></i> ${result.operation.surgeon}</span>
                            <span><i class="fas fa-calendar"></i> ${Utils.formatDateShort(result.operation.date)}</span>
                            <span><i class="fas fa-tag"></i> ${result.operation.type}</span>
                            <span><i class="fas fa-film"></i> ${result.segments.length} совпадений</span>
                        </div>
                    </div>
                </div>
                <div class="result-clips">
                    ${result.segments.slice(0, 6).map(seg => `
                        <div class="result-clip" onclick="openSegment('${result.operation.id}', ${seg.startTime})">
                            <div class="result-clip-thumb">
                                <img src="${seg.thumbnailUrl || result.operation.thumbnailUrl}" alt="${seg.name}">
                            </div>
                            <div class="result-clip-info">
                                <div class="result-clip-name">
                                    <span class="tag ${seg.type}" style="font-size: 9px; padding: 1px 4px;">${getTypeLabel(seg.type)}</span>
                                    ${seg.name}
                                </div>
                                <div class="result-clip-time">
                                    ${Utils.formatTime(seg.startTime)} • ${Utils.formatConfidence(seg.confidence)}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                    ${result.segments.length > 6 ? `
                        <div class="result-clip" style="justify-content: center; background: var(--bg-secondary);"
                             onclick="window.location.href='viewer.html?id=${result.operation.id}'">
                            <span class="text-muted">+${result.segments.length - 6} ещё...</span>
                        </div>
                    ` : ''}
                </div>
            </div>
        `).join('');
    }
    
    function getTypeLabel(type) {
        const labels = {
            'phase': 'Этап',
            'instrument': 'Инстр.',
            'anatomy': 'Анат.',
            'event': 'Событие'
        };
        return labels[type] || type;
    }
    
    function resetFilters() {
        // Сбросить все поля
        document.getElementById('globalSearch').value = '';
        document.getElementById('filterPhases').selectedIndex = -1;
        document.getElementById('filterInstruments').selectedIndex = -1;
        document.getElementById('filterAnatomy').selectedIndex = -1;
        document.getElementById('filterEvents').selectedIndex = -1;
        document.getElementById('filterSurgeon').value = '';
        document.getElementById('filterType').value = '';
        document.getElementById('filterDateFrom').value = '';
        document.getElementById('filterDateTo').value = '';
        document.getElementById('filterConfidence').value = 50;
        document.getElementById('confidenceValue').textContent = '50%';
        
        // Убрать активные фильтры
        document.getElementById('activeFilters').style.display = 'none';
        
        // Повторный поиск
        performSearch();
    }
    
    function applyQuickFilter(filter) {
        // Сбросить все фильтры
        resetFilters();
        
        switch (filter) {
            case 'bleeding':
                selectOption('filterEvents', 'Кровотечение');
                break;
            case 'clipping':
                selectOption('filterPhases', 'Клипирование аневризмы');
                break;
            case 'resection':
                selectOption('filterPhases', 'Резекция опухоли');
                break;
            case 'complications':
                // Выбрать несколько событий
                ['Кровотечение', 'Повреждение ткани', 'Разрыв аневризмы'].forEach(evt => {
                    selectOption('filterEvents', evt);
                });
                break;
        }
        
        performSearch();
    }
    
    function selectOption(selectId, value) {
        const select = document.getElementById(selectId);
        if (!select) return;
        
        Array.from(select.options).forEach(opt => {
            if (opt.value === value) {
                opt.selected = true;
            }
        });
    }
    
    // Глобальные функции
    window.openSegment = function(operationId, startTime) {
        window.location.href = `viewer.html?id=${operationId}&t=${startTime}`;
    };
    
    window.removeFilter = function(type, value) {
        // Для простоты просто выполним новый поиск
        // В реальном приложении нужно удалить конкретный фильтр
        performSearch();
    };
    
})();
