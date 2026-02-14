/**
 * NeuroVision Platform - Archive/Search (Redesigned)
 * Современная страница расширенного поиска и архива
 */

(function() {
    'use strict';
    
    let tags = {};
    let searchResults = [];
    let viewMode = 'grid'; // 'grid' | 'list'
    let filtersOpen = false;
    
    // Состояние чип-фильтров
    let selectedFilters = {
        phases: [],
        instruments: [],
        anatomy: [],
        events: []
    };
    
    document.addEventListener('DOMContentLoaded', init);
    
    async function init() {
        try {
            await loadTags();
            setupEventListeners();
            await performSearch();
        } catch (error) {
            console.error('Init error:', error);
        }
    }
    
    async function loadTags() {
        try {
            const response = await API.getTags();
            tags = response.data;
            
            // Заполнить чип-селекторы
            populateChips('filterPhases', tags.phases, 'name', 'phases');
            populateChips('filterInstruments', tags.instruments, 'name', 'instruments');
            populateChips('filterAnatomy', tags.anatomy, 'name', 'anatomy');
            populateChips('filterEvents', tags.events, 'name', 'events');
            
            // Заполнить select-ы
            populateSelect('filterSurgeon', tags.surgeons, 'name');
            populateSelect('filterType', tags.operationTypes, 'name');
            
        } catch (error) {
            console.error('Tags error:', error);
        }
    }
    
    function populateChips(containerId, items, key, category) {
        const container = document.getElementById(containerId);
        if (!container || !items) return;
        
        container.innerHTML = items.map(item => `
            <div class="chip-option" data-value="${item[key]}">
                <span class="chip-check"><i class="fas fa-check"></i></span>
                ${item[key]}
            </div>
        `).join('');
        
        // Обработчики кликов
        container.querySelectorAll('.chip-option').forEach(chip => {
            chip.addEventListener('click', () => {
                chip.classList.toggle('selected');
                const value = chip.dataset.value;
                
                if (chip.classList.contains('selected')) {
                    selectedFilters[category].push(value);
                } else {
                    selectedFilters[category] = selectedFilters[category].filter(v => v !== value);
                }
                
                updateFilterCountBadge();
                updateActiveFiltersBar();
            });
        });
    }
    
    function populateSelect(selectId, items, key) {
        const select = document.getElementById(selectId);
        if (!select || !items) return;
        
        items.forEach(item => {
            const option = document.createElement('option');
            option.value = item[key];
            option.textContent = item[key];
            select.appendChild(option);
        });
    }
    
    function setupEventListeners() {
        // Toggle Sidebar
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
        
        // Search
        const searchBtn = document.getElementById('searchBtn');
        const searchInput = document.getElementById('globalSearch');
        
        if (searchBtn) {
            searchBtn.addEventListener('click', performSearch);
        }
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') performSearch();
            });
        }
        
        // Filters toggle
        const filtersToggle = document.getElementById('filtersToggleBtn');
        if (filtersToggle) {
            filtersToggle.addEventListener('click', toggleFilters);
        }
        
        // Apply filters
        const applyBtn = document.getElementById('applyFiltersBtn');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => {
                performSearch();
                if (filtersOpen) toggleFilters();
            });
        }
        
        // Reset filters
        const resetBtn = document.getElementById('resetFiltersBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', resetFilters);
        }
        
        const clearSearchBtn = document.getElementById('clearSearchBtn');
        if (clearSearchBtn) {
            clearSearchBtn.addEventListener('click', () => {
                resetFilters();
                performSearch();
            });
        }
        
        const clearAllBtn = document.getElementById('clearAllFilters');
        if (clearAllBtn) {
            clearAllBtn.addEventListener('click', () => {
                resetFilters();
                performSearch();
            });
        }
        
        // Confidence slider
        const confidenceSlider = document.getElementById('filterConfidence');
        const confidenceValue = document.getElementById('confidenceValue');
        if (confidenceSlider && confidenceValue) {
            confidenceSlider.addEventListener('input', (e) => {
                confidenceValue.textContent = e.target.value + '%';
            });
        }
        
        // Sort
        const sortSelect = document.getElementById('sortSelect');
        if (sortSelect) {
            sortSelect.addEventListener('change', () => {
                renderCurrentResults();
            });
        }
        
        // View toggle
        document.querySelectorAll('.view-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.view-toggle-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                viewMode = btn.dataset.view;
                renderCurrentResults();
            });
        });
        
        // Быстрые фильтры
        document.querySelectorAll('[data-quick]').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                applyQuickFilter(e.currentTarget.dataset.quick);
            });
        });
    }
    
    function toggleFilters() {
        const panel = document.getElementById('filtersPanel');
        const btn = document.getElementById('filtersToggleBtn');
        filtersOpen = !filtersOpen;
        
        panel.classList.toggle('open', filtersOpen);
        btn.classList.toggle('active', filtersOpen);
    }
    
    function updateFilterCountBadge() {
        const badge = document.getElementById('filterCountBadge');
        const total = Object.values(selectedFilters).reduce((sum, arr) => sum + arr.length, 0);
        
        const surgeon = document.getElementById('filterSurgeon')?.value;
        const type = document.getElementById('filterType')?.value;
        const count = total + (surgeon ? 1 : 0) + (type ? 1 : 0);
        
        if (count > 0) {
            badge.textContent = count;
            badge.classList.remove('hidden');
        } else {
            badge.classList.add('hidden');
        }
    }
    
    function updateActiveFiltersBar() {
        const bar = document.getElementById('activeFiltersBar');
        const chipsContainer = document.getElementById('activeFilterChips');
        
        const chips = [];
        const query = document.getElementById('globalSearch')?.value?.trim();
        
        if (query) {
            chips.push({ label: '\u00ab' + query + '\u00bb', type: 'query', category: 'query', value: query });
        }
        
        selectedFilters.phases.forEach(p => chips.push({ label: p, type: 'phase', category: 'phases', value: p }));
        selectedFilters.instruments.forEach(i => chips.push({ label: i, type: 'instrument', category: 'instruments', value: i }));
        selectedFilters.anatomy.forEach(a => chips.push({ label: a, type: 'anatomy', category: 'anatomy', value: a }));
        selectedFilters.events.forEach(e => chips.push({ label: e, type: 'event', category: 'events', value: e }));
        
        const surgeon = document.getElementById('filterSurgeon')?.value;
        if (surgeon) chips.push({ label: surgeon, type: 'surgeon', category: 'surgeon', value: surgeon });
        
        const opType = document.getElementById('filterType')?.value;
        if (opType) chips.push({ label: opType, type: 'type', category: 'type', value: opType });
        
        if (chips.length === 0) {
            bar.classList.add('hidden');
            return;
        }
        
        bar.classList.remove('hidden');
        chipsContainer.innerHTML = chips.map((chip, idx) => `
            <span class="active-chip ${chip.type}">
                ${escapeHtml(chip.label)}
                <span class="remove-chip" data-idx="${idx}" data-category="${chip.category}" data-value="${escapeAttr(chip.value)}">
                    <i class="fas fa-times"></i>
                </span>
            </span>
        `).join('');
        
        // Обработчики удаления
        chipsContainer.querySelectorAll('.remove-chip').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const cat = btn.dataset.category;
                const val = btn.dataset.value;
                
                if (cat === 'query') {
                    document.getElementById('globalSearch').value = '';
                } else if (cat === 'surgeon') {
                    document.getElementById('filterSurgeon').value = '';
                } else if (cat === 'type') {
                    document.getElementById('filterType').value = '';
                } else if (selectedFilters[cat]) {
                    selectedFilters[cat] = selectedFilters[cat].filter(v => v !== val);
                    const chipEl = document.querySelector(`#filter${capitalize(cat)} .chip-option[data-value="${val}"]`);
                    if (chipEl) chipEl.classList.remove('selected');
                }
                
                updateFilterCountBadge();
                updateActiveFiltersBar();
                performSearch();
            });
        });
    }
    
    async function performSearch() {
        const resultsContainer = document.getElementById('searchResults');
        const emptyState = document.getElementById('emptyState');
        const resultsCount = document.getElementById('resultsCount');
        
        // Скелетоны загрузки
        resultsContainer.className = 'results-grid';
        resultsContainer.innerHTML = `
            <div class="skeleton-card"><div class="skeleton-thumb"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line medium"></div><div class="skeleton-line short"></div></div></div>
            <div class="skeleton-card"><div class="skeleton-thumb"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line medium"></div><div class="skeleton-line short"></div></div></div>
            <div class="skeleton-card"><div class="skeleton-thumb"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line medium"></div><div class="skeleton-line short"></div></div></div>
        `;
        emptyState.classList.add('hidden');
        
        try {
            const params = {
                query: document.getElementById('globalSearch')?.value || '',
                phases: selectedFilters.phases.length > 0 ? selectedFilters.phases : null,
                instruments: selectedFilters.instruments.length > 0 ? selectedFilters.instruments : null,
                anatomy: selectedFilters.anatomy.length > 0 ? selectedFilters.anatomy : null,
                events: selectedFilters.events.length > 0 ? selectedFilters.events : null,
                surgeons: getSelectedValue('filterSurgeon') ? [getSelectedValue('filterSurgeon')] : null,
                operationTypes: getSelectedValue('filterType') ? [getSelectedValue('filterType')] : null,
                dateFrom: document.getElementById('filterDateFrom')?.value || null,
                dateTo: document.getElementById('filterDateTo')?.value || null,
                minConfidence: (document.getElementById('filterConfidence')?.value || 50) / 100
            };
            
            Object.keys(params).forEach(key => {
                if (!params[key] || (Array.isArray(params[key]) && params[key].length === 0)) {
                    delete params[key];
                }
            });
            
            const response = await API.search(params);
            searchResults = response.data || [];
            
            updateStats(response);
            updateActiveFiltersBar();
            
            const totalSegments = response.pagination?.totalSegments || 0;
            const totalOps = response.pagination?.totalOperations || 0;
            resultsCount.innerHTML = 'Найдено: <strong>' + totalOps + '</strong> операций, <strong>' + totalSegments + '</strong> сегментов';
            
            renderCurrentResults();
            
        } catch (error) {
            console.error('Search error:', error);
            resultsContainer.innerHTML = '';
            resultsCount.textContent = 'Ошибка поиска';
        }
    }
    
    function updateStats(response) {
        const totalOps = response.pagination?.totalOperations || searchResults.length;
        let totalSegments = 0;
        let totalDetections = 0;
        const surgeonSet = new Set();
        
        searchResults.forEach(r => {
            totalSegments += r.segments?.length || 0;
            surgeonSet.add(r.operation?.surgeon);
            const detections = r.segments?.filter(s => s.type === 'instrument' || s.type === 'anatomy') || [];
            totalDetections += detections.length;
        });
        
        document.getElementById('statOps').textContent = totalOps;
        document.getElementById('statSegments').textContent = totalSegments;
        document.getElementById('statSurgeons').textContent = surgeonSet.size;
        document.getElementById('statDetections').textContent = totalDetections;
    }
    
    function renderCurrentResults() {
        if (viewMode === 'grid') {
            renderGridView(sortResults(searchResults));
        } else {
            renderListView(sortResults(searchResults));
        }
    }
    
    function sortResults(results) {
        const sortValue = document.getElementById('sortSelect')?.value || 'date-desc';
        const sorted = [...results];
        
        sorted.sort((a, b) => {
            const opA = a.operation;
            const opB = b.operation;
            
            switch (sortValue) {
                case 'date-desc':
                    return new Date(opB.date) - new Date(opA.date);
                case 'date-asc':
                    return new Date(opA.date) - new Date(opB.date);
                case 'title-asc':
                    return opA.title.localeCompare(opB.title, 'ru');
                case 'duration-desc':
                    return (opB.duration || 0) - (opA.duration || 0);
                default:
                    return 0;
            }
        });
        
        return sorted;
    }
    
    function renderGridView(results) {
        const container = document.getElementById('searchResults');
        const emptyState = document.getElementById('emptyState');
        
        container.className = 'results-grid';
        
        if (!results || results.length === 0) {
            container.innerHTML = '';
            emptyState.classList.remove('hidden');
            return;
        }
        
        emptyState.classList.add('hidden');
        
        container.innerHTML = results.map(result => {
            const op = result.operation;
            const segments = result.segments || [];
            const phases = segments.filter(s => s.type === 'phase');
            const instruments = segments.filter(s => s.type === 'instrument');
            const anatomy = segments.filter(s => s.type === 'anatomy');
            const events = segments.filter(s => s.type === 'event');
            
            let tagsHtml = '';
            const uniquePhases = [...new Set(phases.map(p => p.name))].slice(0, 2);
            const uniqueInstr = [...new Set(instruments.map(i => i.name))].slice(0, 2);
            const uniqueAnat = [...new Set(anatomy.map(a => a.name))].slice(0, 1);
            const uniqueEvents = [...new Set(events.map(e => e.name))].slice(0, 1);
            
            uniquePhases.forEach(n => { tagsHtml += '<span class="result-card-tag phase">' + escapeHtml(n) + '</span>'; });
            uniqueInstr.forEach(n => { tagsHtml += '<span class="result-card-tag instrument">' + escapeHtml(n) + '</span>'; });
            uniqueAnat.forEach(n => { tagsHtml += '<span class="result-card-tag anatomy">' + escapeHtml(n) + '</span>'; });
            uniqueEvents.forEach(n => { tagsHtml += '<span class="result-card-tag event">' + escapeHtml(n) + '</span>'; });
            
            const statusClass = op.status === 'completed' ? 'completed' : 'pending';
            const statusLabel = op.status === 'completed' ? 'Обработано' : 'Ожидает';
            
            return '<div class="result-card" onclick="window.location.href=\'viewer.html?id=' + op.id + '\'">' +
                '<div class="result-card-thumb">' +
                    '<img src="' + (op.thumbnailUrl || 'images/placeholder.svg') + '" alt="' + escapeAttr(op.title) + '" onerror="this.style.display=\'none\'">' +
                    '<div class="result-card-thumb-overlay">' +
                        '<span class="result-card-duration"><i class="fas fa-clock"></i> ' + Utils.formatDuration(op.duration) + '</span>' +
                        '<span class="result-card-status ' + statusClass + '">' + statusLabel + '</span>' +
                    '</div>' +
                '</div>' +
                '<div class="result-card-body">' +
                    '<div class="result-card-title">' + escapeHtml(op.title) + '</div>' +
                    '<div class="result-card-meta">' +
                        '<span class="result-card-meta-item"><i class="fas fa-user-md"></i> ' + escapeHtml(op.surgeon) + '</span>' +
                        '<span class="result-card-meta-item"><i class="fas fa-calendar"></i> ' + Utils.formatDateShort(op.date) + '</span>' +
                        '<span class="result-card-meta-item"><i class="fas fa-tag"></i> ' + escapeHtml(op.type) + '</span>' +
                    '</div>' +
                    '<div class="result-card-tags">' + tagsHtml + '</div>' +
                '</div>' +
                '<div class="result-card-footer">' +
                    '<span class="result-card-matches"><i class="fas fa-crosshairs"></i> ' + segments.length + ' совпадений</span>' +
                    '<span class="result-card-action">Смотреть <i class="fas fa-arrow-right"></i></span>' +
                '</div>' +
            '</div>';
        }).join('');
    }
    
    function renderListView(results) {
        const container = document.getElementById('searchResults');
        const emptyState = document.getElementById('emptyState');
        
        container.className = 'results-list';
        
        if (!results || results.length === 0) {
            container.innerHTML = '';
            emptyState.classList.remove('hidden');
            return;
        }
        
        emptyState.classList.add('hidden');
        
        container.innerHTML = results.map(result => {
            const op = result.operation;
            const segments = result.segments || [];
            const phases = [...new Set(segments.filter(s => s.type === 'phase').map(p => p.name))].slice(0, 3);
            
            let tagsHtml = phases.map(n => '<span class="result-card-tag phase">' + escapeHtml(n) + '</span>').join('');
            
            const statusClass = op.status === 'completed' ? 'completed' : 'pending';
            const statusLabel = op.status === 'completed' ? 'Обработано' : 'Ожидает';
            
            return '<div class="result-list-item" onclick="window.location.href=\'viewer.html?id=' + op.id + '\'">' +
                '<div class="result-list-thumb">' +
                    '<img src="' + (op.thumbnailUrl || 'images/placeholder.svg') + '" alt="' + escapeAttr(op.title) + '" onerror="this.style.display=\'none\'">' +
                '</div>' +
                '<div class="result-list-info">' +
                    '<div class="result-list-title">' + escapeHtml(op.title) + '</div>' +
                    '<div class="result-list-meta">' +
                        '<span><i class="fas fa-user-md"></i> ' + escapeHtml(op.surgeon) + '</span>' +
                        '<span><i class="fas fa-calendar"></i> ' + Utils.formatDateShort(op.date) + '</span>' +
                        '<span><i class="fas fa-clock"></i> ' + Utils.formatDuration(op.duration) + '</span>' +
                        '<span><i class="fas fa-tag"></i> ' + escapeHtml(op.type) + '</span>' +
                    '</div>' +
                    '<div class="result-list-tags">' + tagsHtml + '</div>' +
                '</div>' +
                '<div class="result-list-actions">' +
                    '<span class="result-card-status ' + statusClass + '">' + statusLabel + '</span>' +
                    '<span class="result-card-matches"><i class="fas fa-crosshairs"></i> ' + segments.length + '</span>' +
                '</div>' +
            '</div>';
        }).join('');
    }
    
    function getSelectedValue(selectId) {
        const select = document.getElementById(selectId);
        return select?.value || '';
    }
    
    function resetFilters() {
        document.getElementById('globalSearch').value = '';
        
        Object.keys(selectedFilters).forEach(key => {
            selectedFilters[key] = [];
        });
        document.querySelectorAll('.chip-option.selected').forEach(chip => {
            chip.classList.remove('selected');
        });
        
        document.getElementById('filterSurgeon').value = '';
        document.getElementById('filterType').value = '';
        document.getElementById('filterDateFrom').value = '';
        document.getElementById('filterDateTo').value = '';
        document.getElementById('filterConfidence').value = 50;
        document.getElementById('confidenceValue').textContent = '50%';
        
        updateFilterCountBadge();
        updateActiveFiltersBar();
    }
    
    function applyQuickFilter(filter) {
        resetFilters();
        
        switch (filter) {
            case 'all':
                break;
            case 'bleeding':
                selectChip('filterEvents', 'Кровотечение');
                break;
            case 'clipping':
                selectChip('filterPhases', 'Клипирование аневризмы');
                break;
            case 'resection':
                selectChip('filterPhases', 'Резекция опухоли');
                break;
            case 'complications':
                ['Кровотечение', 'Повреждение ткани', 'Разрыв аневризмы'].forEach(evt => {
                    selectChip('filterEvents', evt);
                });
                break;
        }
        
        updateFilterCountBadge();
        performSearch();
    }
    
    function selectChip(containerId, value) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const chip = container.querySelector('.chip-option[data-value="' + value + '"]');
        if (chip && !chip.classList.contains('selected')) {
            chip.classList.add('selected');
            const category = containerId.replace('filter', '').charAt(0).toLowerCase() + containerId.replace('filter', '').slice(1);
            // Map container ID to selectedFilters key
            const catMap = { 'Phases': 'phases', 'Instruments': 'instruments', 'Anatomy': 'anatomy', 'Events': 'events' };
            const catKey = catMap[containerId.replace('filter', '')];
            if (catKey && selectedFilters[catKey]) {
                selectedFilters[catKey].push(value);
            }
        }
    }
    
    // Утилиты
    function escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    
    function escapeAttr(str) {
        if (!str) return '';
        return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;');
    }
    
    function capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
    
    // Глобальные функции
    window.openSegment = function(operationId, startTime) {
        window.location.href = 'viewer.html?id=' + operationId + '&t=' + startTime;
    };
    
})();
