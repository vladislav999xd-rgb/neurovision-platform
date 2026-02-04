/**
 * NeuroVision Platform - Auth Module
 * Модуль авторизации (заглушка для MVP)
 */

(function() {
    'use strict';
    
    // Проверка авторизации на всех страницах кроме login
    const isLoginPage = window.location.pathname.includes('index.html') || 
                       window.location.pathname.endsWith('/');
    
    if (!isLoginPage) {
        const user = localStorage.getItem('neurovision_user');
        if (!user) {
            window.location.href = 'index.html';
        }
    }
    
    // Обновление UI с данными пользователя
    document.addEventListener('DOMContentLoaded', function() {
        const user = Utils.getCurrentUser();
        
        if (user) {
            // Обновить аватар
            const avatars = document.querySelectorAll('.user-avatar');
            avatars.forEach(avatar => {
                const initials = user.name
                    .split(' ')
                    .map(n => n[0])
                    .join('')
                    .toUpperCase()
                    .slice(0, 2);
                avatar.textContent = initials;
                avatar.title = user.name;
            });
        }
        
        // Sidebar toggle
        const menuBtn = document.getElementById('menuBtn');
        const sidebar = document.getElementById('sidebar');
        
        if (menuBtn && sidebar) {
            menuBtn.addEventListener('click', function() {
                sidebar.classList.toggle('open');
            });
        }
    });
})();
