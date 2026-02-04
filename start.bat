@echo off
chcp 65001 >nul
title NeuroVision Platform - Запуск

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                                                                ║
echo ║     ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗               ║
echo ║     ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗              ║
echo ║     ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║              ║
echo ║     ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║              ║
echo ║     ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝              ║
echo ║     ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝               ║
echo ║                                                                ║
echo ║     VISION PLATFORM                                            ║
echo ║     Интеллектуальная видеоплатформа для анализа               ║
echo ║     нейрохирургических операций                                ║
echo ║                                                                ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

:: Проверяем наличие Node.js
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ОШИБКА] Node.js не найден!
    echo.
    echo Пожалуйста, установите Node.js с https://nodejs.org/
    echo Рекомендуемая версия: 18.x LTS или выше
    echo.
    pause
    exit /b 1
)

:: Показываем версию Node.js
for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo [INFO] Node.js версия: %NODE_VERSION%

:: Переходим в директорию backend
cd /d "%~dp0backend"

:: Проверяем наличие node_modules
if not exist "node_modules" (
    echo.
    echo [INFO] Установка зависимостей...
    echo.
    call npm install
    if %ERRORLEVEL% NEQ 0 (
        echo [ОШИБКА] Не удалось установить зависимости
        pause
        exit /b 1
    )
    echo.
    echo [OK] Зависимости установлены
)

:: Создаём директорию для загрузок если её нет
if not exist "uploads" (
    mkdir uploads
    echo [OK] Создана директория uploads
)

echo.
echo ════════════════════════════════════════════════════════════════
echo.
echo [INFO] Запуск сервера...
echo.
echo   Backend API:    http://localhost:3000
echo   Frontend:       http://localhost:3000/index.html
echo.
echo   Учётные данные для входа:
echo   Email:    doctor@neurovision.ru
echo   Пароль:   demo123
echo.
echo ════════════════════════════════════════════════════════════════
echo.
echo [INFO] Нажмите Ctrl+C для остановки сервера
echo.

:: Запускаем браузер через 3 секунды
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:3000"

:: Запускаем сервер
node server.js

:: Если сервер остановлен
echo.
echo [INFO] Сервер остановлен
pause
