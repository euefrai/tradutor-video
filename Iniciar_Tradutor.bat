@echo off
chcp 65001 >nul
title Tradutor e Dublador Automatico de Videos IA
cd /d "%~dp0"

echo ============================================================================
echo    TRADUTOR E DUBLADOR AUTOMATICO DE VIDEOS COM IA (100%% LOCAL E GRATUITO)
echo    Inicializador Universal Zero-Dependencia (Python + FFmpeg + Whisper)
echo ============================================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_runtime.ps1" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] Ocorreu um problema durante a execucao do sistema (Codigo: %ERRORLEVEL%).
    echo Verifique as mensagens acima e tente novamente.
)
echo.
pause
