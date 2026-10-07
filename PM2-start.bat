@echo off
setlocal EnableDelayedExpansion
title OSEDiff API - PM2 start
cd /d "%~dp0"

REM PM2-start-all.bat pasa "nopause" para no bloquear el arranque del resto.
set "NO_PAUSE="
if /I "%~1"=="nopause" set "NO_PAUSE=1"

set OSEDIFF_HOST=0.0.0.0
set OSEDIFF_PORT=8010
set PM2_APP=API-ImageEnhance
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

echo.
echo  OSEDiff API via PM2
echo  Swagger: http://127.0.0.1:%OSEDIFF_PORT%/docs
echo  Health:  http://127.0.0.1:%OSEDIFF_PORT%/health
echo.

where pm2 >nul 2>&1
if errorlevel 1 (
    echo  PM2 no esta instalado. Instala con:
    echo    npm install -g pm2
    echo.
    if not defined NO_PAUSE pause
    exit /b 1
)

if not exist "%VENV_PY%" (
    echo  Falta .venv — ejecuta setup-venv-gpu.bat o:
    echo    python -m venv .venv
    echo    setup-venv-gpu.bat
    echo.
    if not defined NO_PAUSE pause
    exit /b 1
)

echo  Python: %VENV_PY%
"%VENV_PY%" -c "import torch; assert torch.cuda.is_available(), 'PyTorch sin CUDA en .venv'; print('  CUDA OK:', torch.cuda.get_device_name(0))"
if errorlevel 1 (
    echo.
    echo  Instala PyTorch con GPU: setup-venv-gpu.bat
    echo.
    if not defined NO_PAUSE pause
    exit /b 1
)
echo.

call pm2 describe %PM2_APP% >nul 2>&1
if errorlevel 1 (
    call :free_port %OSEDIFF_PORT%
    echo  Iniciando %PM2_APP%...
    call pm2 start ecosystem.config.cjs
) else (
    echo  Reiniciando %PM2_APP%...
    call pm2 restart ecosystem.config.cjs --update-env
)

if errorlevel 1 (
    echo.
    echo  PM2 termino con error ^(codigo !ERRORLEVEL!^).
    echo  Revisa: pm2 logs %PM2_APP%
    echo.
    if not defined NO_PAUSE pause
    exit /b 1
)

call pm2 save >nul 2>&1
echo.
call pm2 status %PM2_APP%
echo.
echo  Logs: pm2 logs %PM2_APP%
echo.
if not defined NO_PAUSE pause
goto :eof

:free_port
set "PORT=%~1"
echo  Comprobando puerto %PORT%...
powershell -NoProfile -Command ^
  "$port=[int]'%PORT%'; $pids = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; if (-not $pids) { Write-Host '  Puerto libre.'; exit 0 }; foreach ($procId in $pids) { try { $proc = Get-Process -Id $procId -ErrorAction Stop; Write-Host ('  Cerrando {0} (PID {1})...' -f $proc.ProcessName, $procId); Stop-Process -Id $procId -Force -ErrorAction Stop; Write-Host '  OK.' } catch { Write-Host ('  No se pudo cerrar PID {0}' -f $procId); exit 1 } }"
if errorlevel 1 (
    echo  ADVERTENCIA: puerto %PORT% ocupado. Cierra el proceso o cambia OSEDIFF_PORT en ecosystem.config.cjs
    if not defined NO_PAUSE pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
exit /b 0
