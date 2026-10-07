@echo off
setlocal EnableDelayedExpansion
title OSEDiff API
cd /d "%~dp0"
set OSEDIFF_HOST=0.0.0.0
set OSEDIFF_PORT=8010

echo.
echo  OSEDiff API (interno, sin auth)
echo  Swagger: http://127.0.0.1:%OSEDIFF_PORT%/docs
echo  Health:  http://127.0.0.1:%OSEDIFF_PORT%/health
echo.

call :free_port %OSEDIFF_PORT%

echo  La consola permanece abierta mientras el servidor corre.
echo  Ctrl+C para detener.
echo.

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo  ERROR: no existe .venv
    echo  Crea el entorno e instala dependencias:
    echo    python -m venv .venv
    echo    .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo  Python: %PY%
echo.
"%PY%" api.py
set EXITCODE=!ERRORLEVEL!

echo.
if !EXITCODE! NEQ 0 (
    echo El servidor termino con error ^(codigo !EXITCODE!^).
    echo Revisa: CUDA, rutas en preset/models/, logs arriba.
) else (
    echo El servidor se detuvo.
)
echo.
pause
exit /b !EXITCODE!

:free_port
set "PORT=%~1"
echo  Comprobando puerto %PORT%...
powershell -NoProfile -Command ^
  "$port=[int]'%PORT%'; $pids = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; if (-not $pids) { Write-Host '  Puerto libre.'; exit 0 }; foreach ($procId in $pids) { try { $proc = Get-Process -Id $procId -ErrorAction Stop; Write-Host ('  Cerrando {0} (PID {1})...' -f $proc.ProcessName, $procId); Stop-Process -Id $procId -Force -ErrorAction Stop; Write-Host '  OK.' } catch { Write-Host ('  No se pudo cerrar PID {0}: {1}' -f $procId, $_.Exception.Message); exit 1 } }"
if errorlevel 1 (
    echo.
    echo  ADVERTENCIA: no se libero el puerto %PORT%. Cierra el proceso manualmente o usa otro puerto:
    echo  set OSEDIFF_PORT=8011
    echo.
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
exit /b 0
