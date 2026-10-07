@echo off
setlocal
title OSEDiff API - PM2 stop
cd /d "%~dp0"

set PM2_APP=API-ImageEnhance

echo.
echo  Detener OSEDiff API en PM2
echo.

where pm2 >nul 2>&1
if errorlevel 1 (
    echo  PM2 no esta instalado.
    pause
    exit /b 1
)

call pm2 describe %PM2_APP% >nul 2>&1
if errorlevel 1 (
    echo  %PM2_APP% no esta registrado en PM2.
    pause
    exit /b 0
)

echo  Deteniendo y quitando %PM2_APP%...
call pm2 stop %PM2_APP%
call pm2 delete %PM2_APP%
call pm2 save >nul 2>&1

echo.
call pm2 status
echo.
pause
exit /b 0
