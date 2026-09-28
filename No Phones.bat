@echo off
REM Double-click to start No Phones. Press q in the preview window to stop.
title No Phones
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No .venv found. See README.md for setup.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" src\monitor.py %*
if errorlevel 1 (
    echo.
    echo Something went wrong - see the message above.
    pause
) else (
    timeout /t 15
)
