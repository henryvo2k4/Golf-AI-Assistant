@echo off
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0run_app.py"
) else (
    echo [!] Khong tim thay moi truong ao .venv! Hay cai dat thu vien truoc.
    pause
)
