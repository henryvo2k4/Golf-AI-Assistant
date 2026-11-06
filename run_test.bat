@echo off
chcp 65001 >nul
title HE THONG TRO LY AI GOLF (VAR GOLF)

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [!] Khong tim thay Python trong .venv!
    echo Dang thu tim python mac dinh tren he thong...
    set "PYTHON_EXE=python"
)

echo =======================================================
echo     HE THONG TRO LY AI PHAN TICH TU THE GOLF (VAR GOLF)
echo     Che do: Hien thi Truc quan 100%% tren Man hinh
echo =======================================================
echo.
echo [1] Chay voi WEBCAM may tinh (Nguon 0)
echo [2] Chay kiem tra TOAN HOC VECTOR (Unit Tests)
echo [3] Cai dat / Cap nhat lai thu vien (pip install)
echo.

set /p choice="Nhap lua chon cua ban (1, 2 hoac 3): "

if "%choice%"=="1" (
    echo.
    echo Dang khoi dong voi Webcam may tinh...
    "%PYTHON_EXE%" "%~dp0run_app.py" --source 0
) else if "%choice%"=="2" (
    echo.
    echo Dang chay kiem tra toan hoc Vector...
    "%PYTHON_EXE%" "%~dp0tests\test_geometry.py"
) else if "%choice%"=="3" (
    echo.
    echo Dang cap nhat thu vien...
    "%~dp0.venv\Scripts\pip.exe" install -r "%~dp0requirements.txt"
) else (
    echo Lua chon khong hop le!
)

echo.
pause
