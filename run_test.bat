@echo off
chcp 65001 >nul
title HE THONG TRO LY AI GOLF (VAR GOLF)

echo =======================================================
echo     HE THONG TRO LY AI PHAN TICH TU THE GOLF (VAR GOLF)
echo =======================================================
echo.
echo [1] Chay voi WEBCAM may tinh (Nguon 0)
echo [2] Chay voi VIDEO CLIP mau (assets/videos/sample_swing.mp4)
echo [3] Chay kiem tra TOAN HOC VECTOR (Unit Tests)
echo [4] Cai dat tat ca thu vien (pip install -r requirements.txt)
echo.

set /p choice="Nhap lua chon cua ban (1, 2, 3 hoac 4): "

if "%choice%"=="1" (
    echo.
    echo Dang khoi dong voi Webcam...
    python run_app.py --source 0
) else if "%choice%"=="2" (
    echo.
    echo Dang khoi dong voi Video mau...
    python run_app.py --source assets/videos/sample_swing.mp4
) else if "%choice%"=="3" (
    echo.
    echo Dang chay kiem tra toan hoc...
    python tests/test_geometry.py
) else if "%choice%"=="4" (
    echo.
    echo Dang cai dat thu vien...
    pip install -r requirements.txt
) else (
    echo Lua chon khong hop le!
)

echo.
pause
