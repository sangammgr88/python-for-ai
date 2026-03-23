@echo off
echo =============================================
echo   HEAD DETECTOR v3 - WebSocket Server
echo =============================================
echo.

if not exist venv (
    echo [1/3] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/3] Virtual environment already exists, skipping...
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate

echo [3/3] Installing packages...
pip install -r requirements.txt

echo.
echo =============================================
echo   Starting WebSocket server on port 8000...
echo   Keep this window open while exam is running
echo =============================================
echo.

python head_detector.py

pause
