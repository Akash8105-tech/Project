@echo off
title CarGrasp AI Vehicle Recognition System
echo ====================================================================
echo        STARTING CARGRASP AI VEHICLE RECOGNITION SYSTEM
echo ====================================================================
echo.
echo [1/2] Starting Local AI Neural Vision Server (Port 8080)...
start /b python server.py > server.log 2>&1
timeout /t 2 /nobreak >nul
echo [2/2] Opening CarGrasp in Default Browser...
start http://localhost:8080
echo.
echo Server is running live on http://localhost:8080
echo Press Ctrl+C in this window or close it to stop the server.
echo ====================================================================
pause
