@echo off
title WeatherGPT - Live Prototype
echo ==========================================================
echo Starting WeatherGPT Web Application Prototype...
echo Real-Time Weather ^& 7-Day Predictions with Multilingual Voice
echo ==========================================================
cd /d "%~dp0weathergpt-backend"
echo Opening browser at http://127.0.0.1:8000/ ...
start "" "http://127.0.0.1:8000/"
py -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
