@echo off
title WeatherGPT - Push to GitHub & Deploy to Render
echo ================================================================
echo        WeatherGPT: Git & Render Deployment Helper
echo ================================================================
echo.
echo Step 1: Checking Git status...
git status >nul 2>nul
if %errorlevel% neq 0 (
    echo Initializing Git repository...
    git init
    git branch -M main
)

echo Adding files to git...
git add .
git commit -m "Configure WeatherGPT production deployment for Render and Google Cloud Run"

echo.
echo ================================================================
echo Repository is committed and ready!
echo.
echo To link to your GitHub and trigger automatic Render deployment:
echo 1. Create a repository on GitHub (e.g. WeatherGPT)
echo 2. Run:
echo    git remote add origin https://github.com/YOUR_USERNAME/WeatherGPT.git
echo    git push -u origin main
echo.
echo 3. In Render Dashboard (https://dashboard.render.com):
echo    - Click 'New +' -^> 'Web Service'
echo    - Connect your GitHub repo 'WeatherGPT'
echo    - Render will automatically detect render.yaml and deploy!
echo    - You will get an instant public URL:
echo      https://weathergpt-xxxx.onrender.com
echo ================================================================
pause
