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

git remote set-url origin https://github.com/Shahidfateen/WeatherGPT.git 2>nul || git remote add origin https://github.com/Shahidfateen/WeatherGPT.git

echo Adding files to git...
git add .
git commit -m "Configure WeatherGPT production deployment for Render and Google Cloud Run" 2>nul

echo Pushing to GitHub...
git push -u origin main

echo.
echo ================================================================
echo Repository is pushed to https://github.com/Shahidfateen/WeatherGPT
echo.
echo Next step:
echo 1. Open Render Dashboard: https://dashboard.render.com
echo 2. Click 'New +' -^> 'Web Service'
echo 3. Connect your GitHub repo 'Shahidfateen/WeatherGPT'
echo 4. Render will automatically detect render.yaml and deploy!
echo 5. You will get your permanent SIH link:
echo    https://weathergpt-sih.onrender.com
echo ================================================================
pause
