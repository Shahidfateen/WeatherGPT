@echo off
title WeatherGPT - Deploy to Google Cloud Run
echo ================================================================
echo           Deploying WeatherGPT to Google Cloud Run
echo ================================================================
echo Checking gcloud CLI installation...
where gcloud >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] Google Cloud SDK (gcloud) is not installed on this local Windows machine.
    echo.
    echo You can deploy in 2 quick ways:
    echo 1. Google Cloud Shell (Free browser terminal, zero install):
    echo    - Open: https://shell.cloud.google.com
    echo    - Upload or clone this repository
    echo    - Run: ./deploy_cloudrun.sh
    echo.
    echo 2. Install gcloud CLI locally from:
    echo    https://cloud.google.com/sdk/docs/install
    echo.
    pause
    exit /b 1
)

echo.
echo Step 1: Deploying WeatherGPT to Google Cloud Run...
gcloud run deploy weathergpt ^
    --source . ^
    --platform managed ^
    --region us-central1 ^
    --allow-unauthenticated ^
    --port 8080 ^
    --memory 512Mi ^
    --cpu 1 ^
    --min-instances 0 ^
    --max-instances 5

echo.
echo ================================================================
echo Deployment complete! Your live public Cloud Run HTTPS URL is displayed above.
echo ================================================================
pause
