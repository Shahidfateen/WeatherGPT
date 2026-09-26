@echo off
title Push WeatherGPT to GitHub
echo ================================================================
echo Pushing WeatherGPT code to https://github.com/Shahidfateen/WeatherGPT.git
echo ================================================================
cd /d "%~dp0"
git remote set-url origin https://github.com/Shahidfateen/WeatherGPT.git
git push -u origin main
echo.
if %errorlevel% equ 0 (
    echo ================================================================
    echo [SUCCESS] Your code is now live on GitHub!
    echo Next step: Open https://dashboard.render.com to get your permanent link.
    echo ================================================================
) else (
    echo ================================================================
    echo If GitHub showed a login popup in your browser, approve it and re-run.
    echo ================================================================
)
pause
