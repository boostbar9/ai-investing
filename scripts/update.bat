@echo off
REM ai-investing — pull latest from GitHub, rebuild, restart.
cd /d "%~dp0\.."
echo Pulling latest from origin/main...
git pull --ff-only origin main || goto :err
echo Rebuilding containers...
docker compose -f infra\docker\docker-compose.yml pull
docker compose -f infra\docker\docker-compose.yml up -d --build || goto :err
echo.
echo Update complete. Cockpit: http://localhost:3000
pause
exit /b 0
:err
echo.
echo Update failed. Check the messages above.
pause
exit /b 1
