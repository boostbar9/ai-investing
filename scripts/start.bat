@echo off
REM ai-investing — start the stack and open cockpit.
REM Double-click this, or use the tray launcher (recommended).
cd /d "%~dp0\.."
echo Starting Docker stack...
docker compose -f infra\docker\docker-compose.yml up -d || goto :err
echo Waiting for cockpit...
timeout /t 5 /nobreak >nul
start "" "http://localhost:3000"
exit /b 0
:err
echo.
echo Stack failed to start. Is Docker Desktop running?
pause
exit /b 1
