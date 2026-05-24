@echo off
REM ====================================================================
REM ai-investing one-click launcher (Windows Explorer double-click target)
REM
REM Double-click this file from Explorer or use the desktop shortcut
REM created by install-shortcut.ps1. It runs launch.ps1 with the right
REM execution policy so you don't need to right-click anything.
REM
REM Forwards all command-line arguments to launch.ps1 - so you can run
REM    launch.cmd -WithDocker
REM    launch.cmd -NoPull -Port 9000
REM ====================================================================

setlocal

REM Resolve the directory this .cmd lives in, regardless of where it's
REM called from. %~dp0 ends with a backslash.
set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%launch.ps1"

if not exist "%PS_SCRIPT%" (
  echo [error] launch.ps1 not found next to launch.cmd
  echo Expected: %PS_SCRIPT%
  echo.
  pause
  exit /b 1
)

REM -NoProfile skips the user's PowerShell profile so unrelated profile
REM errors (like a missing 'mise' command) don't break the launcher.
REM -ExecutionPolicy Bypass works even when the system policy is Restricted.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*

REM If the user double-clicked, the cmd window will close immediately
REM after PowerShell exits. Pause on non-zero exit so they can read the
REM error message.
if errorlevel 1 (
  echo.
  echo Launcher exited with error code %errorlevel%.
  pause
)

endlocal
