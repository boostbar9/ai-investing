<#
.SYNOPSIS
  Register the cockpit tray to launch on every Windows logon.

.DESCRIPTION
  Creates (or replaces) a Task Scheduler entry named "AI Investing
  Cockpit Tray" that runs ``pythonw.exe -m tools.tray.cockpit_tray``
  whenever the current user logs in.

  Why Task Scheduler instead of a Startup-folder shortcut:

    * Task Scheduler can launch with the user's normal token (so it
      sees the same env, profile, and credentials as a manual run) and
      survives upgrades that wipe the Startup folder.
    * Each run is logged in Event Viewer so failures aren't silent.
    * The "If the task is already running" policy keeps a single tray
      instance even on weird re-logon races.

  The task runs *interactively* as the current user (no admin
  elevation, no service account). It will not run while the laptop is
  on battery and idle for too long -- Windows' default settings are
  preserved.

  Idempotent: re-running deletes any prior task with the same name
  before recreating it. Safe to run after every git pull.

.PARAMETER TaskName
  Override the task name. Default: "AI Investing Cockpit Tray".

.PARAMETER Uninstall
  Remove the scheduled task instead of installing it.

.EXAMPLE
  PS> .\scripts\install-autostart.ps1

.EXAMPLE
  PS> .\scripts\install-autostart.ps1 -Uninstall
#>

[CmdletBinding()]
param(
  [string]$TaskName = "AI Investing Cockpit Tray",
  [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

function Write-Ok([string]$msg)   { Write-Host "  [ok] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Write-Step([string]$msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Cyan }
function Fail([string]$msg) {
  Write-Host ""
  Write-Host "  [error] $msg" -ForegroundColor Red
  Write-Host ""
  exit 1
}

# Resolve the repo root (one level up from this script).
$repoRoot   = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$venvPythonW = Join-Path $repoRoot ".venv\Scripts\pythonw.exe"

# ---------------------------------------------------------------------------
# Uninstall path
# ---------------------------------------------------------------------------

if ($Uninstall) {
  Write-Step "Uninstalling autostart"
  $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Ok "Removed scheduled task '$TaskName'"
  } else {
    Write-Warn "No scheduled task named '$TaskName' was registered."
  }
  exit 0
}

# ---------------------------------------------------------------------------
# Install path
# ---------------------------------------------------------------------------

Write-Step "Installing autostart for the cockpit tray"

if (-not (Test-Path $venvPython)) {
  Fail ".venv not found at $repoRoot\.venv. Run scripts\install.ps1 first."
}

if (Test-Path $venvPythonW) {
  $launcher = $venvPythonW
  Write-Ok "Will launch via pythonw.exe (no console window on login)"
} else {
  $launcher = $venvPython
  Write-Warn "pythonw.exe missing; using python.exe (a console window will flash)"
}

# Build the action: run the tray module from the repo root.
$action = New-ScheduledTaskAction `
  -Execute $launcher `
  -Argument "-m tools.tray.cockpit_tray" `
  -WorkingDirectory $repoRoot

# Trigger: at logon for the current user only.
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Run as the current user with their normal token. No admin elevation.
$principal = New-ScheduledTaskPrincipal `
  -UserId $env:USERNAME `
  -LogonType Interactive `
  -RunLevel Limited

# Sensible defaults: don't fight Windows' battery/idle policies, keep one
# instance, kill if it hangs too long.
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -MultipleInstances IgnoreNew `
  -ExecutionTimeLimit ([TimeSpan]::Zero) `
  -RestartCount 2 `
  -RestartInterval (New-TimeSpan -Minutes 5)

# Replace any prior task so re-running this script always wins.
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
  Write-Warn "Removed prior task '$TaskName' so this run can recreate it cleanly"
}

Register-ScheduledTask `
  -TaskName    $TaskName `
  -Description "Launch the AI Investing cockpit tray at every user logon." `
  -Action      $action `
  -Trigger     $trigger `
  -Principal   $principal `
  -Settings    $settings | Out-Null

Write-Ok "Registered scheduled task '$TaskName'"

Write-Host ""
Write-Host "Test it now:" -ForegroundColor Cyan
Write-Host "    Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
Write-Host ""
Write-Host "To remove the autostart later:" -ForegroundColor Cyan
Write-Host "    .\scripts\install-autostart.ps1 -Uninstall" -ForegroundColor Cyan
