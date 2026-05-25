<#
.SYNOPSIS
  One-click installer for the cockpit tray app.

.DESCRIPTION
  Sets up everything Devin needs to run the cockpit as a daily driver:

    1. Ensures the venv has the [tray] extras (pystray + Pillow).
    2. Creates a desktop shortcut "AI Investing Cockpit" that launches
       the tray via ``pythonw.exe`` (no console window flashes on
       login).
    3. Creates a Start-menu entry with the same target.

  All steps are idempotent: re-running the installer is safe and
  overwrites existing shortcuts so they always point at the current
  Python install.

  Pair with ``install-autostart.ps1`` to also have the tray launch on
  every login.

.PARAMETER Name
  Override the shortcut name. Default: "AI Investing Cockpit".

.EXAMPLE
  PS> .\scripts\install-cockpit-tray.ps1
#>

[CmdletBinding()]
param(
  [string]$Name = "AI Investing Cockpit"
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
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$venvPythonW = Join-Path $repoRoot ".venv\Scripts\pythonw.exe"

if (-not (Test-Path $venvPython)) {
  Fail ".venv not found at $repoRoot\.venv. Run scripts\install.ps1 first."
}

# ---------------------------------------------------------------------------
# 1. Install tray extras
# ---------------------------------------------------------------------------

Write-Step "Installing tray dependencies (pystray + Pillow)"

# Reinstall is idempotent and cheap; -q keeps it quiet on the happy path.
& $venvPython -m pip install -q -e ".[tray]"
if ($LASTEXITCODE -ne 0) {
  Fail "pip install [tray] failed. Re-run with verbose output to debug."
}
Write-Ok "pystray + Pillow installed"

# ---------------------------------------------------------------------------
# 2. Desktop + Start-menu shortcuts
# ---------------------------------------------------------------------------

Write-Step "Creating shortcuts"

if (-not (Test-Path $venvPythonW)) {
  Write-Warn "pythonw.exe missing; falling back to python.exe (a console window will flash)"
  $launcher = $venvPython
} else {
  $launcher = $venvPythonW
}

$arguments       = "-m tools.tray.cockpit_tray"
$workingDir      = $repoRoot
$description     = "AI Investing Cockpit -- system tray launcher"

# Build the shortcut object once, reuse for both locations.
function New-TrayShortcut([string]$path) {
  if (Test-Path $path) {
    Remove-Item $path -Force
    Write-Warn "Removed existing shortcut: $path"
  }
  $shell = New-Object -ComObject WScript.Shell
  $lnk = $shell.CreateShortcut($path)
  $lnk.TargetPath       = $launcher
  $lnk.Arguments        = $arguments
  $lnk.WorkingDirectory = $workingDir
  $lnk.WindowStyle      = 7   # minimized -- the tray app has no main window anyway
  $lnk.Description      = $description

  # Try the repo icon first; fall back to a sensible system icon so the
  # shortcut never lacks an image.
  $iconCandidates = @(
    (Join-Path $repoRoot "scripts\icon.ico"),
    "$env:WINDIR\System32\imageres.dll"
  )
  foreach ($candidate in $iconCandidates) {
    if (Test-Path $candidate) {
      if ($candidate -like "*.dll") {
        # Index 109 in imageres.dll is a generic gauge/dashboard icon.
        $lnk.IconLocation = "$candidate,109"
      } else {
        $lnk.IconLocation = "$candidate,0"
      }
      break
    }
  }

  $lnk.Save()
  Write-Ok "Shortcut created: $path"
}

$desktop = [Environment]::GetFolderPath("Desktop")
if (-not (Test-Path $desktop)) {
  Fail "Could not locate the Desktop folder."
}
$desktopLnk = Join-Path $desktop "$Name.lnk"
New-TrayShortcut $desktopLnk

$startMenuDir = [Environment]::GetFolderPath("Programs")
if (Test-Path $startMenuDir) {
  $startLnk = Join-Path $startMenuDir "$Name.lnk"
  New-TrayShortcut $startLnk
} else {
  Write-Warn "Start menu Programs folder not found; skipping Start-menu shortcut."
}

Write-Host ""
Write-Host "Done. Double-click '$Name' on your desktop to launch the cockpit tray." -ForegroundColor Cyan
Write-Host "To also launch automatically on login, run:" -ForegroundColor Cyan
Write-Host "    .\scripts\install-autostart.ps1" -ForegroundColor Cyan
