<#
.SYNOPSIS
  One-command installer for ai-investing on Windows.

.DESCRIPTION
  Run from a fresh PowerShell window in any directory. The script will:
    1. Verify Python 3.12+ and Git are installed
    2. Clone the repo (or update it if present)
    3. Create a virtual environment
    4. Install Python dependencies
    5. Create a .env file from .env.example if missing
    6. Run the doctor smoke test

  Designed to be re-runnable: safe to invoke any number of times.

.EXAMPLE
  PS> iwr https://raw.githubusercontent.com/boostbar9/ai-investing/main/scripts/install.ps1 -UseBasicParsing | iex

.EXAMPLE
  PS> .\scripts\install.ps1 -InstallDir C:\dev\ai-investing
#>

[CmdletBinding()]
param(
  [string]$InstallDir = (Join-Path $env:USERPROFILE "ai-investing"),
  [string]$RepoUrl    = "https://github.com/boostbar9/ai-investing.git",
  [switch]$SkipDoctor
)

$ErrorActionPreference = "Stop"

function Invoke-Checked([string]$label, [scriptblock]$action) {
  & $action
  if ($LASTEXITCODE -ne 0) {
    Fail "$label failed (exit code $LASTEXITCODE). See output above."
  }
}

function Write-Section([string]$msg) {
  Write-Host ""
  Write-Host "=== $msg ===" -ForegroundColor Cyan
}

function Write-Ok([string]$msg) {
  Write-Host "  [ok] $msg" -ForegroundColor Green
}

function Write-Warn([string]$msg) {
  Write-Host "  [warn] $msg" -ForegroundColor Yellow
}

function Fail([string]$msg) {
  Write-Host ""
  Write-Host "  [error] $msg" -ForegroundColor Red
  Write-Host ""
  exit 1
}

# ----------------------------------------------------------------------
# 1. Prereq checks
# ----------------------------------------------------------------------
Write-Section "Checking prerequisites"

# Python
try {
  $pyVersion = (& python --version 2>&1).ToString()
} catch {
  Fail "Python is not on PATH. Install Python 3.12+ from https://python.org/downloads/ and check 'Add to PATH'."
}
if ($pyVersion -notmatch "Python 3\.(1[2-9]|[2-9][0-9])") {
  Fail "Need Python 3.12+. Found: $pyVersion. Get it from https://python.org/downloads/."
}
Write-Ok $pyVersion

# Git
try {
  $gitVersion = (& git --version 2>&1).ToString()
} catch {
  Fail "Git is not on PATH. Install from https://git-scm.com/download/win."
}
Write-Ok $gitVersion

# ----------------------------------------------------------------------
# 2. Clone or update repo
# ----------------------------------------------------------------------
Write-Section "Getting the code"

if (Test-Path (Join-Path $InstallDir ".git")) {
  Write-Ok "Repo already exists at $InstallDir - pulling latest"
  Push-Location $InstallDir
  git pull --ff-only
  Pop-Location
} else {
  if (Test-Path $InstallDir) {
    Fail "$InstallDir exists but is not a git repo. Move or delete it, then re-run."
  }
  Write-Ok "Cloning into $InstallDir"
  git clone $RepoUrl $InstallDir
}

Set-Location $InstallDir

# ----------------------------------------------------------------------
# 3. Virtual environment
# ----------------------------------------------------------------------
Write-Section "Creating virtual environment"

$venvPath = Join-Path $InstallDir ".venv"
if (-not (Test-Path (Join-Path $venvPath "Scripts\python.exe"))) {
  python -m venv .venv
  Write-Ok ".venv created"
} else {
  Write-Ok ".venv already exists"
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip    = Join-Path $venvPath "Scripts\pip.exe"

# ----------------------------------------------------------------------
# 4. Install dependencies
# ----------------------------------------------------------------------
Write-Section "Installing Python dependencies (this takes 2-5 minutes)"

Invoke-Checked "pip upgrade" { & $venvPython -m pip install --upgrade pip --quiet }
Write-Ok "pip upgraded"

# No --quiet here: build errors are real and must be visible.
Invoke-Checked "pip install -e .[dev]" { & $venvPip install -e ".[dev]" }
Write-Ok "ai-investing + dev extras installed"

# ----------------------------------------------------------------------
# 5. Environment file
# ----------------------------------------------------------------------
Write-Section "Setting up .env"

$envFile = Join-Path $InstallDir ".env"
if (-not (Test-Path $envFile)) {
  Copy-Item (Join-Path $InstallDir ".env.example") $envFile
  Write-Ok ".env created from .env.example"
  Write-Warn "Edit $envFile and fill in your Alpaca paper keys before running paper trading."
} else {
  Write-Ok ".env already exists (not overwriting)"
}

# ----------------------------------------------------------------------
# 6. Doctor smoke test
# ----------------------------------------------------------------------
if (-not $SkipDoctor) {
  Write-Section "Running doctor"
  $env:PYTHONPATH = "."
  try {
    & $venvPython tools/doctor.py
  } catch {
    Write-Warn "Doctor reported issues - review the output above. Most often this means .env still needs Alpaca keys."
  }
}

# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------
Write-Section "Install complete"

Write-Host ""
Write-Host "Project installed at: " -NoNewline
Write-Host $InstallDir -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit .env and add your Alpaca paper keys (from https://app.alpaca.markets/paper/dashboard/overview)"
Write-Host "  2. Activate the venv:    cd $InstallDir; .\.venv\Scripts\Activate.ps1"
Write-Host "  3. Download market data: `$env:PYTHONPATH='.'; python -m packages.data.pretrain"
Write-Host "  4. First dry-run:        python tools/paper_trade.py --strategy ensemble --dry-run"
Write-Host "  5. Open the cockpit GUI:  `$env:PYTHONPATH='.'; python tools/cockpit.py"
Write-Host "     (then visit http://127.0.0.1:8765 - opens automatically)"
Write-Host ""
