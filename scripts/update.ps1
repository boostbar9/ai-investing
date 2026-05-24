<#
.SYNOPSIS
  Safe in-place updater for ai-investing on Windows.

.DESCRIPTION
  Pulls the latest code, reinstalls Python deps only if pyproject.toml
  changed, warns about new env vars added to .env.example, and re-runs
  the doctor smoke test.

  Designed to be re-runnable and to never destroy local changes:
    * Refuses to pull if the working tree has uncommitted edits
    * Refuses to pull if you're on a branch other than main
    * Stashes nothing automatically — surfaces the problem and exits

.EXAMPLE
  PS> cd C:\Users\devfa\ai-investing; .\scripts\update.ps1

.EXAMPLE
  PS> iwr https://raw.githubusercontent.com/boostbar9/ai-investing/main/scripts/update.ps1 -UseBasicParsing | iex
#>

[CmdletBinding()]
param(
  [string]$InstallDir = (Get-Location).Path,
  [switch]$SkipDoctor,
  [switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-Section([string]$msg) {
  Write-Host ""
  Write-Host "=== $msg ===" -ForegroundColor Cyan
}
function Write-Ok([string]$msg)   { Write-Host "  [ok] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Fail([string]$msg) {
  Write-Host ""
  Write-Host "  [error] $msg" -ForegroundColor Red
  Write-Host ""
  exit 1
}

# ----------------------------------------------------------------------
# 1. Verify we're in a real install
# ----------------------------------------------------------------------
Write-Section "Locating install"

# If the script was run from outside the repo (e.g. via iwr|iex),
# fall back to ~/ai-investing.
if (-not (Test-Path (Join-Path $InstallDir ".git"))) {
  $fallback = Join-Path $env:USERPROFILE "ai-investing"
  if (Test-Path (Join-Path $fallback ".git")) {
    $InstallDir = $fallback
  } else {
    Fail "Not inside an ai-investing checkout and ~/ai-investing doesn't exist. cd to your install first, or run scripts/install.ps1."
  }
}
Set-Location $InstallDir
Write-Ok "Using $InstallDir"

# ----------------------------------------------------------------------
# 2. Safety checks
# ----------------------------------------------------------------------
Write-Section "Safety checks"

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "main") {
  if (-not $Force) {
    Fail "On branch '$branch', not 'main'. Switch with 'git switch main' or re-run with -Force."
  }
  Write-Warn "On branch '$branch' (continuing because -Force was set)"
}

$dirty = (git status --porcelain).Trim()
if ($dirty) {
  if (-not $Force) {
    Write-Host ""
    Write-Host "  Uncommitted changes detected:" -ForegroundColor Yellow
    git status --short
    Fail "Commit, stash, or discard local changes first. Or re-run with -Force to attempt git pull anyway."
  }
  Write-Warn "Continuing with uncommitted changes (-Force set)"
}
Write-Ok "Working tree clean, on main"

# ----------------------------------------------------------------------
# 3. Capture pre-pull state for diff detection
# ----------------------------------------------------------------------
$preHash      = (git rev-parse HEAD).Trim()
$preToml      = if (Test-Path "pyproject.toml") { (Get-FileHash pyproject.toml).Hash } else { "" }
$preEnvExample = if (Test-Path ".env.example") { (Get-FileHash .env.example).Hash } else { "" }

# ----------------------------------------------------------------------
# 4. Pull
# ----------------------------------------------------------------------
Write-Section "Fetching updates"

git fetch origin main --quiet
$behindCount = [int]((git rev-list --count "HEAD..origin/main").Trim())
if ($behindCount -eq 0) {
  Write-Ok "Already up to date — nothing to pull"
} else {
  Write-Ok "$behindCount new commit(s) on origin/main"
  git pull --ff-only origin main
}

$postHash = (git rev-parse HEAD).Trim()

# ----------------------------------------------------------------------
# 5. Update Python deps if pyproject.toml changed
# ----------------------------------------------------------------------
Write-Section "Python dependencies"

$venvPython = Join-Path $InstallDir ".venv\Scripts\python.exe"
$venvPip    = Join-Path $InstallDir ".venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
  Write-Warn ".venv not found — running fresh dependency install"
  python -m venv .venv
  & $venvPython -m pip install --upgrade pip --quiet
  & $venvPip install -e ".[dev]" --quiet
  Write-Ok "Dependencies installed"
} else {
  $postToml = if (Test-Path "pyproject.toml") { (Get-FileHash pyproject.toml).Hash } else { "" }
  if ($preToml -ne $postToml -or $preHash -eq "") {
    Write-Ok "pyproject.toml changed — reinstalling dependencies"
    & $venvPip install -e ".[dev]" --quiet
    Write-Ok "Dependencies updated"
  } else {
    Write-Ok "pyproject.toml unchanged — skipping pip install"
  }
}

# ----------------------------------------------------------------------
# 6. Warn about new env vars
# ----------------------------------------------------------------------
Write-Section "Environment variables"

if (Test-Path ".env.example") {
  $postEnvExample = (Get-FileHash .env.example).Hash
  if ($preEnvExample -ne $postEnvExample -and $preEnvExample -ne "") {
    Write-Warn ".env.example changed since your last update. Compare with your .env:"
    Write-Host "  PS> code --diff .env .env.example" -ForegroundColor Gray
    Write-Host "  (or run 'git diff $preHash..$postHash -- .env.example')" -ForegroundColor Gray
  } else {
    Write-Ok ".env.example unchanged"
  }
}

# ----------------------------------------------------------------------
# 7. Doctor smoke test
# ----------------------------------------------------------------------
if (-not $SkipDoctor -and (Test-Path $venvPython)) {
  Write-Section "Running doctor"
  $env:PYTHONPATH = "."
  try {
    & $venvPython tools/doctor.py
  } catch {
    Write-Warn "Doctor reported issues — review the output above."
  }
}

# ----------------------------------------------------------------------
# 8. Changelog summary
# ----------------------------------------------------------------------
if ($preHash -ne $postHash -and $preHash -ne "") {
  Write-Section "What changed"
  git log --oneline "$preHash..$postHash"
}

Write-Section "Update complete"
Write-Host ""
Write-Host "You can now re-run the nightly paper job:" -ForegroundColor Cyan
Write-Host "  PS> .\.venv\Scripts\Activate.ps1"
Write-Host "  PS> `$env:PYTHONPATH='.'; python tools/paper_trade.py --strategy ensemble"
Write-Host ""
