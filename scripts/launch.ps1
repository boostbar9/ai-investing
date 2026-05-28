<#
.SYNOPSIS
  One-click launcher for ai-investing on Windows.

.DESCRIPTION
  Designed to be run by double-clicking ``launch.cmd`` from Windows Explorer
  or from a desktop shortcut created by ``install-shortcut.ps1``. The script:

    1. Locates the install directory (walks up from script path)
    2. Verifies the venv exists, recreates if missing
    3. Activates the venv in this process
    4. Validates .env exists and contains Alpaca paper keys
    5. Runs git pull (fast-forward only; never destructive)
    6. Optionally starts the Docker stack (--WithDocker switch)
    7. Sets PYTHONPATH and launches the cockpit on http://127.0.0.1:8765
    8. Opens the browser to the cockpit

  Every external command is wrapped in a null-safe helper that fails loud
  with a clear message instead of silently continuing.

.PARAMETER WithDocker
  Start the docker-compose stack (Postgres, Dragonfly, Temporal, Ollama,
  Grafana, etc.) before launching the cockpit. Requires Docker Desktop.
  Without this switch, only the cockpit + paper runner are started, which
  is all you need for paper trading.

.PARAMETER NoPull
  Skip the ``git pull`` step. Useful when you're offline or have local
  edits you don't want to risk rebasing.

.PARAMETER Port
  Port for the cockpit web GUI. Default 8765.

.PARAMETER NoBrowser
  Don't auto-open a browser tab. Useful if launching headless.

.EXAMPLE
  PS> .\scripts\launch.ps1

.EXAMPLE
  PS> .\scripts\launch.ps1 -WithDocker

.EXAMPLE
  PS> .\scripts\launch.ps1 -NoPull -Port 9000
#>

[CmdletBinding()]
param(
  [switch]$WithDocker,
  [switch]$NoPull,
  [int]$Port = 8765,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

function Write-Section([string]$msg) {
  Write-Host ""
  Write-Host "=== $msg ===" -ForegroundColor Cyan
}
function Write-Ok([string]$msg)   { Write-Host "  [ok] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Write-Info([string]$msg) { Write-Host "  $msg" -ForegroundColor Gray }
function Fail([string]$msg) {
  Write-Host ""
  Write-Host "  [error] $msg" -ForegroundColor Red
  Write-Host ""
  Write-Host "Press any key to exit..." -ForegroundColor DarkGray
  try { [void]$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") } catch { Start-Sleep -Seconds 8 }
  exit 1
}

# Null-safe stdout capture - returns "" instead of crashing under iwr|iex.
function Get-Output([scriptblock]$action) {
  $out = & $action 2>$null
  if ($null -eq $out) { return "" }
  if ($out -is [array]) { $out = $out -join "`n" }
  return [string]$out
}

# Run a command, fail loudly if its exit code is non-zero.
function Invoke-Checked([string]$label, [scriptblock]$action) {
  & $action
  if ($LASTEXITCODE -ne 0) {
    Fail "$label failed (exit code $LASTEXITCODE). See output above."
  }
}

function Test-CommandExists([string]$name) {
  return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# ----------------------------------------------------------------------
# 1. Locate the install directory
# ----------------------------------------------------------------------
Write-Section "Locating install"

# When invoked from launch.cmd or a shortcut, $PSScriptRoot is scripts/.
# Walk up one level to find the repo root.
if ($PSScriptRoot) {
  $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
  $repoRoot = (Get-Location).Path
}

if (-not (Test-Path (Join-Path $repoRoot ".git"))) {
  Fail "Not inside an ai-investing checkout. Expected .git at $repoRoot. Run scripts/install.ps1 first."
}

Set-Location $repoRoot
Write-Ok "Using $repoRoot"

# ----------------------------------------------------------------------
# 2. Verify or create the venv
# ----------------------------------------------------------------------
Write-Section "Checking virtual environment"

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$venvActivate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvPython)) {
  Write-Warn ".venv not found - creating one"
  if (-not (Test-CommandExists "python")) {
    Fail "Python is not on PATH. Install Python 3.12+ from https://python.org/downloads/."
  }
  Invoke-Checked "python -m venv" { python -m venv .venv }
  Invoke-Checked "pip install -e .[dev]" { & $venvPython -m pip install -e ".[dev]" }
  Write-Ok "venv created and dependencies installed"
} else {
  Write-Ok ".venv ready"
}

# Activate in-process so child commands see it.
. $venvActivate
Write-Ok "venv activated"

# ----------------------------------------------------------------------
# 3. Validate .env
# ----------------------------------------------------------------------
Write-Section "Validating .env"

$envFile = Join-Path $repoRoot ".env"
if (-not (Test-Path $envFile)) {
  if (Test-Path (Join-Path $repoRoot ".env.example")) {
    Copy-Item (Join-Path $repoRoot ".env.example") $envFile
    Write-Warn ".env did not exist - created one from .env.example"
    Write-Warn "Edit $envFile and add your Alpaca paper keys before next launch."
  } else {
    Fail ".env not found and no .env.example to seed from. Run scripts/install.ps1."
  }
}

# Quick sanity check: the Alpaca paper keys must be set to something non-empty.
$envText = Get-Content $envFile -Raw
$hasKeyId  = $envText -match "(?m)^\s*ALPACA_PAPER_KEY_ID\s*=\s*\S"
$hasSecret = $envText -match "(?m)^\s*ALPACA_PAPER_SECRET\s*=\s*\S"
if (-not ($hasKeyId -and $hasSecret)) {
  Write-Warn "ALPACA_PAPER_KEY_ID or ALPACA_PAPER_SECRET appears blank in $envFile"
  Write-Warn "Get free paper keys at https://app.alpaca.markets/paper/dashboard/overview"
  Write-Warn "Continuing anyway - the cockpit will start but paper trades will halt."
} else {
  Write-Ok "Alpaca paper keys present in .env"
}

# ----------------------------------------------------------------------
# 4. Git pull (optional)
# ----------------------------------------------------------------------
if (-not $NoPull) {
  Write-Section "Syncing latest changes"

  if (-not (Test-CommandExists "git")) {
    Write-Warn "git not on PATH - skipping pull"
  } else {
    $branch = (Get-Output { git rev-parse --abbrev-ref HEAD }).Trim()
    $dirty  = (Get-Output { git status --porcelain }).Trim()

    if ($dirty) {
      Write-Warn "Working tree has local changes - skipping pull (commit/stash first to update)"
    } elseif ($branch -ne "main") {
      Write-Warn "On branch '$branch' (not main) - skipping pull"
    } else {
      $preHash = (Get-Output { git rev-parse HEAD }).Trim()
      git fetch origin main --quiet 2>$null
      $behindStr = (Get-Output { git rev-list --count "HEAD..origin/main" }).Trim()
      if (-not $behindStr) { $behindStr = "0" }
      $behind = [int]$behindStr
      if ($behind -eq 0) {
        Write-Ok "Already up to date"
      } else {
        Write-Info "$behind new commit(s) on origin/main - pulling"
        Invoke-Checked "git pull" { git pull --ff-only origin main --quiet }
        $postHash = (Get-Output { git rev-parse HEAD }).Trim()
        Write-Ok "Updated $($preHash.Substring(0,7)) -> $($postHash.Substring(0,7))"

        # If pyproject.toml or dependencies changed, reinstall.
        $changed = (Get-Output { git diff --name-only "$preHash..$postHash" })
        if ($changed -match "(?m)^pyproject\.toml$") {
          Write-Info "pyproject.toml changed - reinstalling dependencies"
          Invoke-Checked "pip install -e .[dev]" { & $venvPython -m pip install -e ".[dev]" --quiet }
          Write-Ok "Dependencies refreshed"
        }
      }
    }
  }
} else {
  Write-Info "Skipping git pull (-NoPull set)"
}

# ----------------------------------------------------------------------
# 5. Docker stack (optional)
# ----------------------------------------------------------------------
if ($WithDocker) {
  Write-Section "Starting Docker stack"

  if (-not (Test-CommandExists "docker")) {
    Fail "Docker is not on PATH. Install Docker Desktop from https://www.docker.com/products/docker-desktop/ or omit -WithDocker."
  }

  # Make sure Docker Desktop is actually running.
  $dockerOk = $false
  try {
    docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $dockerOk = $true }
  } catch {}

  if (-not $dockerOk) {
    Write-Warn "Docker daemon not responding. Make sure Docker Desktop is running, then retry."
    Write-Warn "Continuing without Docker - only paper-trading paths will work."
  } else {
    $composeFile = Join-Path $repoRoot "infra\docker\docker-compose.yml"
    if (-not (Test-Path $composeFile)) {
      Write-Warn "docker-compose.yml not found at $composeFile - skipping"
    } else {
      Write-Info "Bringing up infra services (this may take a few minutes on first run)..."
      Invoke-Checked "docker compose up -d" {
        docker compose -f $composeFile --env-file $envFile up -d
      }
      Write-Ok "Docker stack is up"
    }
  }
} else {
  Write-Info "Docker stack not requested (use -WithDocker to start Postgres + Temporal + Ollama + Grafana)"
}

# ----------------------------------------------------------------------
# 6. Run the one-click boot orchestrator (tools/boot.py).
#
# This warms up Ollama, pulls any missing models, creates the data/
# subtree, runs the doctor, and confirms the cockpit port is free.
# Skipped steps and degraded states are surfaced inline so the user sees
# the same picture the cockpit will. A 'failed' overall blocks the
# cockpit launch; 'degraded' continues with a yellow badge.
# ----------------------------------------------------------------------
Write-Section "Warming up the stack"

$env:PYTHONPATH = "."
# Force unbuffered, UTF-8 stdout so nothing the orchestrator prints can be
# swallowed by PowerShell's stderr-as-error trap when $ErrorActionPreference
# is 'Stop'. Capturing combined stdout+stderr into a tee file gives us a
# permanent record the user can paste even after the window scrolls or
# closes.
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$bootLogDir = Join-Path $repoRoot "data\cockpit"
if (-not (Test-Path $bootLogDir)) { New-Item -ItemType Directory -Path $bootLogDir -Force | Out-Null }
$bootLog = Join-Path $bootLogDir "boot_launcher.log"

# Temporarily relax ErrorActionPreference so a native command's stderr
# emission cannot terminate the script before we read $LASTEXITCODE.
$oldEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  # 2>&1 merges stderr into the success stream so PowerShell shows the
  # Python traceback / step output instead of dropping it. Tee-Object
  # mirrors everything to disk for post-mortem inspection.
  & $venvPython -u -m tools.boot 2>&1 | Tee-Object -FilePath $bootLog
  $bootExit = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $oldEAP
}

if ($bootExit -ne 0) {
  $hint = switch ($bootExit) {
    1 { "Python interpreter exited 1 -- typically a ModuleNotFoundError or import-time crash. See $bootLog." }
    2 { "One or more boot steps failed (look for [XX] above)." }
    3 { "The boot orchestrator itself crashed (Python traceback above)." }
    default { "Unexpected exit code $bootExit from tools.boot. See $bootLog." }
  }
  # Best-effort: surface the failed step name from data/cockpit/boot.json so
  # the user has somewhere concrete to look even if scrollback is gone.
  $bootJson = Join-Path $repoRoot "data\cockpit\boot.json"
  $failedStep = ""
  if (Test-Path $bootJson) {
    try {
      $bootData = Get-Content $bootJson -Raw | ConvertFrom-Json
      $failed = @($bootData.results | Where-Object { $_.status -eq "failed" })
      if ($failed.Count -gt 0) {
        $failedStep = ($failed | ForEach-Object { "$($_.name): $($_.message)" }) -join "; "
      }
    } catch {
      # Best-effort only.
    }
  }

  # If the orchestrator died before writing boot.json (typical for exit 1),
  # tail the captured log so the user sees the actual Python error inline.
  $logTail = ""
  if ((Test-Path $bootLog) -and (-not $failedStep)) {
    try {
      $tail = Get-Content $bootLog -Tail 25 -ErrorAction SilentlyContinue
      if ($tail) { $logTail = ($tail -join "`n") }
    } catch {
      # Best-effort only.
    }
  }

  Write-Host ""
  Write-Host "  --- last 25 lines of $bootLog ---" -ForegroundColor DarkGray
  if ($logTail) {
    Write-Host $logTail -ForegroundColor DarkGray
  } else {
    Write-Host "  (log file empty -- Python exited before producing any output)" -ForegroundColor DarkGray
  }
  Write-Host "  --- end log ---" -ForegroundColor DarkGray

  if ($failedStep) {
    Fail "boot orchestrator failed (exit $bootExit). $hint`n  Failed step(s): $failedStep`n  Full log: $bootLog"
  } else {
    Fail "boot orchestrator failed (exit $bootExit). $hint`n  Full log: $bootLog"
  }
}

# ----------------------------------------------------------------------
# 7. Launch the cockpit
# ----------------------------------------------------------------------
Write-Section "Starting cockpit"
Write-Ok "PYTHONPATH=."
Write-Ok "Cockpit will be available at http://127.0.0.1:$Port"

$browserFlag = if ($NoBrowser) { "--no-browser" } else { "" }
$cockpitArgs = @("tools/cockpit.py", "--port", "$Port")
if ($NoBrowser) { $cockpitArgs += "--no-browser" }

Write-Host ""
Write-Host "Press Ctrl+C to stop the cockpit." -ForegroundColor DarkGray
Write-Host ""

# Run in the foreground so the PowerShell window shows the server logs.
# When the user closes the window, Python exits and (if docker is up)
# they can leave Docker running or `docker compose down` manually.
& $venvPython @cockpitArgs
