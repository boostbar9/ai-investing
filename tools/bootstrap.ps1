#requires -Version 5.1
<#
.SYNOPSIS
    One-click bootstrap for ai-investing (Phase 36f).

.DESCRIPTION
    Double-click tools\bootstrap.cmd to run this. It:
      1. Pulls latest from origin/main (fast-forward only)
      2. Reinstalls the package: pip install -e . --quiet
      3. Stops any running cockpit on :8000
      4. Launches start_cockpit.ps1 which boots cockpit + tunnel + publishes handle

    After this single run, the agent can call /api/remote/restart from
    chat to ship + apply updates without you touching anything.
#>

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$StartCockpit = Join-Path $PSScriptRoot "start_cockpit.ps1"

function Section($text) {
    Write-Host ""
    Write-Host "=== $text ===" -ForegroundColor Cyan
}

Section "Bootstrap starting"
Write-Host "Repo: $RepoRoot" -ForegroundColor DarkGray

# --------------------------------------------------------------------
# 1. Pull
# --------------------------------------------------------------------
Section "Step 1: Pull latest from origin"
Push-Location $RepoRoot
try {
    # Use --untracked-files=no so runtime-generated dirs like data/learning/
    # and tools/bin/cloudflared.exe don't trigger a false-positive prompt.
    # Pull only touches tracked files, so untracked content is irrelevant.
    $dirty = & git status --porcelain --untracked-files=no 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "git status failed; aborting bootstrap." -ForegroundColor Red
        Pop-Location
        Read-Host "Press Enter to close"
        exit 1
    }
    if ($dirty) {
        Write-Host "Uncommitted changes detected:" -ForegroundColor Yellow
        $dirty -split "`n" | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
        Write-Host ""
        $ans = Read-Host "Continue WITHOUT pulling? (y/N)"
        if ($ans -notmatch '^[Yy]') {
            Write-Host "Aborted by user. Commit or stash your changes, then re-run." -ForegroundColor Yellow
            Pop-Location
            Read-Host "Press Enter to close"
            exit 1
        }
        Write-Host "Continuing without pull." -ForegroundColor Yellow
    } else {
        $before = (& git rev-parse --short HEAD 2>$null)
        & git fetch --quiet origin 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "git fetch failed (offline?). Continuing on $before." -ForegroundColor Yellow
        } else {
            $behindStr = (& git rev-list --count "HEAD..@{u}" 2>$null)
            try { $behind = [int]$behindStr } catch { $behind = 0 }
            if ($behind -le 0) {
                Write-Host "Already up to date at $before." -ForegroundColor Green
            } else {
                Write-Host "Pulling $behind commit(s)..." -ForegroundColor Yellow
                & git pull --ff-only --quiet 2>$null
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "git pull failed. Resolve manually." -ForegroundColor Red
                    Pop-Location
                    Read-Host "Press Enter to close"
                    exit 1
                }
                $after = (& git rev-parse --short HEAD 2>$null)
                Write-Host "Updated: $before -> $after" -ForegroundColor Green
            }
        }
    }
} finally {
    Pop-Location
}

# --------------------------------------------------------------------
# 2. Reinstall package
# --------------------------------------------------------------------
Section "Step 2: pip install -e ."
if (-not (Test-Path $VenvPython)) {
    Write-Host "FATAL: venv python missing at $VenvPython" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}
Push-Location $RepoRoot
try {
    & $VenvPython -m pip install -e . --quiet --disable-pip-version-check
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Pip install OK." -ForegroundColor Green
    } else {
        Write-Host "Pip install returned exit=$LASTEXITCODE; continuing anyway." -ForegroundColor Yellow
    }
} finally {
    Pop-Location
}

# --------------------------------------------------------------------
# 3. Stop any running cockpit on :8000
# --------------------------------------------------------------------
Section "Step 3: Stop existing cockpit (if any)"
try {
    $conns = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    $killedAny = $false
    foreach ($c in $conns) {
        try {
            $p = Get-Process -Id $c.OwningProcess -ErrorAction Stop
            Write-Host "Killing PID $($p.Id) ($($p.ProcessName)) on :8000" -ForegroundColor Yellow
            Stop-Process -Id $p.Id -Force -ErrorAction Stop
            $killedAny = $true
        } catch { }
    }
    if ($killedAny) {
        Start-Sleep -Seconds 2
        Write-Host "Stopped." -ForegroundColor Green
    } else {
        Write-Host "Nothing was on :8000." -ForegroundColor DarkGray
    }
} catch {
    Write-Host "Could not enumerate :8000 owners: $_" -ForegroundColor Yellow
}

# --------------------------------------------------------------------
# 4. Hand off to start_cockpit.ps1
# --------------------------------------------------------------------
Section "Step 4: Launching cockpit + tunnel"
if (-not (Test-Path $StartCockpit)) {
    Write-Host "FATAL: start_cockpit.ps1 missing at $StartCockpit" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "Starting cockpit + tunnel (this window will tail it)..." -ForegroundColor Green
Write-Host ""

# Run start_cockpit.ps1 IN THIS WINDOW so the user sees one continuous
# log stream and can Ctrl+C to stop. -NoPull because we just pulled.
& $StartCockpit -NoPull
