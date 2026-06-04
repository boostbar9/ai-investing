#requires -Version 5.1
<#
.SYNOPSIS
    Detached helper that kills the running cockpit uvicorn, pulls latest
    code from origin, reinstalls the package, then relaunches uvicorn.

.DESCRIPTION
    Phase 36f -- invoked as a fully detached child by the /api/remote/restart
    endpoint. Must NOT be invoked directly by humans in normal flow; it
    assumes the calling cockpit is about to die.

    Lifecycle:
      1. Wait $DelaySec seconds so the parent HTTP response can return.
      2. Locate and kill the uvicorn PID we were told to kill (or all
         python.exe holding port 8000 if no PID was given).
      3. Run `git fetch + git pull --ff-only` in $RepoRoot. On dirty
         worktree or non-ff state, log the fact and proceed without pull
         -- the bot still needs to come back up.
      4. Run `pip install -e . --quiet` so package metadata picks up.
      5. Re-spawn uvicorn in a fresh PowerShell window with
         COCKPIT_REMOTE_TOKEN set from the env we inherited.
      6. Write a transcript to $env:TEMP\cockpit_restart_<timestamp>.log
         so the agent can fetch it post-mortem.

    Failure modes are LOGGED but never raise; the goal is "best-effort
    keep the bot running" not "clean restart or nothing."

.PARAMETER UvicornPid
    PID of the uvicorn process to kill. If 0/missing we fall back to
    "any python.exe owning :8000" via Get-NetTCPConnection.

.PARAMETER RepoRoot
    Absolute path to the ai-investing repo root.

.PARAMETER VenvPython
    Absolute path to .venv\Scripts\python.exe inside the repo.

.PARAMETER Token
    Existing COCKPIT_REMOTE_TOKEN to pass into the relaunched cockpit.
    If empty we read from the User env var.

.PARAMETER DelaySec
    How long to wait before killing the parent. Default 2s -- long
    enough for the /api/remote/restart HTTP response to flush.

.PARAMETER NoPull
    Skip the git pull step (just kill + relaunch on existing code).

.PARAMETER Port
    Port uvicorn binds to. Default 8000. Used only as a fallback when
    UvicornPid is not provided.
#>

[CmdletBinding()]
param(
    [int]$UvicornPid = 0,
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [Parameter(Mandatory = $true)]
    [string]$VenvPython,
    [string]$Token = "",
    [int]$DelaySec = 2,
    [switch]$NoPull,
    [int]$Port = 8000
)

# Best-effort -- never let an unhandled error abort the helper, the whole
# point is to keep the bot resilient.
$ErrorActionPreference = "Continue"

$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$logPath = Join-Path $env:TEMP "cockpit_restart_${ts}.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Add-Content -Path $logPath -Value $line
    Write-Host $line
}

Log "=== cockpit_restart_helper starting ==="
Log "RepoRoot   = $RepoRoot"
Log "VenvPython = $VenvPython"
Log "UvicornPid = $UvicornPid"
Log "NoPull     = $NoPull"
Log "Port       = $Port"
Log "DelaySec   = $DelaySec"

# ---------------------------------------------------------------------------
# 1. Let the parent return its HTTP response
# ---------------------------------------------------------------------------
Start-Sleep -Seconds $DelaySec

# ---------------------------------------------------------------------------
# 2. Kill the uvicorn process
# ---------------------------------------------------------------------------
function Kill-OnePid {
    param([int]$Target)
    if ($Target -le 0) { return $false }
    try {
        $p = Get-Process -Id $Target -ErrorAction Stop
        Log "Killing PID $Target ($($p.ProcessName))"
        Stop-Process -Id $Target -Force -ErrorAction Stop
        # Wait for the process slot to actually free up so the port
        # rebind in step 5 doesn't race.
        for ($i = 0; $i -lt 20; $i++) {
            if (-not (Get-Process -Id $Target -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 250
        }
        return $true
    } catch {
        Log "Could not kill PID ${Target}: $_"
        return $false
    }
}

$killed = $false
if ($UvicornPid -gt 0) {
    $killed = Kill-OnePid -Target $UvicornPid
}

if (-not $killed) {
    # Fallback: find whoever owns :$Port and kill them.
    Log "Falling back to port-based PID lookup on :$Port"
    try {
        $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        foreach ($c in $conns) {
            Kill-OnePid -Target $c.OwningProcess | Out-Null
        }
    } catch {
        Log "Get-NetTCPConnection failed: $_"
    }
}

# ---------------------------------------------------------------------------
# 3. Token resolution (must happen BEFORE relaunch so the new uvicorn
#    inherits it). Pull from the User env var if the caller didn't pass one.
# ---------------------------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($Token)) {
    $Token = [Environment]::GetEnvironmentVariable("COCKPIT_REMOTE_TOKEN", "User")
}
if ([string]::IsNullOrWhiteSpace($Token)) {
    Log "WARN: no COCKPIT_REMOTE_TOKEN available; relaunch will be UNAUTHENTICATED-DISABLED."
}

# ---------------------------------------------------------------------------
# 4. Pull + pip install (the actual update)
# ---------------------------------------------------------------------------
Push-Location $RepoRoot
try {
    if ($NoPull) {
        Log "Skipping pull (-NoPull)"
    } else {
        # Ignore untracked files -- pull only touches tracked content.
        $dirty = & git status --porcelain --untracked-files=no 2>$null
        if ($LASTEXITCODE -ne 0) {
            Log "git status failed; skipping pull."
        } elseif ($dirty) {
            Log "Uncommitted local changes; skipping pull to avoid clobber."
            $dirty -split "`n" | Select-Object -First 5 | ForEach-Object { Log "  $_" }
        } else {
            $before = (& git rev-parse --short HEAD 2>$null)
            & git fetch --quiet origin 2>$null
            if ($LASTEXITCODE -ne 0) {
                Log "git fetch failed; booting on $before"
            } else {
                $behindStr = (& git rev-list --count "HEAD..@{u}" 2>$null)
                if ($LASTEXITCODE -ne 0) { $behindStr = "0" }
                try { $behind = [int]$behindStr } catch { $behind = 0 }
                if ($behind -le 0) {
                    Log "Already up to date at $before"
                } else {
                    Log "Pulling $behind commit(s)..."
                    & git pull --ff-only --quiet 2>$null
                    if ($LASTEXITCODE -ne 0) {
                        Log "git pull failed; booting on $before"
                    } else {
                        $after = (& git rev-parse --short HEAD 2>$null)
                        Log "Updated: $before -> $after"
                        Log "Running pip install -e . --quiet"
                        & $VenvPython -m pip install -e . --quiet --disable-pip-version-check 2>&1 | ForEach-Object { Log "  pip: $_" }
                        Log "pip install exit=$LASTEXITCODE"
                    }
                }
            }
        }
    }
} finally {
    Pop-Location
}

# ---------------------------------------------------------------------------
# 5. Relaunch uvicorn in a new window so it lives past this helper.
# ---------------------------------------------------------------------------
if (-not (Test-Path $VenvPython)) {
    Log "FATAL: venv python missing at $VenvPython -- cannot relaunch."
    exit 1
}

$tokenAssign = if ([string]::IsNullOrWhiteSpace($Token)) {
    "# no token; remote surface disabled"
} else {
    "`$env:COCKPIT_REMOTE_TOKEN='$Token'"
}

$startCmd = "$tokenAssign; & '$VenvPython' -m uvicorn packages.cockpit.web.server:app --host 127.0.0.1 --port $Port"

Log "Relaunching uvicorn..."
try {
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoExit", "-Command", $startCmd) `
        -WorkingDirectory $RepoRoot | Out-Null
    Log "Uvicorn launch dispatched."
} catch {
    Log "FATAL: relaunch failed: $_"
    exit 1
}

# ---------------------------------------------------------------------------
# 6. Wait for cockpit to come back up so the agent can poll /version
#    and confirm the new SHA is live.
# ---------------------------------------------------------------------------
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/remote/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}

if ($ready) {
    Log "Cockpit is back up on :$Port after restart."
} else {
    Log "WARN: cockpit did not become reachable in 60s; check the new window."
}

Log "=== helper done ==="
exit 0
