#requires -Version 5.1
<#
.SYNOPSIS
    One-command launcher for the ai-investing cockpit + remote tunnel.

.DESCRIPTION
    Phase 36d -- starts everything you need for the Perplexity agent
    to remote-control your cockpit:

      1. Ensures COCKPIT_REMOTE_TOKEN is set (generates if missing).
      2. Starts uvicorn in a new window (with the token in env).
      3. Waits for cockpit to be reachable on 127.0.0.1:8000.
      4. Ensures cloudflared.exe is installed (auto-downloads if missing).
      5. Starts a Cloudflare quick-tunnel, parses the public URL.
      6. Writes URL + token to data/cockpit/remote_handle.json.
      7. Publishes the handle to a private GitHub branch
         (refs/heads/cockpit-handle) so the agent can always find it.
      8. Tails the tunnel; Ctrl+C stops everything.

    Run this ONCE and the agent can reach your bot for as long as the
    window stays open. Re-run anytime; URL refreshes automatically.

.PARAMETER NoPublish
    Skip the GitHub branch publish step. Use if you want to keep the
    handle local only and paste the URL/token to the agent yourself.

.PARAMETER NewToken
    Force-generate a new remote token, replacing the existing one.

.PARAMETER NoPull
    Skip the auto-pull step. Use if you have uncommitted local changes
    you don't want to risk, or you want to boot from your current code.

.EXAMPLE
    PS> .\tools\start_cockpit.ps1
    Starts everything; agent auto-discovers via GitHub handle branch.
#>

[CmdletBinding()]
param(
    [switch]$NoPublish,
    [switch]$NewToken,
    [switch]$NoPull
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BinDir = Join-Path $RepoRoot "tools\bin"
$CloudflaredExe = Join-Path $BinDir "cloudflared.exe"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$HandleDir = Join-Path $RepoRoot "data\cockpit"
$HandlePath = Join-Path $HandleDir "remote_handle.json"
$CockpitUrl = "http://127.0.0.1:8000"

function Write-Section($text) {
    Write-Host ""
    Write-Host "=== $text ===" -ForegroundColor Cyan
}

function Test-CockpitReachable {
    param([string]$Url)
    try {
        $r = Invoke-WebRequest -Uri "$Url/api/remote/health" -UseBasicParsing -TimeoutSec 3
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function New-RemoteToken {
    $bytes = New-Object byte[] 16
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    return -join ($bytes | ForEach-Object { $_.ToString("x2") })
}

function Install-Cloudflared {
    if (Test-Path $CloudflaredExe) { return $CloudflaredExe }
    Write-Host "cloudflared not found -- downloading to $BinDir ..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -Uri $url -OutFile $CloudflaredExe -UseBasicParsing
    Write-Host "Installed: $CloudflaredExe" -ForegroundColor Green
    return $CloudflaredExe
}

# --------------------------------------------------------------------
# 0. Auto-pull latest code from origin (Phase 36e+)
# --------------------------------------------------------------------
# Rationale: the cockpit boots from whatever is on disk. Without this
# step, a fresh launch reuses stale code even if the agent has pushed
# fixes. Doing a fast-forward-only pull here means every double-click
# of the desktop shortcut starts a clean, current cockpit.
#
# Safety properties:
#   - Fast-forward only: a divergent local branch is left untouched.
#   - Uncommitted changes block the pull (git refuses); we detect and
#     warn but continue booting on the existing code instead of
#     halting -- the bot's job is to be running, not to be pristine.
#   - --NoPull skips the whole block for the rare case you want to
#     boot exactly what is on disk.
#   - All git output is captured and shown; no silent failures.
Write-Section "Step 0: Sync code from origin"
if ($NoPull) {
    Write-Host "Skipped (-NoPull flag set)." -ForegroundColor Yellow
} else {
    # Git writes progress to stderr; relax error policy for this block
    # and rely on $LASTEXITCODE for real failure detection (same trick
    # we use in the publish step).
    $prevErrPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $RepoRoot
    try {
        # 0a. Refuse to pull if there are uncommitted changes -- pulling
        #     would either fail or risk a stash dance we don't want.
        #     Ignore untracked files (data/learning/, tools/bin/, etc.)
        #     since pull only touches tracked files.
        $dirty = & git status --porcelain --untracked-files=no 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "git status failed (not a repo or git missing?); skipping pull." -ForegroundColor Yellow
        } elseif ($dirty) {
            Write-Host "Uncommitted local changes detected -- skipping pull." -ForegroundColor Yellow
            Write-Host "  Commit or stash to enable auto-pull on next launch." -ForegroundColor DarkGray
            $dirty -split "`n" | Select-Object -First 5 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
        } else {
            $beforeSha = (& git rev-parse --short HEAD 2>$null)
            & git fetch --quiet origin 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "git fetch failed (offline? auth?); booting on current code." -ForegroundColor Yellow
            } else {
                # Count commits we're behind so we only pip-install if we actually moved.
                $behindStr = (& git rev-list --count "HEAD..@{u}" 2>$null)
                if ($LASTEXITCODE -ne 0) { $behindStr = "0" }
                try { $behind = [int]$behindStr } catch { $behind = 0 }
                if ($behind -le 0) {
                    Write-Host "Already up to date at $beforeSha." -ForegroundColor Green
                } else {
                    Write-Host "Pulling $behind commit(s) from origin..." -ForegroundColor Yellow
                    & git pull --ff-only --quiet 2>$null
                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "git pull --ff-only failed; booting on current code at $beforeSha." -ForegroundColor Yellow
                        Write-Host "  Resolve manually: git status / git pull / git merge" -ForegroundColor DarkGray
                    } else {
                        $afterSha = (& git rev-parse --short HEAD 2>$null)
                        Write-Host "Updated: $beforeSha -> $afterSha" -ForegroundColor Green
                        # Pip-install in editable mode so package metadata + entry
                        # points pick up. --quiet keeps the console readable; the
                        # full log goes to a temp file for post-mortem if needed.
                        $pipLog = Join-Path $env:TEMP "cockpit_pip.log"
                        Write-Host "Reinstalling package (pip install -e .)..." -ForegroundColor Yellow
                        & $VenvPython -m pip install -e . --quiet --disable-pip-version-check *> $pipLog
                        if ($LASTEXITCODE -eq 0) {
                            Write-Host "Pip install OK." -ForegroundColor Green
                        } else {
                            Write-Host "Pip install returned exit=$LASTEXITCODE; booting anyway." -ForegroundColor Yellow
                            Write-Host "  Full log: $pipLog" -ForegroundColor DarkGray
                        }
                    }
                }
            }
        }
    } finally {
        Pop-Location
        $ErrorActionPreference = $prevErrPref
    }
}

# --------------------------------------------------------------------
# 1. Token
# --------------------------------------------------------------------
Write-Section "Step 1: Auth token"
$existing = [Environment]::GetEnvironmentVariable("COCKPIT_REMOTE_TOKEN", "User")
if ($NewToken -or [string]::IsNullOrWhiteSpace($existing) -or $existing.Length -lt 16) {
    $token = New-RemoteToken
    [Environment]::SetEnvironmentVariable("COCKPIT_REMOTE_TOKEN", $token, "User")
    Write-Host "Generated NEW remote token, saved to User env." -ForegroundColor Green
} else {
    $token = $existing
    Write-Host "Using existing COCKPIT_REMOTE_TOKEN (length=$($token.Length))." -ForegroundColor Green
}
$env:COCKPIT_REMOTE_TOKEN = $token

# --------------------------------------------------------------------
# 2. Start cockpit (or detect already-running)
# --------------------------------------------------------------------
Write-Section "Step 2: Cockpit (uvicorn)"
if (Test-CockpitReachable -Url $CockpitUrl) {
    Write-Host "Cockpit already running at $CockpitUrl." -ForegroundColor Green
    Write-Host "NOTE: if you JUST changed the token, restart it: Ctrl+C in cockpit window." -ForegroundColor Yellow
} else {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "Venv python not found at $VenvPython" -ForegroundColor Red
        Write-Host "Create one first: python -m venv .venv ; .\.venv\Scripts\python -m pip install -e ." -ForegroundColor Red
        exit 1
    }
    Write-Host "Starting cockpit in a new window..." -ForegroundColor Yellow
    # Pass token through environment to the child process.
    $startCmd = "`$env:COCKPIT_REMOTE_TOKEN='$token'; & '$VenvPython' -m uvicorn packages.cockpit.web.server:app --host 127.0.0.1 --port 8000"
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-Command", $startCmd) -WorkingDirectory $RepoRoot | Out-Null

    Write-Host "Waiting for cockpit to come up..." -ForegroundColor DarkGray
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        if (Test-CockpitReachable -Url $CockpitUrl) { $ready = $true; break }
    }
    if (-not $ready) {
        Write-Host "Cockpit did not come up within 60s. Check the uvicorn window." -ForegroundColor Red
        exit 1
    }
    Write-Host "Cockpit is up at $CockpitUrl." -ForegroundColor Green
}

# --------------------------------------------------------------------
# 3. cloudflared
# --------------------------------------------------------------------
Write-Section "Step 3: cloudflared"
$cf = Install-Cloudflared

# --------------------------------------------------------------------
# 4. Tunnel
# --------------------------------------------------------------------
Write-Section "Step 4: Starting tunnel"
$logPath = Join-Path $env:TEMP "cockpit_tunnel.log"
if (Test-Path $logPath) { Remove-Item $logPath -Force }

$tunnelArgs = @("tunnel", "--url", $CockpitUrl, "--no-autoupdate", "--logfile", $logPath)
$proc = Start-Process -FilePath $cf -ArgumentList $tunnelArgs -PassThru -NoNewWindow

$publicUrl = $null
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    if (-not (Test-Path $logPath)) { continue }
    $logContent = Get-Content $logPath -Raw -ErrorAction SilentlyContinue
    if ($logContent -match "https://[a-z0-9-]+\.trycloudflare\.com") {
        $publicUrl = $matches[0]
        break
    }
}

if (-not $publicUrl) {
    Write-Host "Could not detect tunnel URL within 60s. Check $logPath." -ForegroundColor Red
    try { Stop-Process -Id $proc.Id -Force } catch {}
    exit 1
}

Write-Host "Tunnel URL: $publicUrl" -ForegroundColor Green

# --------------------------------------------------------------------
# 5. Write handle JSON
# --------------------------------------------------------------------
Write-Section "Step 5: Write handle"
New-Item -ItemType Directory -Force -Path $HandleDir | Out-Null
$handle = [ordered]@{
    url        = $publicUrl
    token      = $token
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    host_name  = $env:COMPUTERNAME
    repo_root  = $RepoRoot
}
$handleJson = $handle | ConvertTo-Json
Set-Content -Path $HandlePath -Value $handleJson -Encoding UTF8
Write-Host "Wrote $HandlePath" -ForegroundColor Green

# --------------------------------------------------------------------
# 6. Publish handle to GitHub branch (optional)
# --------------------------------------------------------------------
if (-not $NoPublish) {
    Write-Section "Step 6: Publish handle to GitHub"
    # Git writes progress lines to stderr; PowerShell's `Stop` policy treats
    # those as terminating errors. Temporarily relax it for this block and
    # rely on $LASTEXITCODE for real failure detection.
    $prevErrPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $RepoRoot
    $publishOk = $false
    try {
        $tmpWorktree = Join-Path $env:TEMP ("cockpit-handle-" + [Guid]::NewGuid().ToString("N"))
        $branchName = "cockpit-handle"

        & git worktree add --detach $tmpWorktree *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "git worktree add failed; skipping publish. Handle is local only." -ForegroundColor Yellow
        } else {
            Push-Location $tmpWorktree
            try {
                & git checkout --orphan $branchName *> $null
                & git rm -rf . *> $null
                $remoteHandleDir = Join-Path $tmpWorktree "data\cockpit"
                New-Item -ItemType Directory -Force -Path $remoteHandleDir | Out-Null
                Copy-Item $HandlePath (Join-Path $remoteHandleDir "remote_handle.json") -Force

                & git add data/cockpit/remote_handle.json *> $null
                $commitMsg = "cockpit: publish handle " + (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
                & git -c user.email=devfarinsky@gmail.com -c user.name="Devin Farinsky" commit -m $commitMsg *> $null
                # Orphan single-file state ref, rewritten every launch on purpose.
                & git push --force origin ("HEAD:refs/heads/" + $branchName) *> $null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "Published handle to origin/$branchName" -ForegroundColor Green
                    $publishOk = $true
                } else {
                    Write-Host "Publish to origin failed (exit=$LASTEXITCODE)." -ForegroundColor Yellow
                    Write-Host "Handle is still available locally at $HandlePath" -ForegroundColor Yellow
                }
            } finally {
                Pop-Location
            }
            & git worktree remove --force $tmpWorktree *> $null
        }
    } finally {
        Pop-Location
        $ErrorActionPreference = $prevErrPref
    }
}

# --------------------------------------------------------------------
# 7. Status block + block on tunnel
# --------------------------------------------------------------------
Write-Section "Cockpit + tunnel are LIVE"
Write-Host ""
Write-Host "Public URL : $publicUrl" -ForegroundColor Green
Write-Host "Token      : $token" -ForegroundColor Green
Write-Host "Handle     : $HandlePath" -ForegroundColor DarkGray
if (-not $NoPublish) {
    Write-Host "Published  : origin/cockpit-handle" -ForegroundColor DarkGray
}
Write-Host ""
Write-Host "The agent can now reach your cockpit." -ForegroundColor Yellow
Write-Host "You don't need to tell it the URL -- it will read the published handle." -ForegroundColor Yellow
Write-Host ""
Write-Host "Leave this window open while you want remote access." -ForegroundColor DarkGray
Write-Host "Press Ctrl+C to stop the tunnel." -ForegroundColor DarkGray
Write-Host ""

Wait-Process -Id $proc.Id
