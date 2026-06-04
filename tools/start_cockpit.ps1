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

.EXAMPLE
    PS> .\tools\start_cockpit.ps1
    Starts everything; agent auto-discovers via GitHub handle branch.
#>

[CmdletBinding()]
param(
    [switch]$NoPublish,
    [switch]$NewToken
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
    Push-Location $RepoRoot
    try {
        # Use a detached worktree so we don't touch your current branch state.
        $tmpWorktree = Join-Path $env:TEMP ("cockpit-handle-" + [Guid]::NewGuid().ToString("N"))
        $branchName = "cockpit-handle"

        # Make sure we have a fresh ref to push against (orphan-style).
        & git worktree add --detach $tmpWorktree 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "git worktree add failed; skipping publish. Handle is local only." -ForegroundColor Yellow
        } else {
            Push-Location $tmpWorktree
            try {
                # Build a single-file orphan commit.
                & git checkout --orphan $branchName 2>&1 | Out-Null
                & git rm -rf . 2>&1 | Out-Null
                $remoteHandleDir = Join-Path $tmpWorktree "data\cockpit"
                New-Item -ItemType Directory -Force -Path $remoteHandleDir | Out-Null
                Copy-Item $HandlePath (Join-Path $remoteHandleDir "remote_handle.json") -Force

                & git add data/cockpit/remote_handle.json 2>&1 | Out-Null
                & git -c user.email=devfarinsky@gmail.com -c user.name="Devin Farinsky" commit -m "cockpit: publish handle $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')" 2>&1 | Out-Null
                # NOTE: --force-with-lease isn't useful here because the branch
                # is an orphan single-file state ref that we intentionally rewrite
                # on every cockpit launch. We use plain -f and gate it behind the
                # -NoPublish opt-out flag for safety.
                $pushOut = & git push --force origin "HEAD:refs/heads/$branchName" 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "Published handle to origin/$branchName" -ForegroundColor Green
                } else {
                    Write-Host "Publish failed:" -ForegroundColor Yellow
                    Write-Host $pushOut -ForegroundColor DarkGray
                    Write-Host "Handle is still available locally at $HandlePath" -ForegroundColor Yellow
                }
            } finally {
                Pop-Location
            }
            # Clean up worktree.
            & git worktree remove --force $tmpWorktree 2>&1 | Out-Null
        }
    } finally {
        Pop-Location
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
