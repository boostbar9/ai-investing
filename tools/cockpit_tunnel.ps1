#requires -Version 5.1
<#
.SYNOPSIS
    One-click Cloudflare tunnel launcher for the ai-investing cockpit.

.DESCRIPTION
    Phase 36c -- exposes your locally running cockpit (default
    http://127.0.0.1:8000) to a public *.trycloudflare.com URL using
    Cloudflare's free quick-tunnel feature (no Cloudflare account
    required for testing). Designed for the Windows PC that runs the
    cockpit so the Perplexity agent can reach the /api/remote/*
    surface.

    What this script does, in order:

      1. Verifies the cockpit is reachable at -CockpitUrl.
      2. Verifies COCKPIT_REMOTE_TOKEN is set (or generates one).
      3. Ensures cloudflared.exe is installed (auto-downloads to
         tools\bin\ if missing).
      4. Starts the quick-tunnel and parses the assigned public URL.
      5. Prints a single block with:
            * the public URL
            * the auth token
            * a curl one-liner to verify
            * a short instruction for the agent.

.PARAMETER CockpitUrl
    Local URL where the cockpit is listening. Default
    http://127.0.0.1:8000.

.PARAMETER GenerateToken
    Generate and persist a new COCKPIT_REMOTE_TOKEN. Default: only
    generate if one is not already set in the current session.

.EXAMPLE
    PS> .\tools\cockpit_tunnel.ps1
    Starts the tunnel, reusing any existing token.

.EXAMPLE
    PS> .\tools\cockpit_tunnel.ps1 -GenerateToken
    Rotates the remote token and starts a fresh tunnel.

.NOTES
    Quick tunnels (*.trycloudflare.com) are ephemeral -- the URL
    changes every restart. For a stable hostname, log into Cloudflare
    and create a named tunnel; the script will use it automatically if
    CLOUDFLARED_TUNNEL_NAME is set in the environment.
#>

[CmdletBinding()]
param(
    [string]$CockpitUrl = "http://127.0.0.1:8000",
    [switch]$GenerateToken
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BinDir = Join-Path $RepoRoot "tools\bin"
$CloudflaredExe = Join-Path $BinDir "cloudflared.exe"

function Write-Section($text) {
    Write-Host ""
    Write-Host "=== $text ===" -ForegroundColor Cyan
}

function Test-CockpitReachable {
    param([string]$Url)
    try {
        $r = Invoke-WebRequest -Uri "$Url/api/remote/health" -UseBasicParsing -TimeoutSec 5
        if ($r.StatusCode -eq 200) { return $true }
    } catch {
        return $false
    }
    return $false
}

function New-RemoteToken {
    # 32 hex chars = 128 bits. Generated via .NET RNG, not Math.Random.
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
# 1. Cockpit reachability
# --------------------------------------------------------------------
Write-Section "Step 1: Verify cockpit is running"
if (-not (Test-CockpitReachable -Url $CockpitUrl)) {
    Write-Host "Cockpit is NOT reachable at $CockpitUrl." -ForegroundColor Red
    Write-Host "Start the cockpit first (e.g. .\.venv\Scripts\python.exe -m uvicorn packages.cockpit.web.server:app)." -ForegroundColor Red
    exit 1
}
Write-Host "Cockpit reachable at $CockpitUrl" -ForegroundColor Green

# --------------------------------------------------------------------
# 2. Token management
# --------------------------------------------------------------------
Write-Section "Step 2: Auth token"
$existing = $env:COCKPIT_REMOTE_TOKEN
if ($GenerateToken -or [string]::IsNullOrWhiteSpace($existing) -or $existing.Length -lt 16) {
    $token = New-RemoteToken
    $env:COCKPIT_REMOTE_TOKEN = $token
    # Persist for future shells on this machine.
    [Environment]::SetEnvironmentVariable("COCKPIT_REMOTE_TOKEN", $token, "User")
    Write-Host "Generated NEW remote token and saved to User env." -ForegroundColor Green
    Write-Host "Restart the cockpit so it picks up the new token." -ForegroundColor Yellow
} else {
    $token = $existing
    Write-Host "Using existing COCKPIT_REMOTE_TOKEN (length=$($token.Length))" -ForegroundColor Green
}

# --------------------------------------------------------------------
# 3. cloudflared install
# --------------------------------------------------------------------
Write-Section "Step 3: cloudflared"
$cf = Install-Cloudflared

# --------------------------------------------------------------------
# 4. Launch tunnel
# --------------------------------------------------------------------
Write-Section "Step 4: Starting tunnel"
$namedTunnel = $env:CLOUDFLARED_TUNNEL_NAME
if (-not [string]::IsNullOrWhiteSpace($namedTunnel)) {
    Write-Host "Using named tunnel: $namedTunnel" -ForegroundColor Green
    & $cf tunnel run $namedTunnel
    exit $LASTEXITCODE
}

Write-Host "Launching ephemeral quick-tunnel. URL will appear below." -ForegroundColor Yellow
Write-Host "(For a stable hostname, set CLOUDFLARED_TUNNEL_NAME after creating a named tunnel.)" -ForegroundColor DarkGray

# Run cloudflared and tee its output so we can pull the public URL.
$logPath = Join-Path $env:TEMP "cockpit_tunnel.log"
if (Test-Path $logPath) { Remove-Item $logPath -Force }

$proc = Start-Process -FilePath $cf `
    -ArgumentList @("tunnel", "--url", $CockpitUrl, "--no-autoupdate", "--logfile", $logPath) `
    -PassThru -NoNewWindow

# Poll the log for the assigned URL. Cloudflared prints something like:
#   "Your quick Tunnel has been created! Visit it at:"
#   "https://random-name.trycloudflare.com"
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
    Write-Host "Could not detect tunnel URL within 60s. Check $logPath for details." -ForegroundColor Red
    exit 1
}

# --------------------------------------------------------------------
# 5. Print the connection block
# --------------------------------------------------------------------
Write-Section "Tunnel is LIVE"
Write-Host ""
Write-Host "Public URL : $publicUrl" -ForegroundColor Green
Write-Host "Token      : $token" -ForegroundColor Green
Write-Host ""
Write-Host "Verify with:" -ForegroundColor Yellow
$dq = [char]34
$verifyCmd = "  curl -H ${dq}Authorization: Bearer ${token}${dq} ${publicUrl}/api/remote/whoami"
Write-Host $verifyCmd
Write-Host ""
Write-Host "Tell the agent:" -ForegroundColor Yellow
Write-Host "  My cockpit is at $publicUrl with token $token"
Write-Host ""
Write-Host "Leave this window open while you want remote access." -ForegroundColor DarkGray
Write-Host "Press Ctrl+C to stop the tunnel." -ForegroundColor DarkGray

# Block until cloudflared exits.
Wait-Process -Id $proc.Id
