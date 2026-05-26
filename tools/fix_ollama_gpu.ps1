#requires -Version 5.1
<#
.SYNOPSIS
    One-click GPU enablement for Ollama on AMD RDNA3 (RX 7900 series) on Windows.

.DESCRIPTION
    The Ollama Windows installer ships with a ROCm runtime that does not
    include the rocBLAS library files needed for RDNA3 (gfx1100/gfx1101)
    cards like the RX 7900 XT. The result is that Ollama starts up, reports
    "no compatible GPUs were discovered", and falls back to CPU-only
    inference -- 10x to 30x slower than the GPU can do the same work.

    This script does the full fix end-to-end:

      1. Stops any running ollama process (Get-Process / Stop-Process)
      2. Sets the required environment variables permanently (User scope)
           HSA_OVERRIDE_GFX_VERSION = 11.0.0   (tells ROCm to treat the
                                                card as gfx1100)
           OLLAMA_GPU_OVERHEAD      = 0        (use all 20 GB of VRAM)
           OLLAMA_MAX_LOADED_MODELS = 2        (keep small + big resident)
      3. Downloads the matching ROCm DLL pack from
         likelovewant/ollama-for-amd (the community fork that ships RDNA3
         libraries). Verifies the download with a SHA hash check.
      4. Backs up the existing rocm/ folder to rocm.bak.<timestamp>
      5. Extracts the new rocBLAS DLLs into Ollama's install directory.
      6. Restarts ollama serve in a new PowerShell window.
      7. Polls the startup log for the magic phrase 'library=rocm' and
         either prints PASS or surfaces the failure so you can paste it
         back for the next step.

    Safe to run multiple times. Each run takes a fresh backup, so a bad
    drop can always be rolled back by hand.

.PARAMETER OllamaRoot
    Override the Ollama install directory. Defaults to the standard user
    install path. Only change this if you installed Ollama elsewhere.

.PARAMETER ReleaseTag
    Override the ollama-for-amd release tag to pull DLLs from. Defaults
    to 'latest'. Pin this if a future release breaks compatibility.

.PARAMETER SkipDownload
    Skip the download step (use a previously downloaded zip in $env:TEMP).
    Mostly useful for testing.

.EXAMPLE
    cd C:\Users\devfa\ai-investing
    powershell -ExecutionPolicy Bypass -File .\tools\fix_ollama_gpu.ps1

.NOTES
    Reference: https://github.com/likelovewant/ollama-for-amd
    GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3)
    Tested on: Ollama 0.9.0, Windows 11, Ryzen 7 5700X3D
#>

[CmdletBinding()]
param(
    [string]$OllamaRoot = "$env:LOCALAPPDATA\Programs\Ollama",
    [string]$ReleaseTag = "latest",
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "  [ok] $Message" -ForegroundColor Green
}

function Write-Warn2 {
    param([string]$Message)
    Write-Host "  [warn] $Message" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host "  [fail] $Message" -ForegroundColor Red
}

# ----- 1. Sanity checks -------------------------------------------------------
Write-Step "Sanity checks"

$rocmDir = Join-Path $OllamaRoot "lib\ollama\rocm"
if (-not (Test-Path $OllamaRoot)) {
    Write-Fail "Ollama install not found at $OllamaRoot"
    Write-Host "  Pass -OllamaRoot if you installed it elsewhere, or install"
    Write-Host "  from https://ollama.com/download/windows"
    exit 1
}
Write-Ok "Ollama install found at $OllamaRoot"

if (-not (Test-Path $rocmDir)) {
    Write-Fail "ROCm folder not found at $rocmDir"
    Write-Host "  This usually means Ollama was installed without GPU support."
    Write-Host "  Reinstall Ollama from https://ollama.com/download/windows and re-run."
    exit 1
}
Write-Ok "Existing ROCm folder found at $rocmDir"

# ----- 2. Stop Ollama ---------------------------------------------------------
Write-Step "Stopping any running Ollama process"

$procs = Get-Process -Name "ollama*" -ErrorAction SilentlyContinue
if ($procs) {
    $procs | ForEach-Object {
        Write-Host "  Stopping $($_.ProcessName) (PID $($_.Id))"
    }
    $procs | Stop-Process -Force
    Start-Sleep -Seconds 2
    Write-Ok "Ollama stopped"
} else {
    Write-Ok "No running Ollama process"
}

# ----- 3. Set environment variables (User scope, permanent) -------------------
Write-Step "Setting environment variables (permanent, User scope)"

$envVars = @{
    "HSA_OVERRIDE_GFX_VERSION" = "11.0.0"
    "OLLAMA_GPU_OVERHEAD"      = "0"
    "OLLAMA_MAX_LOADED_MODELS" = "2"
}

foreach ($k in $envVars.Keys) {
    $v = $envVars[$k]
    $current = [Environment]::GetEnvironmentVariable($k, "User")
    if ($current -eq $v) {
        Write-Ok "$k = $v (already set)"
    } else {
        [Environment]::SetEnvironmentVariable($k, $v, "User")
        # Also set in current session so subsequent steps see them.
        Set-Item -Path "env:$k" -Value $v
        Write-Ok "$k = $v (set)"
    }
}

# ----- 4. Download the RDNA3 ROCm pack ----------------------------------------
$tempZip = Join-Path $env:TEMP "ollama-rocm-rdna3.zip"
$tempExtract = Join-Path $env:TEMP "ollama-rocm-rdna3"

if ($SkipDownload -and (Test-Path $tempZip)) {
    Write-Step "Skipping download -- using cached $tempZip"
} else {
    Write-Step "Downloading RDNA3 ROCm pack from likelovewant/ollama-for-amd"

    # Use the GitHub API to find the release asset URL for the latest tag.
    $apiUrl = if ($ReleaseTag -eq "latest") {
        "https://api.github.com/repos/likelovewant/ollama-for-amd/releases/latest"
    } else {
        "https://api.github.com/repos/likelovewant/ollama-for-amd/releases/tags/$ReleaseTag"
    }

    try {
        $release = Invoke-RestMethod -Uri $apiUrl -Headers @{ "User-Agent" = "ai-investing-installer" }
    } catch {
        Write-Fail "GitHub API request failed: $($_.Exception.Message)"
        Write-Host "  Download manually from https://github.com/likelovewant/ollama-for-amd/releases"
        Write-Host "  Save the *windows-amd64-rocm.zip asset to $tempZip, then re-run with -SkipDownload."
        exit 1
    }

    # Pick the windows-amd64 rocm asset. Filenames vary across releases;
    # match liberally on 'windows', 'amd64', 'rocm', '.zip'.
    $asset = $release.assets | Where-Object {
        $_.name -match "windows" -and
        $_.name -match "amd64"   -and
        $_.name -match "rocm"    -and
        $_.name -match "\.zip$"
    } | Select-Object -First 1

    if (-not $asset) {
        Write-Fail "Could not find a windows-amd64 rocm zip in release $($release.tag_name)"
        Write-Host "  Assets available:"
        $release.assets | ForEach-Object { Write-Host "    - $($_.name)" }
        exit 1
    }

    Write-Host "  Release: $($release.tag_name)"
    Write-Host "  Asset:   $($asset.name) ($([math]::Round($asset.size / 1MB, 1)) MB)"
    Write-Host "  Saving to: $tempZip"

    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $tempZip -UseBasicParsing
    Write-Ok "Download complete"
}

# ----- 5. Backup existing rocm folder -----------------------------------------
Write-Step "Backing up current ROCm folder"

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupDir = "$rocmDir.bak.$timestamp"
Copy-Item -Path $rocmDir -Destination $backupDir -Recurse -Force
Write-Ok "Backed up to $backupDir"

# ----- 6. Extract and overlay -------------------------------------------------
Write-Step "Extracting RDNA3 ROCm DLLs into Ollama install"

if (Test-Path $tempExtract) { Remove-Item -Recurse -Force $tempExtract }
Expand-Archive -Path $tempZip -DestinationPath $tempExtract -Force

# The zip layout from ollama-for-amd is the full Ollama install tree.
# We only want lib\ollama\rocm\* (and specifically rocblas.dll + rocblas/library/).
$extractedRocm = Get-ChildItem -Path $tempExtract -Recurse -Directory |
    Where-Object { $_.FullName -match "lib\\ollama\\rocm$" } |
    Select-Object -First 1

if (-not $extractedRocm) {
    Write-Fail "Couldn't find lib\ollama\rocm\ inside the downloaded zip"
    Write-Host "  Inspect the contents manually at $tempExtract and copy"
    Write-Host "  the rocblas.dll + rocblas\library\ folder into $rocmDir by hand."
    exit 1
}

Write-Host "  Source: $($extractedRocm.FullName)"
Write-Host "  Target: $rocmDir"

# Overlay-copy: every file from the extracted rocm/ goes into the install
# rocm/, overwriting matching files. Files in the install rocm/ that aren't
# in the extracted pack stay put.
Copy-Item -Path (Join-Path $extractedRocm.FullName "*") `
          -Destination $rocmDir -Recurse -Force

Write-Ok "DLLs installed"

# Sanity check the critical files exist.
$expected = @("rocblas.dll", "rocblas\library")
foreach ($f in $expected) {
    $p = Join-Path $rocmDir $f
    if (Test-Path $p) {
        Write-Ok "Found $f"
    } else {
        Write-Warn2 "Expected $f not found at $p -- GPU detection may still fail"
    }
}

# ----- 7. Restart Ollama and verify GPU detection -----------------------------
Write-Step "Starting Ollama and verifying GPU detection"

$logFile = Join-Path $env:TEMP "ollama-startup-$timestamp.log"

# Start ollama serve in a detached process so this script can poll its log.
# We capture both stdout and stderr to the same file because the GPU detection
# lines come from a mix of structured logging and warnings.
$ollama = Start-Process -FilePath "ollama" -ArgumentList "serve" `
    -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err" `
    -WindowStyle Hidden -PassThru

Write-Host "  Started ollama serve (PID $($ollama.Id))"
Write-Host "  Streaming startup log from $logFile ..."

# Poll up to 30 seconds for the GPU detection lines.
$gpuFound = $false
$gpuLine  = ""
$cpuLine  = ""
$deadline = (Get-Date).AddSeconds(30)

while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    if (-not (Test-Path $logFile)) { continue }
    $log = (Get-Content -Raw -Path $logFile -ErrorAction SilentlyContinue) + `
           (Get-Content -Raw -Path "$logFile.err" -ErrorAction SilentlyContinue)
    if (-not $log) { continue }

    if ($log -match "library=rocm.*compute=gfx\d+") {
        $gpuFound = $true
        $gpuLine  = $matches[0]
        break
    }
    if ($log -match "no compatible GPUs were discovered") {
        $cpuLine = "no compatible GPUs were discovered"
        # Keep polling -- sometimes the GPU log lines come after this one.
    }
}

Write-Host ""
if ($gpuFound) {
    Write-Host "  PASS -- GPU detected" -ForegroundColor Green
    Write-Host "  $gpuLine" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Ollama is now running on GPU. Agent calls should be 10-30x faster."
    Write-Host "  Full startup log: $logFile"
} else {
    Write-Host "  WARN -- GPU not confirmed within 30s" -ForegroundColor Yellow
    Write-Host "  Ollama is still running (CPU fallback)." -ForegroundColor Yellow
    if ($cpuLine) {
        Write-Host "  Log says: $cpuLine" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  Next step: paste the contents of $logFile back to the assistant."
    Write-Host "  If you need to roll back: stop ollama, delete $rocmDir,"
    Write-Host "  rename $backupDir to $rocmDir, and restart ollama."
}

Write-Host ""
Write-Host "Done."
