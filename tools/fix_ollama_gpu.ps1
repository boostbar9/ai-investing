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
    Skip the download step (use a previously downloaded archive in
    $env:TEMP -- either ollama-rocm-rdna3.7z or ollama-rocm-rdna3.zip).
    Mostly useful for testing.

.NOTES
    Asset format compatibility:
      * Older releases (<=v0.16.1) shipped a separate *-windows-amd64-rocm.zip.
      * Newer releases (v0.18.2+) ship a single ollama-windows-amd64.7z
        that contains the same lib\ollama\rocm\ tree.
    This script prefers the rocm.zip if present, falls back to the .7z,
    and downloads the standalone 7zr.exe extractor on demand if 7-Zip
    isn't installed locally.

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

# IWR + write-progress in a piped/headless host throttles downloads to a
# crawl (sometimes <50 KB/s). Killing the progress preference here makes
# Invoke-WebRequest run at full speed even when we DO use it as a fallback.
$ProgressPreference = "SilentlyContinue"

function Invoke-FastDownload {
    <#
      Headless-friendly downloader with live progress.

      Streams the response to disk in 256 KB chunks and emits a single
      progress line every ~2 s to stdout so the cockpit log shows the
      download moving. PowerShell's IWR with -UseBasicParsing buffers
      the entire body in memory and emits nothing to stdout, which is
      why earlier runs of this script went silent for minutes on a 73 MB
      download.
    #>
    param(
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)][string]$OutFile
    )
    Add-Type -AssemblyName System.Net.Http
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.AutomaticDecompression = [System.Net.DecompressionMethods]::GZip -bor [System.Net.DecompressionMethods]::Deflate
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromMinutes(30)
    $client.DefaultRequestHeaders.UserAgent.ParseAdd("ai-investing-installer/1.0")
    try {
        $resp = $client.GetAsync($Uri, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        if (-not $resp.IsSuccessStatusCode) {
            throw "HTTP $([int]$resp.StatusCode) $($resp.ReasonPhrase) for $Uri"
        }
        $total = $resp.Content.Headers.ContentLength
        $totalMB = if ($total) { [math]::Round($total / 1MB, 1) } else { $null }
        $inStream = $resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $outStream = [System.IO.File]::Open($OutFile, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
        try {
            $buf = New-Object byte[] 262144  # 256 KB chunks
            $read = 0
            [long]$bytes = 0
            $lastReport = [DateTime]::UtcNow
            $startedAt  = [DateTime]::UtcNow
            while (($read = $inStream.Read($buf, 0, $buf.Length)) -gt 0) {
                $outStream.Write($buf, 0, $read)
                $bytes += $read
                $now = [DateTime]::UtcNow
                if (($now - $lastReport).TotalSeconds -ge 2) {
                    $mb = [math]::Round($bytes / 1MB, 1)
                    $elapsed = ($now - $startedAt).TotalSeconds
                    $rate = if ($elapsed -gt 0) { [math]::Round(($bytes / 1MB) / $elapsed, 2) } else { 0 }
                    if ($totalMB) {
                        $pct = [math]::Round(($bytes / [double]$total) * 100, 1)
                        Write-Host ("  [dl] {0,5:N1} / {1,5:N1} MB ({2,4:N1} %) @ {3,5:N2} MB/s" -f $mb, $totalMB, $pct, $rate)
                    } else {
                        Write-Host ("  [dl] {0,5:N1} MB @ {1,5:N2} MB/s" -f $mb, $rate)
                    }
                    $lastReport = $now
                }
            }
            $totalElapsed = ([DateTime]::UtcNow - $startedAt).TotalSeconds
            $finalMB = [math]::Round($bytes / 1MB, 1)
            $finalRate = if ($totalElapsed -gt 0) { [math]::Round(($bytes / 1MB) / $totalElapsed, 2) } else { 0 }
            Write-Host ("  [dl] done: {0:N1} MB in {1:N1}s ({2:N2} MB/s avg)" -f $finalMB, $totalElapsed, $finalRate)
        } finally {
            $outStream.Dispose()
            $inStream.Dispose()
        }
    } finally {
        $client.Dispose()
        $handler.Dispose()
    }
}

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
# Newer likelovewant/ollama-for-amd releases (v0.18.2+) ship a single
# `ollama-windows-amd64.7z` instead of the old split `*-rocm.zip`. We
# accept either, and pick the extractor accordingly further down.
$tempExtract = Join-Path $env:TEMP "ollama-rocm-rdna3"
$tempArchive = $null
$tempArchiveExt = $null

# Detect a cached archive from a previous run so -SkipDownload works for
# either format.
foreach ($ext in ".7z", ".zip") {
    $candidate = Join-Path $env:TEMP "ollama-rocm-rdna3$ext"
    if ($SkipDownload -and (Test-Path $candidate)) {
        $tempArchive = $candidate
        $tempArchiveExt = $ext
        Write-Step "Skipping download -- using cached $candidate"
        break
    }
}

if (-not $tempArchive) {
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
        Write-Host "  Save the ollama-windows-amd64.7z (or *-rocm.zip) asset to"
        Write-Host "  $env:TEMP\ollama-rocm-rdna3.7z, then re-run with -SkipDownload."
        exit 1
    }

    # Prefer a separate rocm zip (older layout). If absent, fall back to
    # the consolidated windows-amd64 7z which contains lib\ollama\rocm\.
    # Match liberally because filenames have drifted across releases.
    $asset = $release.assets | Where-Object {
        $_.name -match "windows" -and
        $_.name -match "amd64"   -and
        $_.name -match "rocm"    -and
        $_.name -match "\.zip$"
    } | Select-Object -First 1

    if (-not $asset) {
        $asset = $release.assets | Where-Object {
            $_.name -match "windows" -and
            $_.name -match "amd64"   -and
            $_.name -match "\.7z$"
        } | Select-Object -First 1
    }

    if (-not $asset) {
        Write-Fail "Could not find a windows-amd64 ROCm archive in release $($release.tag_name)"
        Write-Host "  Looked for *windows*amd64*rocm*.zip or *windows*amd64*.7z"
        Write-Host "  Assets available:"
        $release.assets | ForEach-Object { Write-Host "    - $($_.name)" }
        exit 1
    }

    $tempArchiveExt = if ($asset.name -match "\.7z$") { ".7z" } else { ".zip" }
    $tempArchive = Join-Path $env:TEMP "ollama-rocm-rdna3$tempArchiveExt"

    Write-Host "  Release: $($release.tag_name)"
    Write-Host "  Asset:   $($asset.name) ($([math]::Round($asset.size / 1MB, 1)) MB)"
    Write-Host "  Saving to: $tempArchive"

    try {
        Invoke-FastDownload -Uri $asset.browser_download_url -OutFile $tempArchive
    } catch {
        Write-Fail "Download failed: $($_.Exception.Message)"
        Write-Host "  Retry the cockpit Fix GPU button, or download manually from"
        Write-Host "  https://github.com/likelovewant/ollama-for-amd/releases"
        Write-Host "  to $tempArchive and re-run with -SkipDownload."
        exit 1
    }
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
New-Item -ItemType Directory -Path $tempExtract -Force | Out-Null

if ($tempArchiveExt -eq ".zip") {
    Expand-Archive -Path $tempArchive -DestinationPath $tempExtract -Force
} else {
    # 7z extraction. PowerShell has no native 7z support, so we try a
    # short chain of options: installed 7z.exe (most dev boxes have one),
    # then a cached 7zr.exe in $env:TEMP, then we download the official
    # standalone 7zr.exe (~600 KB) from 7-zip.org. 7zr handles 7z files
    # without needing the full installer.
    function Resolve-SevenZip {
        $candidates = @(
            "$env:ProgramFiles\7-Zip\7z.exe",
            "$env:ProgramFiles(x86)\7-Zip\7z.exe",
            "$env:TEMP\7zr.exe"
        )
        foreach ($p in $candidates) {
            if ($p -and (Test-Path $p)) { return $p }
        }
        # System PATH (covers chocolatey/scoop installs).
        $cmd = Get-Command 7z.exe -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        $cmd = Get-Command 7zr.exe -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        return $null
    }

    $sevenZip = Resolve-SevenZip
    if (-not $sevenZip) {
        Write-Host "  No 7-Zip found locally -- downloading standalone 7zr.exe (~600 KB)"
        $sevenZip = Join-Path $env:TEMP "7zr.exe"
        try {
            Invoke-FastDownload -Uri "https://www.7-zip.org/a/7zr.exe" -OutFile $sevenZip
            Write-Ok "Fetched 7zr.exe"
        } catch {
            Write-Fail "Couldn't fetch 7zr.exe: $($_.Exception.Message)"
            Write-Host "  Install 7-Zip from https://www.7-zip.org/ and re-run with -SkipDownload,"
            Write-Host "  or extract $tempArchive manually into $tempExtract and re-run with -SkipDownload."
            exit 1
        }
    }

    Write-Host "  Using 7-Zip at: $sevenZip"
    Write-Host "  Extracting $tempArchive -> $tempExtract"
    # -y: assume Yes on prompts; -o (no space) sets the output dir.
    & $sevenZip x $tempArchive "-o$tempExtract" -y | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "7-Zip extraction failed with exit code $LASTEXITCODE"
        exit 1
    }
}

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

# RDNA3-specific: rocblas needs the gfx1100 tensor files. The consolidated
# v0.30.0 7z usually ships them but the standard upstream Ollama 7z does
# not. If gfx1100 is missing, GPU detection silently falls back to CPU.
$libDir = Join-Path $rocmDir "rocblas\library"
if (Test-Path $libDir) {
    $gfx1100Files = Get-ChildItem -Path $libDir -Filter "*gfx1100*" -ErrorAction SilentlyContinue
    if ($gfx1100Files -and $gfx1100Files.Count -gt 0) {
        Write-Ok ("Found {0} gfx1100 tensor file(s) for RDNA3" -f $gfx1100Files.Count)
    } else {
        Write-Warn2 "No gfx1100 tensor files in rocblas\library -- this is the most common"
        Write-Host  "           cause of 'no compatible GPUs' on RX 7900 XT. The consolidated"
        Write-Host  "           upstream 7z dropped them; grab the RDNA3-specific pack from"
        Write-Host  "           https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU"
        Write-Host  "           or an older likelovewant/ollama-for-amd release (<=v0.16.1)."
    }
}

# ----- 7. Restart Ollama and verify GPU detection -----------------------------
Write-Step "Starting Ollama and verifying GPU detection"

$logFile = Join-Path $env:TEMP "ollama-startup-$timestamp.log"
$errFile = "$logFile.err"

# Start ollama serve in a detached process so this script can poll its log.
# We capture both stdout and stderr to the same file because the GPU detection
# lines come from a mix of structured logging and warnings.
$ollama = Start-Process -FilePath "ollama" -ArgumentList "serve" `
    -RedirectStandardOutput $logFile -RedirectStandardError $errFile `
    -WindowStyle Hidden -PassThru

Write-Host "  Started ollama serve (PID $($ollama.Id))"
Write-Host "  Streaming startup log from $logFile ..."

# Helper to read a file even while another process holds a write handle.
# Get-Content with -ErrorAction SilentlyContinue silently returns nothing
# when the file is locked by ollama, which is why the previous run's log
# capture appeared empty even though ollama was writing to it.
function Read-LockedFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return "" }
    try {
        $fs = [System.IO.File]::Open(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite
        )
        try {
            $sr = New-Object System.IO.StreamReader($fs)
            return $sr.ReadToEnd()
        } finally {
            $fs.Dispose()
        }
    } catch {
        return ""
    }
}

# Poll up to 30 seconds for the GPU detection lines.
$gpuFound = $false
$gpuLine  = ""
$cpuLine  = ""
$deadline = (Get-Date).AddSeconds(30)

while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    $log = (Read-LockedFile -Path $logFile) + "`n" + (Read-LockedFile -Path $errFile)
    if (-not $log.Trim()) { continue }

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
    Write-Host "  === Last 60 lines of the Ollama startup log ===" -ForegroundColor Yellow
    $finalLog = (Read-LockedFile -Path $logFile) + "`n" + (Read-LockedFile -Path $errFile)
    $finalLines = $finalLog -split "`r?`n" | Where-Object { $_.Trim() }
    $tailCount = [Math]::Min(60, $finalLines.Count)
    if ($tailCount -gt 0) {
        $finalLines | Select-Object -Last $tailCount | ForEach-Object {
            Write-Host "    $_"
        }
    } else {
        Write-Host "    (the startup log was empty -- ollama may have failed to start)"
        Write-Host "    Check that the 'ollama' binary is on PATH (try 'where.exe ollama')"
    }
    Write-Host ""
    Write-Host "  === Local GPU environment ===" -ForegroundColor Yellow
    Write-Host ("    HSA_OVERRIDE_GFX_VERSION = {0}" -f ([Environment]::GetEnvironmentVariable("HSA_OVERRIDE_GFX_VERSION", "User")))
    Write-Host ("    HIP_VISIBLE_DEVICES      = {0}" -f ([Environment]::GetEnvironmentVariable("HIP_VISIBLE_DEVICES", "User")))
    Write-Host ("    OLLAMA_VISIBLE_DEVICES   = {0}" -f ([Environment]::GetEnvironmentVariable("OLLAMA_VISIBLE_DEVICES", "User")))
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
            Select-Object Name, DriverVersion, Status
        if ($gpus) {
            Write-Host "    Display adapters detected:"
            $gpus | ForEach-Object {
                Write-Host ("      - {0}  driver={1}  status={2}" -f $_.Name, $_.DriverVersion, $_.Status)
            }
            $parsec = $gpus | Where-Object { $_.Name -match "Parsec|Virtual|Remote" }
            if ($parsec) {
                Write-Warn2 "A virtual/remote display adapter was detected. On some setups Ollama"
                Write-Host  "           picks up the virtual adapter first and skips the real GPU."
                Write-Host  "           Try setting HIP_VISIBLE_DEVICES=0 (or 1) to force the real GPU:"
                Write-Host  "             [Environment]::SetEnvironmentVariable('HIP_VISIBLE_DEVICES','0','User')"
                Write-Host  "           then run this script again."
            }
        }
    } catch {}
    Write-Host ""
    Write-Host "  Full startup log: $logFile"
    Write-Host "  If you need to roll back: stop ollama, delete $rocmDir,"
    Write-Host "  rename $backupDir to $rocmDir, and restart ollama."
}

Write-Host ""
Write-Host "Done."
