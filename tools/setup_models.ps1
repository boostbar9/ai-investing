#requires -Version 5.1
<#
.SYNOPSIS
    One-command setup of all LLMs needed for the ai-investing soak.

.DESCRIPTION
    Does the full LLM provisioning in one shot:

      1. Picks the right hardware profile for this box (default
         rx_7900_xt; override with -Profile)
      2. Writes HARDWARE_PROFILE=<profile> to .env if missing
      3. Asks Python (via the repo's own model_profiles.py) what models
         that profile needs
      4. Asks Ollama what's already installed
      5. Pulls every missing model from ollama.com, streaming progress
         to the console
      6. Optionally removes obsolete / unused models to free disk
         (only with -Cleanup; never removes anything the profile needs)
      7. Verifies the final inventory and prints a green PASS

    Designed to be fire-and-forget. Run it, walk away. Pulls happen
    sequentially so the GPU isn't thrashed; total wall time is
    bandwidth-bound (~40 GB for rx_7900_xt over a 100 Mbps line is
    roughly an hour).

.PARAMETER Profile
    Hardware profile to provision. Defaults to rx_7900_xt (Devin's
    box). Valid values: cpu_only, balanced, rx_7900_xt, high_end,
    workstation.

.PARAMETER Cleanup
    Also remove models that are NOT needed by the chosen profile.
    Frees disk but irreversible (you'd re-pull on demand). Off by
    default; pass -Cleanup to enable.

.PARAMETER RepoRoot
    Override the repo path. Defaults to C:\Users\devfa\ai-investing.

.EXAMPLE
    # Most common usage \u2014 set up the rx_7900_xt profile and pull what's missing.
    cd C:\Users\devfa\ai-investing
    powershell -ExecutionPolicy Bypass -File .\tools\setup_models.ps1

.EXAMPLE
    # Same, but also remove unused models afterward.
    powershell -ExecutionPolicy Bypass -File .\tools\setup_models.ps1 -Cleanup

.EXAMPLE
    # Force CPU-only profile (smaller pulls, ~7 GB total).
    powershell -ExecutionPolicy Bypass -File .\tools\setup_models.ps1 -Profile cpu_only
#>

[CmdletBinding()]
param(
    [ValidateSet("cpu_only", "balanced", "rx_7900_xt", "high_end", "workstation")]
    [string]$Profile = "rx_7900_xt",
    [switch]$Cleanup,
    [string]$RepoRoot = "C:\Users\devfa\ai-investing"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Write-Ok    { param([string]$M) Write-Host "  [ok] $M"   -ForegroundColor Green  }
function Write-Warn2 { param([string]$M) Write-Host "  [warn] $M" -ForegroundColor Yellow }
function Write-Fail  { param([string]$M) Write-Host "  [fail] $M" -ForegroundColor Red    }

# ----- 1. Sanity checks ------------------------------------------------------
Write-Step "Sanity checks"

if (-not (Test-Path $RepoRoot)) {
    Write-Fail "Repo not found at $RepoRoot"
    Write-Host "  Pass -RepoRoot if your install lives elsewhere."
    exit 1
}
Write-Ok "Repo at $RepoRoot"

# Ollama must be running for /api/tags + ollama pull to work.
try {
    $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
    Write-Ok "Ollama responding ($($tags.models.Count) model(s) installed)"
} catch {
    Write-Fail "Ollama is not reachable on http://127.0.0.1:11434"
    Write-Host "  Start it first: open a new PowerShell and run  ollama serve"
    Write-Host "  Or click the Ollama icon in the system tray."
    exit 1
}

# Python venv we'll use to ask model_profiles.py what's required.
$pythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Fail "Python venv not found at $pythonExe"
    Write-Host "  Run dev.ps1 once to bootstrap the venv, then re-run this script."
    exit 1
}
Write-Ok "Python venv at $pythonExe"

# ----- 2. Write HARDWARE_PROFILE into .env -----------------------------------
Write-Step "Setting HARDWARE_PROFILE=$Profile in .env"

$envFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Fail ".env not found at $envFile"
    Write-Host "  Copy .env.example to .env and re-run."
    exit 1
}

$existing = Get-Content $envFile
$hasLine  = $existing | Select-String -Pattern "^HARDWARE_PROFILE=" -Quiet
$correct  = $existing | Select-String -Pattern "^HARDWARE_PROFILE=$Profile\s*$" -Quiet

if ($correct) {
    Write-Ok "HARDWARE_PROFILE=$Profile already set"
} elseif ($hasLine) {
    # Replace the existing line.
    $new = $existing -replace "^HARDWARE_PROFILE=.*$", "HARDWARE_PROFILE=$Profile"
    Set-Content -Path $envFile -Value $new
    Write-Ok "Updated HARDWARE_PROFILE -> $Profile"
} else {
    Add-Content -Path $envFile -Value "`nHARDWARE_PROFILE=$Profile"
    Write-Ok "Added HARDWARE_PROFILE=$Profile"
}

# ----- 3. Ask the repo what models the profile needs -------------------------
Write-Step "Resolving required models from the active profile"

Push-Location $RepoRoot
try {
    $env:PYTHONPATH    = "."
    $env:HARDWARE_PROFILE = $Profile
    $requiredJson = & $pythonExe -c @"
import json, os
from packages.agents.model_profiles import active_profile, all_models
p = active_profile()
print(json.dumps({'profile': p.name, 'description': p.description, 'models': all_models(p)}))
"@
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Failed to resolve profile via Python"
        Write-Host $requiredJson
        exit 1
    }
} finally {
    Pop-Location
}

$resolved = $requiredJson | ConvertFrom-Json
$required = $resolved.models
Write-Ok "Profile: $($resolved.profile)"
Write-Host "  $($resolved.description)"
Write-Host "  Required models ($($required.Count)):"
foreach ($m in $required) { Write-Host "    - $m" }

# ----- 4. Diff against what's installed --------------------------------------
Write-Step "Comparing against installed inventory"

$installed = @($tags.models | ForEach-Object { $_.name })
Write-Host "  Installed ($($installed.Count)):"
foreach ($m in $installed) { Write-Host "    - $m" }

function Test-ModelInstalled {
    # Mirror packages/agents/llm_router.installed_matches \u2014 a declared
    # model is satisfied by any installed tag that either matches exactly,
    # matches its bare base name, or differs only by quant suffix.
    param([string]$Required, [string[]]$Installed)
    if ($Installed -contains $Required) { return $true }
    $base = ($Required -split ":", 2)[0]
    $targetTag = if ($Required -match ":") { ($Required -split ":", 2)[1] } else { "" }
    foreach ($tag in $Installed) {
        if ($tag -eq $base) { return $true }
        if (-not $tag.StartsWith("$base`:")) { continue }
        $installedTag = if ($tag -match ":") { ($tag -split ":", 2)[1] } else { "" }
        if (-not $targetTag) { return $true }
        if ($installedTag.StartsWith($targetTag) -or $targetTag.StartsWith($installedTag)) {
            return $true
        }
    }
    return $false
}

$missing = @()
foreach ($m in $required) {
    if (-not (Test-ModelInstalled -Required $m -Installed $installed)) {
        $missing += $m
    }
}

if ($missing.Count -eq 0) {
    Write-Ok "Every required model is already installed"
} else {
    Write-Host ""
    Write-Host "  Missing ($($missing.Count)):" -ForegroundColor Yellow
    foreach ($m in $missing) { Write-Host "    - $m" -ForegroundColor Yellow }
}

# ----- 5. Pull missing models ------------------------------------------------
if ($missing.Count -gt 0) {
    Write-Step "Pulling missing models (this is the slow part)"
    Write-Host "  You can leave this window open and walk away \u2014 the cockpit"
    Write-Host "  keeps running independently."
    Write-Host ""

    $idx = 0
    foreach ($model in $missing) {
        $idx++
        Write-Host "  [$idx/$($missing.Count)] ollama pull $model" -ForegroundColor Cyan
        $started = Get-Date
        & ollama pull $model
        $code = $LASTEXITCODE
        $elapsed = (Get-Date) - $started
        if ($code -eq 0) {
            Write-Ok "$model pulled in $([math]::Round($elapsed.TotalMinutes, 1)) min"
        } else {
            Write-Warn2 "$model exited with code $code \u2014 continuing with next model"
        }
    }
}

# ----- 6. Optional cleanup of models not needed by this profile --------------
if ($Cleanup) {
    Write-Step "Removing models not used by profile '$Profile'"

    # Refresh inventory \u2014 the pulls above changed it.
    try {
        $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
        $installed = @($tags.models | ForEach-Object { $_.name })
    } catch {
        Write-Warn2 "Could not refresh inventory \u2014 skipping cleanup"
        $installed = @()
    }

    # A model is "needed" if any required model matches it under the
    # liberal-match rules. Anything else is fair game to remove.
    $needed = @()
    foreach ($req in $required) {
        foreach ($tag in $installed) {
            if (Test-ModelInstalled -Required $req -Installed @($tag)) {
                $needed += $tag
            }
        }
    }
    $needed = $needed | Select-Object -Unique
    $unneeded = $installed | Where-Object { $_ -notin $needed }

    if ($unneeded.Count -eq 0) {
        Write-Ok "Nothing to clean up"
    } else {
        Write-Host "  Will remove:"
        foreach ($m in $unneeded) { Write-Host "    - $m" }
        foreach ($m in $unneeded) {
            Write-Host "  ollama rm $m"
            & ollama rm $m
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "removed $m"
            } else {
                Write-Warn2 "could not remove $m (exit $LASTEXITCODE)"
            }
        }
    }
}

# ----- 7. Verify final state -------------------------------------------------
Write-Step "Verifying final inventory"

try {
    $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
    $installed = @($tags.models | ForEach-Object { $_.name })
} catch {
    Write-Warn2 "Could not verify \u2014 Ollama not responding"
    exit 1
}

$stillMissing = @()
foreach ($m in $required) {
    if (-not (Test-ModelInstalled -Required $m -Installed $installed)) {
        $stillMissing += $m
    }
}

Write-Host ""
if ($stillMissing.Count -eq 0) {
    Write-Host "  PASS \u2014 every required model for profile '$Profile' is installed" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Restart the cockpit (Ctrl+C, then .\dev.ps1) so it picks up the new profile."
    Write-Host "  2. Open http://127.0.0.1:8765/health \u2014 'Agent models pulled' should be green."
    Write-Host "  3. Open http://127.0.0.1:8765/agents \u2014 click Run pipeline."
    Write-Host "  4. The 6 PM PDT daily digest will fire on its own."
} else {
    Write-Warn2 "Still missing $($stillMissing.Count) model(s):"
    foreach ($m in $stillMissing) { Write-Host "    - $m" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "  These probably hit a network error or aren't published on ollama.com."
    Write-Host "  Re-run this script \u2014 already-pulled models will be skipped."
    exit 1
}
