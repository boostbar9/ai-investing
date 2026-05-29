<#
.SYNOPSIS
  One-click repair for a broken ai-investing launcher.

.DESCRIPTION
  When ``launch.ps1`` fails to parse (PowerShell parser errors at startup),
  the most likely cause is that the local file is stale, locally-modified,
  or has been corrupted by a Windows code-page round-trip. This script:

    1. Refreshes ``origin/main`` from GitHub.
    2. Stashes any local edits to the launcher scripts so ``git`` can't
       refuse to overwrite them.
    3. Force-checks-out ``scripts\launch.ps1``, ``launch.cmd``,
       ``launch.sh``, and ``tools\boot.py`` from ``origin/main`` -- the
       four files the launcher actually depends on.
    4. Verifies the repaired file: confirms the UTF-8 BOM is present and
       prints the first few bytes so the user can SEE the fix landed.
    5. Optionally re-launches the cockpit (``-AndLaunch``).

  Safe to double-click. Never destroys uncommitted work elsewhere in the
  repo -- only the explicitly-named launcher files are reset.

.PARAMETER AndLaunch
  After repairing, immediately run ``scripts\launch.ps1``.

.EXAMPLE
  PS> .\scripts\repair.ps1
  PS> .\scripts\repair.ps1 -AndLaunch
#>

[CmdletBinding()]
param(
  [switch]$AndLaunch
)

# Don't use ErrorActionPreference=Stop here -- we want the script to keep
# going even if a single step fails, so the user sees the full diagnostic
# instead of a single-line crash.
$ErrorActionPreference = "Continue"

function Write-Ok([string]$msg)    { Write-Host "  [ok]    $msg" -ForegroundColor Green }
function Write-Info([string]$msg)  { Write-Host "  [info]  $msg" -ForegroundColor Cyan }
function Write-Warn2([string]$msg) { Write-Host "  [warn]  $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg)   { Write-Host "  [error] $msg" -ForegroundColor Red }

# Resolve the repo root (one level up from this script).
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host ""
Write-Host "=== ai-investing launcher repair ===" -ForegroundColor Cyan
Write-Host "Repo: $repoRoot" -ForegroundColor DarkGray
Write-Host ""

# 1. Make sure we're actually inside a git repo. If someone unzipped the
#    project instead of cloning, ``git fetch`` will fail noisily and we
#    can't repair anything -- tell them up front.
$gitDir = Join-Path $repoRoot ".git"
if (-not (Test-Path $gitDir)) {
  Write-Err ".git directory not found at $gitDir"
  Write-Err "This folder isn't a git clone -- repair requires git."
  Write-Err "Reclone with:  git clone https://github.com/boostbar9/ai-investing"
  Write-Host ""
  Read-Host "Press Enter to exit"
  exit 1
}

# 2. Fetch latest main.
Write-Info "Fetching origin/main..."
git fetch origin main 2>&1 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
if ($LASTEXITCODE -ne 0) {
  Write-Err "git fetch failed -- check your network connection and that 'git' is on PATH."
  Read-Host "Press Enter to exit"
  exit 1
}
Write-Ok "Fetched origin/main"

# 3. Stash any local edits to launcher files so checkout can't refuse.
$launcherFiles = @(
  "scripts/launch.ps1",
  "scripts/launch.cmd",
  "scripts/launch.sh",
  "tools/boot.py"
)

# Are any of them locally modified?
$dirtyFiles = @()
foreach ($f in $launcherFiles) {
  $status = git status --porcelain -- $f 2>$null
  if ($status) { $dirtyFiles += $f }
}

if ($dirtyFiles.Count -gt 0) {
  Write-Warn2 "Found local edits to: $($dirtyFiles -join ', ')"
  $stashMsg = "repair-$(Get-Date -Format yyyyMMdd-HHmmss)"
  git stash push -m $stashMsg -- $dirtyFiles 2>&1 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
  if ($LASTEXITCODE -eq 0) {
    Write-Ok "Stashed local edits under '$stashMsg' (recover with: git stash list)"
  } else {
    Write-Warn2 "Stash failed -- continuing with checkout --force anyway."
  }
}

# 4. Force-checkout the launcher files from origin/main.
Write-Info "Resetting launcher files to origin/main..."
foreach ($f in $launcherFiles) {
  git checkout origin/main -- $f 2>&1 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
  if ($LASTEXITCODE -eq 0) {
    Write-Ok "Restored $f"
  } else {
    Write-Warn2 "Could not restore $f (may not exist on origin/main yet)"
  }
}

# 5. Verify the fix: launch.ps1 must start with the UTF-8 BOM (EF BB BF)
#    and contain no em-dash bytes (E2 80 94).
$launchPs1 = Join-Path $repoRoot "scripts\launch.ps1"
if (-not (Test-Path $launchPs1)) {
  Write-Err "launch.ps1 still missing after repair -- something is very wrong."
  Read-Host "Press Enter to exit"
  exit 1
}

$raw = [System.IO.File]::ReadAllBytes($launchPs1)
$head = ($raw[0..2] | ForEach-Object { $_.ToString("X2") }) -join " "
Write-Host ""
Write-Host "  First 3 bytes of launch.ps1: $head" -ForegroundColor DarkGray
$hasBom = ($raw.Length -ge 3 -and $raw[0] -eq 0xEF -and $raw[1] -eq 0xBB -and $raw[2] -eq 0xBF)
if ($hasBom) {
  Write-Ok "UTF-8 BOM present (PowerShell 5.1 will parse the file correctly)"
} else {
  Write-Err "UTF-8 BOM MISSING -- the parse error will recur."
  Write-Err "Try: git fetch origin main; git reset --hard origin/main"
}

# Scan for em-dash byte triple (E2 80 94)
$hasEmDash = $false
for ($i = 0; $i -lt $raw.Length - 2; $i++) {
  if ($raw[$i] -eq 0xE2 -and $raw[$i+1] -eq 0x80 -and $raw[$i+2] -eq 0x94) {
    $hasEmDash = $true
    break
  }
}
if ($hasEmDash) {
  Write-Err "launch.ps1 still contains a U+2014 em-dash -- repair did not land."
} else {
  Write-Ok "No em-dash bytes detected"
}

# 6. Show the repaired HEAD so the user can sanity-check.
Write-Host ""
Write-Host "=== Repaired to commit ===" -ForegroundColor Cyan
git log --oneline -1 origin/main | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
Write-Host ""

if ($hasBom -and -not $hasEmDash) {
  Write-Host "=== Repair complete ===" -ForegroundColor Green
  Write-Host ""
  if ($AndLaunch) {
    Write-Info "Launching cockpit..."
    & (Join-Path $repoRoot "scripts\launch.ps1")
  } else {
    Write-Host "Double-click your 'ai-investing' desktop shortcut to launch the cockpit." -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Press Enter to exit"
  }
} else {
  Write-Host "=== Repair INCOMPLETE -- see [error] lines above ===" -ForegroundColor Red
  Write-Host ""
  Read-Host "Press Enter to exit"
  exit 1
}
