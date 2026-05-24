<#
.SYNOPSIS
  Create a desktop shortcut that launches ai-investing with one click.

.DESCRIPTION
  Drops an ``ai-investing.lnk`` on the user's desktop pointing at
  ``scripts\launch.cmd``. The shortcut runs in the repo's working directory
  and uses the repo icon if available.

  Re-runnable: deletes any prior shortcut with the same name before
  recreating it.

.PARAMETER WithDocker
  If set, the shortcut will pass ``-WithDocker`` to launch.ps1 so the
  Docker stack starts every time you double-click the icon.

.PARAMETER Name
  Override the shortcut's display name. Default: "ai-investing".

.EXAMPLE
  PS> .\scripts\install-shortcut.ps1

.EXAMPLE
  PS> .\scripts\install-shortcut.ps1 -WithDocker -Name "ai-investing (full stack)"
#>

[CmdletBinding()]
param(
  [switch]$WithDocker,
  [string]$Name = "ai-investing"
)

$ErrorActionPreference = "Stop"

function Write-Ok([string]$msg)   { Write-Host "  [ok] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Fail([string]$msg) {
  Write-Host ""
  Write-Host "  [error] $msg" -ForegroundColor Red
  Write-Host ""
  exit 1
}

# Resolve the repo root (one level up from this script).
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launchCmd = Join-Path $repoRoot "scripts\launch.cmd"

if (-not (Test-Path $launchCmd)) {
  Fail "launch.cmd not found at $launchCmd. Pull the latest from main."
}

# Desktop path - works for OneDrive-redirected desktops too.
$desktop = [Environment]::GetFolderPath("Desktop")
if (-not (Test-Path $desktop)) {
  Fail "Could not locate the Desktop folder."
}

$shortcutPath = Join-Path $desktop "$Name.lnk"

# Remove any existing shortcut so re-running this script always wins.
if (Test-Path $shortcutPath) {
  Remove-Item $shortcutPath -Force
  Write-Warn "Removed existing shortcut: $shortcutPath"
}

# Build the shortcut.
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($shortcutPath)
$lnk.TargetPath       = $launchCmd
$lnk.WorkingDirectory = $repoRoot
$lnk.WindowStyle      = 1
$lnk.Description      = "Launch ai-investing cockpit + paper-trading bot"

if ($WithDocker) {
  $lnk.Arguments = "-WithDocker"
  Write-Ok "Shortcut will pass -WithDocker on every launch"
}

# Use a sensible icon. Windows ships powershell.exe with a usable icon
# at index 0; the actual file we're targeting is .cmd which has no icon
# of its own.
$iconCandidates = @(
  (Join-Path $repoRoot "scripts\icon.ico"),
  "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
)
foreach ($candidate in $iconCandidates) {
  if (Test-Path $candidate) {
    $lnk.IconLocation = "$candidate,0"
    break
  }
}

$lnk.Save()

Write-Ok "Desktop shortcut created: $shortcutPath"
Write-Host ""
Write-Host "Double-click the '$Name' icon on your desktop to start the cockpit." -ForegroundColor Cyan
