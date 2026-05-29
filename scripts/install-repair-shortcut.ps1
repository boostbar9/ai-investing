<#
.SYNOPSIS
  Drop a "Repair ai-investing" shortcut on the desktop.

.DESCRIPTION
  Creates a desktop shortcut that runs ``scripts\repair.ps1`` with
  ``-AndLaunch`` -- one click force-syncs the launcher files from
  origin/main and then starts the cockpit. Useful as a panic button
  when the regular ``ai-investing`` shortcut throws PowerShell parse
  errors at startup.

  Re-runnable: deletes any prior shortcut with the same name before
  recreating it.

.PARAMETER Name
  Override the shortcut's display name. Default: "Repair ai-investing".

.EXAMPLE
  PS> .\scripts\install-repair-shortcut.ps1
#>

[CmdletBinding()]
param(
  [string]$Name = "Repair ai-investing"
)

$ErrorActionPreference = "Stop"

function Write-Ok([string]$msg)   { Write-Host "  [ok] $msg" -ForegroundColor Green }
function Write-Warn3([string]$msg){ Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Fail([string]$msg) {
  Write-Host ""
  Write-Host "  [error] $msg" -ForegroundColor Red
  Write-Host ""
  exit 1
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repairPs1 = Join-Path $repoRoot "scripts\repair.ps1"

if (-not (Test-Path $repairPs1)) {
  Fail "repair.ps1 not found at $repairPs1. Pull the latest from main."
}

# We can't put -File <ps1> straight in a .lnk TargetPath because Windows
# will execute the .ps1 according to its file association (usually
# Notepad, sometimes nothing). Wrap it in powershell.exe with the right
# args so a double-click always runs the script.
$pwshExe = "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
if (-not (Test-Path $pwshExe)) {
  Fail "powershell.exe not found at $pwshExe -- this isn't a Windows host."
}

$desktop = [Environment]::GetFolderPath("Desktop")
if (-not (Test-Path $desktop)) {
  Fail "Could not locate the Desktop folder."
}

$shortcutPath = Join-Path $desktop "$Name.lnk"

if (Test-Path $shortcutPath) {
  Remove-Item $shortcutPath -Force
  Write-Warn3 "Removed existing shortcut: $shortcutPath"
}

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($shortcutPath)
$lnk.TargetPath       = $pwshExe
$lnk.Arguments        = "-NoProfile -ExecutionPolicy Bypass -File `"$repairPs1`" -AndLaunch"
$lnk.WorkingDirectory = $repoRoot
$lnk.WindowStyle      = 1
$lnk.Description      = "Force-sync the launcher scripts from origin/main and start the cockpit"

# Icon: a wrench-like glyph isn't built in, so reuse the shell32 settings
# icon (index 238 = gear in modern Windows). Falls back to powershell.exe
# if that index doesn't render.
$iconCandidates = @(
  "$env:WINDIR\System32\shell32.dll,238",
  "$pwshExe,0"
)
$lnk.IconLocation = $iconCandidates[0]
$lnk.Save()

# Sanity-check by re-reading.
if (Test-Path $shortcutPath) {
  Write-Ok "Repair shortcut created: $shortcutPath"
} else {
  Fail "Shortcut was not written to disk."
}

Write-Host ""
Write-Host "Double-click '$Name' on your desktop the next time the regular" -ForegroundColor Cyan
Write-Host "launcher throws PowerShell parse errors. It will re-pull the" -ForegroundColor Cyan
Write-Host "fixed launcher files from GitHub and start the cockpit." -ForegroundColor Cyan
