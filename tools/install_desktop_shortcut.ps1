#requires -Version 5.1
<#
.SYNOPSIS
    Create a "Start AI Trading Bot" shortcut on your Desktop.

.DESCRIPTION
    One-time installer. Double-click this script (or run it once from
    PowerShell) and it puts a shortcut on your Desktop that launches
    tools\start_cockpit.ps1 -- which starts the cockpit, the tunnel,
    and publishes the handle so the agent can connect.

    After running this, you can ignore PowerShell entirely. Just
    double-click the desktop icon to start everything.

.PARAMETER Name
    Shortcut display name on the desktop. Default: "Start AI Trading Bot".

.EXAMPLE
    PS> .\tools\install_desktop_shortcut.ps1
#>

[CmdletBinding()]
param(
    [string]$Name = "Start AI Trading Bot"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
# Prefer the root one-click START.cmd entry point; fall back to the
# PS1 launcher if the cmd wrapper is somehow missing.
$StartCmd = Join-Path $RepoRoot "START.cmd"
$Launcher = if (Test-Path $StartCmd) { $StartCmd } else { Join-Path $RepoRoot "tools\start_cockpit.ps1" }
$IconSource = Join-Path $env:SystemRoot "System32\imageres.dll"   # built-in Windows icons
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath ("$Name.lnk")

if (-not (Test-Path $Launcher)) {
    Write-Host "Launcher not found at $Launcher" -ForegroundColor Red
    Write-Host "Run this from inside the ai-investing repo." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Creating shortcut..." -ForegroundColor Cyan
Write-Host "  Target  : $Launcher"
Write-Host "  Desktop : $ShortcutPath"
Write-Host ""

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($ShortcutPath)
if ($Launcher -like "*.cmd") {
    # Point straight at the cmd wrapper -- simplest, no policy flags needed.
    $lnk.TargetPath = $Launcher
    $lnk.Arguments = ""
} else {
    $lnk.TargetPath = "powershell.exe"
    $lnk.Arguments = "-NoExit -NoProfile -ExecutionPolicy Bypass -File `"$Launcher`""
}
$lnk.WorkingDirectory = $RepoRoot
$lnk.WindowStyle = 1
$lnk.Description = "Starts the ai-investing cockpit and remote tunnel."
# imageres.dll index 109 is a chart/graph icon. Fallback to PS icon if not present.
if (Test-Path $IconSource) {
    $lnk.IconLocation = "$IconSource,109"
} else {
    $lnk.IconLocation = "powershell.exe,0"
}
$lnk.Save()

# Mark the .lnk as not-from-internet so it doesn't get blocked.
try {
    Unblock-File -Path $ShortcutPath -ErrorAction SilentlyContinue
} catch {}

Write-Host "Shortcut created:" -ForegroundColor Green
Write-Host "  $ShortcutPath" -ForegroundColor Green
Write-Host ""
Write-Host "Done. Double-click '$Name' on your Desktop to start the bot." -ForegroundColor Yellow
Write-Host "First run will download cloudflared (~20 MB) -- give it a minute." -ForegroundColor DarkGray
