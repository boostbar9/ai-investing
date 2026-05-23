# ai-investing Windows installer
# ----------------------------------------------------------------------------
# Run this once on a fresh Windows 11 PC. It will:
#   1. Verify prerequisites (or install them via winget where possible).
#   2. Clone the repo to %USERPROFILE%\ai-investing (if not already present).
#   3. Create a .env file from .env.example.
#   4. Run `make setup-windows` then `make pull-models` (rx_7900_xt profile).
#   5. Drop a desktop shortcut for the tray launcher.
#
# Usage (in PowerShell, NOT cmd):
#   irm https://raw.githubusercontent.com/boostbar9/ai-investing/main/scripts/install.ps1 | iex
#
# Or, if you already cloned the repo:
#   cd <repo>; .\scripts\install.ps1
#
# Re-running this script is safe. It is idempotent.
# ----------------------------------------------------------------------------

$ErrorActionPreference = 'Stop'

function Write-Step { param([string]$msg) Write-Host ""; Write-Host ">>> $msg" -ForegroundColor Cyan }
function Write-Ok   { param([string]$msg) Write-Host "    [ok] $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host "    [!]  $msg" -ForegroundColor Yellow }
function Have-Cmd   { param([string]$name) return [bool](Get-Command $name -ErrorAction SilentlyContinue) }

# ----- 1. Prerequisites ------------------------------------------------------
Write-Step "Checking prerequisites"

if (-not (Have-Cmd winget)) {
    Write-Warn "winget is not installed. Install 'App Installer' from the Microsoft Store, then re-run this script."
    exit 1
}

$prereqs = @(
    @{ name = 'git';          id = 'Git.Git' },
    @{ name = 'node';         id = 'OpenJS.NodeJS.LTS' },
    @{ name = 'python';       id = 'Python.Python.3.12' },
    @{ name = 'docker';       id = 'Docker.DockerDesktop' },
    @{ name = 'ollama';       id = 'Ollama.Ollama' }
)

foreach ($p in $prereqs) {
    if (Have-Cmd $p.name) {
        Write-Ok "$($p.name) already installed"
    } else {
        Write-Host "    installing $($p.name)..."
        winget install --id $p.id --silent --accept-package-agreements --accept-source-agreements
    }
}

# pnpm via corepack, uv via pip
if (-not (Have-Cmd pnpm)) {
    Write-Host "    enabling pnpm via corepack..."
    corepack enable
    corepack prepare pnpm@9 --activate
}
if (-not (Have-Cmd uv)) {
    Write-Host "    installing uv..."
    pip install --user uv
}

# Make is not on Windows by default. Try chocolatey-style winget id.
if (-not (Have-Cmd make)) {
    Write-Host "    installing make..."
    winget install --id GnuWin32.Make --silent --accept-package-agreements --accept-source-agreements
}

# ----- 2. Clone repo ---------------------------------------------------------
$RepoDir = Join-Path $env:USERPROFILE 'ai-investing'
Write-Step "Locating repo at $RepoDir"
if (-not (Test-Path $RepoDir)) {
    git clone https://github.com/boostbar9/ai-investing $RepoDir
    Write-Ok "cloned"
} else {
    Write-Ok "already cloned — pulling latest"
    git -C $RepoDir pull --ff-only origin main
}
Set-Location $RepoDir

# ----- 3. .env ---------------------------------------------------------------
Write-Step "Configuring .env"
$envFile = Join-Path $RepoDir '.env'
$envExample = Join-Path $RepoDir '.env.example'
if (-not (Test-Path $envFile)) {
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile
        Write-Ok ".env created from .env.example — open it to add your Alpaca paper keys"
    } else {
        Write-Warn ".env.example not found; you'll need to create .env manually"
    }
} else {
    Write-Ok ".env already exists"
}

# Pin the hardware profile for Devin's RX 7900 XT
if (-not (Select-String -Path $envFile -Pattern '^HARDWARE_PROFILE=' -Quiet -ErrorAction SilentlyContinue)) {
    Add-Content -Path $envFile -Value "`nHARDWARE_PROFILE=rx_7900_xt"
    Write-Ok "set HARDWARE_PROFILE=rx_7900_xt"
}

# ----- 4. Setup + model pulls -----------------------------------------------
Write-Step "Running setup-windows + pull-models (this takes a while; ~30GB of models)"
make setup-windows
$env:HARDWARE_PROFILE = 'rx_7900_xt'
make pull-models

# ----- 5. Tray launcher deps + desktop shortcut ------------------------------
Write-Step "Installing tray launcher deps"
pip install --user pystray Pillow

Write-Step "Creating desktop shortcut"
$desktop  = [Environment]::GetFolderPath('Desktop')
$lnkPath  = Join-Path $desktop 'ai-investing.lnk'
$wsh      = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($lnkPath)
$shortcut.TargetPath       = (Get-Command pythonw).Source
$shortcut.Arguments        = "-m tools.tray.launcher"
$shortcut.WorkingDirectory = $RepoDir
$shortcut.IconLocation     = (Get-Command pythonw).Source
$shortcut.Description      = "Launch ai-investing tray app"
$shortcut.Save()
Write-Ok "shortcut placed on Desktop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Setup complete." -ForegroundColor Green
Write-Host " 1. Edit .env and paste your Alpaca paper keys." -ForegroundColor Green
Write-Host "    (Get them from https://app.alpaca.markets/paper/dashboard/overview)" -ForegroundColor Green
Write-Host " 2. Double-click the 'ai-investing' desktop shortcut." -ForegroundColor Green
Write-Host " 3. Use the tray icon to start the stack and open cockpit." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
