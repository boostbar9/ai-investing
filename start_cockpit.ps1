<#
.SYNOPSIS
  Shim that forwards to scripts/launch.ps1.

.DESCRIPTION
  The canonical launcher lives at scripts/launch.ps1, but operators
  (including me) reach for ``.\start_cockpit.ps1`` out of muscle memory.
  This shim accepts the same parameters and forwards them through so
  either invocation works.

.EXAMPLE
  PS> .\start_cockpit.ps1
  PS> .\start_cockpit.ps1 -WithDocker
  PS> .\start_cockpit.ps1 -NoPull -Port 9000
#>

[CmdletBinding()]
param(
  [switch]$WithDocker,
  [switch]$NoPull,
  [int]$Port = 8765,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$target = Join-Path $PSScriptRoot "scripts\launch.ps1"
if (-not (Test-Path $target)) {
  Write-Host "[error] scripts/launch.ps1 is missing from this checkout." -ForegroundColor Red
  Write-Host "Re-clone the repo or run scripts/install.ps1." -ForegroundColor Red
  exit 1
}

# Forward only the parameters that were explicitly bound, so launch.ps1's
# defaults still kick in when the caller didn't override them.
$forward = @{}
foreach ($key in $PSBoundParameters.Keys) {
  $forward[$key] = $PSBoundParameters[$key]
}
& $target @forward
exit $LASTEXITCODE
