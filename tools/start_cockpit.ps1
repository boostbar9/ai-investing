#requires -Version 5.1
<#
.SYNOPSIS
    One-command launcher for the ai-investing cockpit + remote tunnel.

.DESCRIPTION
    Phase 36d -- starts everything you need for the Perplexity agent
    to remote-control your cockpit:

      1. Ensures COCKPIT_REMOTE_TOKEN is set (generates if missing).
      2. Starts uvicorn in a new window (with the token in env).
      3. Waits for cockpit to be reachable on 127.0.0.1:8000.
      4. Ensures cloudflared.exe is installed (auto-downloads if missing).
      5. Starts a Cloudflare quick-tunnel, parses the public URL.
      6. Writes URL + token to data/cockpit/remote_handle.json.
      7. Publishes the handle to a private GitHub branch
         (refs/heads/cockpit-handle) so the agent can always find it.
      8. Tails the tunnel; Ctrl+C stops everything.

    Run this ONCE and the agent can reach your bot for as long as the
    window stays open. Re-run anytime; URL refreshes automatically.

.PARAMETER NoPublish
    Skip the GitHub branch publish step. Use if you want to keep the
    handle local only and paste the URL/token to the agent yourself.

.PARAMETER NewToken
    Force-generate a new remote token, replacing the existing one.

.PARAMETER NoPull
    Skip the auto-pull step. Use if you have uncommitted local changes
    you don't want to risk, or you want to boot from your current code.

.EXAMPLE
    PS> .\tools\start_cockpit.ps1
    Starts everything; agent auto-discovers via GitHub handle branch.
#>

[CmdletBinding()]
param(
    [switch]$NoPublish,
    [switch]$NewToken,
    [switch]$NoPull
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BinDir = Join-Path $RepoRoot "tools\bin"
$CloudflaredExe = Join-Path $BinDir "cloudflared.exe"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$HandleDir = Join-Path $RepoRoot "data\cockpit"
$HandlePath = Join-Path $HandleDir "remote_handle.json"
$CockpitUrl = "http://127.0.0.1:8000"

function Write-Section($text) {
    Write-Host ""
    Write-Host "=== $text ===" -ForegroundColor Cyan
}

function Test-CockpitReachable {
    param([string]$Url)
    try {
        $r = Invoke-WebRequest -Uri "$Url/api/remote/health" -UseBasicParsing -TimeoutSec 3
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function New-RemoteToken {
    $bytes = New-Object byte[] 16
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    return -join ($bytes | ForEach-Object { $_.ToString("x2") })
}

function Publish-HandleViaGh {
    # Fallback publisher: when `git push` fails (commonly because local
    # git has no GitHub credentials configured), use the GitHub CLI to
    # PUT the handle file straight onto the cockpit-handle branch via the
    # REST contents API. `gh` carries its own auth (gh auth login), so it
    # works even when plain git push gets exit=128.
    #
    # Returns $true on success, $false if gh is missing/unauthenticated
    # or the API call fails -- caller then falls back to manual paste.
    param(
        [string]$HandlePath,
        [string]$Branch
    )
    $gh = Get-Command gh -ErrorAction SilentlyContinue
    if (-not $gh) {
        Write-Host "  gh CLI not installed; cannot use API fallback." -ForegroundColor DarkGray
        return $false
    }
    try {
        # Resolve owner/repo from origin remote.
        $originUrl = (& git remote get-url origin 2>$null)
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($originUrl)) { return $false }
        if ($originUrl -match "github\.com[:/]([^/]+)/([^/.]+)") {
            $owner = $matches[1]; $repo = $matches[2]
        } else { return $false }

        $apiPath = "repos/$owner/$repo/contents/data/cockpit/remote_handle.json"
        $contentB64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($HandlePath))
        $msg = "cockpit: publish handle " + (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")

        # Need the current blob sha on that branch (if the file exists) to update it.
        $sha = $null
        $existing = (& gh api "$apiPath`?ref=$Branch" 2>$null | ConvertFrom-Json)
        if ($LASTEXITCODE -eq 0 -and $existing.sha) { $sha = $existing.sha }

        $args = @("api", "--method", "PUT", $apiPath,
                  "-f", "message=$msg",
                  "-f", "content=$contentB64",
                  "-f", "branch=$Branch")
        if ($sha) { $args += @("-f", "sha=$sha") }

        & gh @args *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Published handle via gh API to origin/$Branch" -ForegroundColor Green
            return $true
        }
        return $false
    } catch {
        return $false
    }
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

function Repair-MiseProfile {
    # Some users' PowerShell profiles run `mise activate pwsh | Out-String |
    # Invoke-Expression` unconditionally. When mise isn't installed, every
    # new session -- including each cockpit/uvicorn window this launcher
    # spawns -- prints a noisy "The term 'mise' is not recognized" error.
    #
    # Guard any `mise activate` line behind a Get-Command check so it cleanly
    # no-ops when mise is absent. SAFE + idempotent:
    #   - only rewrites lines matching `mise activate` that aren't already
    #     guarded (line already contains `Get-Command mise`);
    #   - backs the profile up once to <profile>.bak before the first edit;
    #   - never throws (each profile is handled in its own try/catch) so a
    #     missing/read-only profile can't crash the launcher.
    $targets = @($PROFILE.CurrentUserAllHosts, $PROFILE.CurrentUserCurrentHost) |
        Where-Object { $_ } | Select-Object -Unique
    $patchedAny = $false
    $cleanAny = $false
    foreach ($profilePath in $targets) {
        try {
            if (-not (Test-Path $profilePath)) { continue }
            $lines = @(Get-Content -LiteralPath $profilePath)
            $changed = $false
            $out = foreach ($line in $lines) {
                if ($line -match '^\s*mise\s+activate' -and $line -notmatch 'Get-Command\s+mise') {
                    $changed = $true
                    $indent = if ($line -match '^(\s*)') { $matches[1] } else { '' }
                    $indent + 'if (Get-Command mise -ErrorAction SilentlyContinue) { ' + $line.TrimStart() + ' }'
                } else {
                    $line
                }
            }
            if ($changed) {
                $bak = "$profilePath.bak"
                if (-not (Test-Path $bak)) {
                    Copy-Item -LiteralPath $profilePath -Destination $bak -Force -ErrorAction SilentlyContinue
                }
                Set-Content -LiteralPath $profilePath -Value $out -Encoding UTF8
                Write-Host "Patched PowerShell profile to silence mise error: $profilePath" -ForegroundColor Green
                $patchedAny = $true
            } else {
                $cleanAny = $true
            }
        } catch {
            Write-Host "  Could not patch profile $profilePath (skipped, non-fatal): $($_.Exception.Message)" -ForegroundColor DarkGray
        }
    }
    if (-not $patchedAny -and $cleanAny) {
        Write-Host "PowerShell profile already clean (no mise guard needed)." -ForegroundColor Green
    } elseif (-not $patchedAny -and -not $cleanAny) {
        Write-Host "No user PowerShell profile found to check (nothing to do)." -ForegroundColor DarkGray
    }
}

# --------------------------------------------------------------------
# Pre-flight: silence the `mise` PowerShell-profile error (if present)
# --------------------------------------------------------------------
# Done before anything spawns a child PowerShell window so those windows
# inherit the cleaned profile. Fully non-fatal.
Write-Section "Pre-flight: PowerShell profile hygiene"
try {
    Repair-MiseProfile
} catch {
    Write-Host "Profile hygiene step skipped (non-fatal): $($_.Exception.Message)" -ForegroundColor DarkGray
}

# --------------------------------------------------------------------
# 0. Auto-pull latest code from origin (Phase 36e+)
# --------------------------------------------------------------------
# Rationale: the cockpit boots from whatever is on disk. Without this
# step, a fresh launch reuses stale code even if the agent has pushed
# fixes. Doing a fast-forward-only pull here means every double-click
# of the desktop shortcut starts a clean, current cockpit.
#
# Safety properties:
#   - Uncommitted changes block the update (we never discard user work):
#     we detect a dirty tree, warn, and boot on the existing code.
#   - On a CLEAN tree we land exactly on origin/main: fast-forward when
#     possible, else hard-reset (safe -- no local work to lose). This is
#     what makes the update RELIABLE instead of silently skipping.
#   - Credential-less PCs: a plain `git fetch` (exit=128) is retried once
#     with the GitHub CLI's token, scoped to that single fetch.
#   - If GitHub is unreachable we print a LOUD warning and keep booting --
#     the update never hard-fails the launcher (try/catch around it all).
#   - --NoPull skips the whole block for the rare case you want to
#     boot exactly what is on disk.
#   - All update output is visible; no silent failures.
Write-Section "Step 0: Sync code from origin"
if ($NoPull) {
    Write-Host "Skipped (-NoPull flag set)." -ForegroundColor Yellow
} else {
    # Git writes progress to stderr; relax error policy for this block
    # and rely on $LASTEXITCODE for real failure detection (same trick
    # we use in the publish step).
    $prevErrPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $RepoRoot
    try {
        # 0a. Refuse to pull if there are uncommitted changes -- pulling
        #     would either fail or risk a stash dance we don't want.
        #     Ignore untracked files (data/learning/, tools/bin/, etc.)
        #     since pull only touches tracked files.
        $dirty = & git status --porcelain --untracked-files=no 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "git status failed (not a repo or git missing?); skipping pull." -ForegroundColor Yellow
        } elseif ($dirty) {
            Write-Host "Uncommitted local changes detected -- skipping pull." -ForegroundColor Yellow
            Write-Host "  Commit or stash to enable auto-pull on next launch." -ForegroundColor DarkGray
            $dirty -split "`n" | Select-Object -First 5 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
        } else {
            $beforeSha = (& git rev-parse --short HEAD 2>$null)

            # 0b. Fetch origin/main. The user's PC frequently has NO git
            #     credentials configured, so a plain fetch returns exit=128.
            #     We retry once using the GitHub CLI's own auth token, scoped
            #     to THIS single fetch via -c http.extraheader (no global
            #     config change, token never persisted).
            & git fetch --quiet origin main 2>$null
            $fetchOk = ($LASTEXITCODE -eq 0)
            if (-not $fetchOk) {
                $gh = Get-Command gh -ErrorAction SilentlyContinue
                if ($gh) {
                    & gh auth status *> $null
                    if ($LASTEXITCODE -eq 0) {
                        $ghToken = (& gh auth token 2>$null)
                        if (-not [string]::IsNullOrWhiteSpace($ghToken)) {
                            Write-Host "Plain git fetch failed; retrying with GitHub CLI token..." -ForegroundColor Yellow
                            & git -c http.extraheader="AUTHORIZATION: bearer $ghToken" fetch --quiet origin main 2>$null
                            $fetchOk = ($LASTEXITCODE -eq 0)
                        }
                    }
                }
            }

            if (-not $fetchOk) {
                # LOUD, unmissable message -- this is the failure mode that
                # kept the user on stale code. Never hard-fail the launcher.
                Write-Host "============================================================" -ForegroundColor Red
                Write-Host " Could not reach GitHub to update -- booting on current code $beforeSha." -ForegroundColor Red
                Write-Host " Your PC may be missing the latest fixes." -ForegroundColor Red
                Write-Host " (Fix: run ``gh auth login`` once, or configure git credentials.)" -ForegroundColor DarkGray
                Write-Host "============================================================" -ForegroundColor Red
            } else {
                $targetSha = (& git rev-parse --short origin/main 2>$null)
                if ($beforeSha -eq $targetSha) {
                    Write-Host "Already up to date at $beforeSha." -ForegroundColor Green
                } else {
                    # Prefer a fast-forward. If the local branch has diverged
                    # (e.g. stale orphan/handle commits) we are SAFE to hard-
                    # reset because we already verified the tree is clean
                    # above -- no user work can be lost.
                    & git merge --ff-only --quiet origin/main 2>$null
                    $moved = ($LASTEXITCODE -eq 0)
                    if (-not $moved) {
                        Write-Host "Fast-forward not possible (local branch diverged); hard-resetting CLEAN tree to origin/main..." -ForegroundColor Yellow
                        & git reset --hard origin/main 2>$null
                        $moved = ($LASTEXITCODE -eq 0)
                    }
                    if ($moved) {
                        $afterSha = (& git rev-parse --short HEAD 2>$null)
                        Write-Host "============================================================" -ForegroundColor Green
                        Write-Host " Updated: $beforeSha -> $afterSha  (now on origin/main)" -ForegroundColor Green
                        Write-Host "============================================================" -ForegroundColor Green
                        # Pip-install in editable mode so package metadata + entry
                        # points pick up. --quiet keeps the console readable; the
                        # full log goes to a temp file for post-mortem if needed.
                        $pipLog = Join-Path $env:TEMP "cockpit_pip.log"
                        Write-Host "Reinstalling package (pip install -e .)..." -ForegroundColor Yellow
                        & $VenvPython -m pip install -e . --quiet --disable-pip-version-check *> $pipLog
                        if ($LASTEXITCODE -eq 0) {
                            Write-Host "Pip install OK." -ForegroundColor Green
                        } else {
                            Write-Host "Pip install returned exit=$LASTEXITCODE; booting anyway." -ForegroundColor Yellow
                            Write-Host "  Full log: $pipLog" -ForegroundColor DarkGray
                        }
                    } else {
                        Write-Host "Update could not be applied; booting on current code at $beforeSha." -ForegroundColor Yellow
                        Write-Host "  Resolve manually: git status / git fetch / git reset --hard origin/main" -ForegroundColor DarkGray
                    }
                }
            }
        }
    } catch {
        Write-Host "Auto-update step hit an unexpected error; booting on current code (non-fatal): $($_.Exception.Message)" -ForegroundColor Yellow
    } finally {
        Pop-Location
        $ErrorActionPreference = $prevErrPref
    }
}

# --------------------------------------------------------------------
# 1. Token
# --------------------------------------------------------------------
Write-Section "Step 1: Auth token"
$existing = [Environment]::GetEnvironmentVariable("COCKPIT_REMOTE_TOKEN", "User")
if ($NewToken -or [string]::IsNullOrWhiteSpace($existing) -or $existing.Length -lt 16) {
    $token = New-RemoteToken
    [Environment]::SetEnvironmentVariable("COCKPIT_REMOTE_TOKEN", $token, "User")
    Write-Host "Generated NEW remote token, saved to User env." -ForegroundColor Green
} else {
    $token = $existing
    Write-Host "Using existing COCKPIT_REMOTE_TOKEN (length=$($token.Length))." -ForegroundColor Green
}
$env:COCKPIT_REMOTE_TOKEN = $token

# --------------------------------------------------------------------
# 2. Start cockpit (or detect already-running)
# --------------------------------------------------------------------
Write-Section "Step 2: Cockpit (uvicorn)"
if (Test-CockpitReachable -Url $CockpitUrl) {
    Write-Host "Cockpit already running at $CockpitUrl." -ForegroundColor Green
    Write-Host "NOTE: if you JUST changed the token, restart it: Ctrl+C in cockpit window." -ForegroundColor Yellow
} else {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "Venv python not found at $VenvPython" -ForegroundColor Red
        Write-Host "Create one first: python -m venv .venv ; .\.venv\Scripts\python -m pip install -e ." -ForegroundColor Red
        exit 1
    }
    # --------------------------------------------------------------------
    # 2a. Stale-port guard.
    # --------------------------------------------------------------------
    # We are here because Test-CockpitReachable returned $false. If port
    # 8000 is nonetheless held by a LISTENING process, that process is a
    # dead/stale cockpit (or an unrelated app) -- either way the new
    # uvicorn will fail to bind and crash-loop. Conservatively free the
    # port ONLY if the owner is a python/uvicorn process; otherwise leave
    # it alone and tell the user, so we never kill unrelated software.
    try {
        $stale = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    } catch {
        $stale = $null
    }
    if ($stale) {
        $freedAny = $false
        foreach ($conn in @($stale)) {
            $owningPid = $conn.OwningProcess
            $owner = Get-Process -Id $owningPid -ErrorAction SilentlyContinue
            if ($owner -and ($owner.ProcessName -match '(?i)python|uvicorn')) {
                Write-Host "Port 8000 held by a stale cockpit ($($owner.ProcessName), PID $owningPid); freeing it..." -ForegroundColor Yellow
                Stop-Process -Id $owningPid -Force -ErrorAction SilentlyContinue
                $freedAny = $true
            } else {
                $ownerName = if ($owner) { $owner.ProcessName } else { "unknown" }
                Write-Host "Port 8000 is in use by '$ownerName' (PID $owningPid), which is NOT python/uvicorn." -ForegroundColor Red
                Write-Host "  Close that program (or the old cockpit window) and re-run this launcher." -ForegroundColor Yellow
            }
        }
        if ($freedAny) { Start-Sleep -Seconds 1 }
    }

    Write-Host "Starting cockpit in a new window (auto-restart enabled)..." -ForegroundColor Yellow
    # Pass token through environment to the child process and wrap uvicorn
    # in a while-loop so a single bad request (e.g. an unhandled error in
    # the OAuth flow) can't take the UI down: if uvicorn exits, we log it,
    # wait 2s, and restart. -NoExit keeps the window open so repeated
    # failures stay visible. Console output is tee'd to a temp log so a
    # crash is captured even if the window is missed; the web server also
    # writes its own data/cockpit/logs/cockpit_web.log (RotatingFileHandler)
    # which /api/remote/weblog exposes.
    #
    # Quoting: $token (hex) and $VenvPython (Windows path) are expanded by
    # THIS parent shell via single-quoted literals; backtick-$ tokens stay
    # literal so they evaluate inside the child shell.
    $startCmd = "`$env:COCKPIT_REMOTE_TOKEN='$token'; " +
        "`$ErrorActionPreference='Continue'; " +
        "`$cockpitLog = Join-Path `$env:TEMP 'cockpit_server.log'; " +
        "Write-Host ('[cockpit] console log: ' + `$cockpitLog) -ForegroundColor DarkGray; " +
        "while (`$true) { " +
            "Write-Host '[cockpit] starting uvicorn on 127.0.0.1:8000 ...' -ForegroundColor Cyan; " +
            "& '$VenvPython' -m uvicorn packages.cockpit.web.server:app --host 127.0.0.1 --port 8000 2>&1 | Tee-Object -FilePath `$cockpitLog -Append; " +
            "Write-Host ('[cockpit] server exited (code ' + `$LASTEXITCODE + '). Restarting in 2s... close this window to stop.') -ForegroundColor Yellow; " +
            "Start-Sleep -Seconds 2 " +
        "}"
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-Command", $startCmd) -WorkingDirectory $RepoRoot | Out-Null

    Write-Host "Waiting for cockpit to come up..." -ForegroundColor DarkGray
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        if (Test-CockpitReachable -Url $CockpitUrl) { $ready = $true; break }
    }
    if (-not $ready) {
        Write-Host "Cockpit did not come up within 60s. Check the uvicorn window." -ForegroundColor Red
        exit 1
    }
    Write-Host "Cockpit is up at $CockpitUrl." -ForegroundColor Green
}

# --------------------------------------------------------------------
# 3. cloudflared
# --------------------------------------------------------------------
Write-Section "Step 3: cloudflared"
$cf = Install-Cloudflared

# --------------------------------------------------------------------
# 4. Tunnel
# --------------------------------------------------------------------
Write-Section "Step 4: Starting tunnel"
$logPath = Join-Path $env:TEMP "cockpit_tunnel.log"
if (Test-Path $logPath) { Remove-Item $logPath -Force }

$tunnelArgs = @("tunnel", "--url", $CockpitUrl, "--no-autoupdate", "--logfile", $logPath)
$proc = Start-Process -FilePath $cf -ArgumentList $tunnelArgs -PassThru -NoNewWindow

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
    Write-Host "Could not detect tunnel URL within 60s. Check $logPath." -ForegroundColor Red
    try { Stop-Process -Id $proc.Id -Force } catch {}
    exit 1
}

Write-Host "Tunnel URL: $publicUrl" -ForegroundColor Green

# Open the LOCAL dashboard as the primary browser tab -- NOT the tunnel URL.
# The Robinhood OAuth callback redirects to http://127.0.0.1:8000/callback,
# which only resolves on THIS machine; completing Connect from the tunnel
# tab lands the loopback callback in the wrong context. So the tab the user
# should actually use is the local one. The tunnel URL stays first-class for
# phone/remote: printed prominently here, written to the handle JSON, and
# rendered as a QR page below -- it just isn't the auto-opened primary tab.
$localUrl = "$CockpitUrl/"
Write-Host ""
Write-Host "Open this on THIS computer to use the dashboard & connect Robinhood:  $localUrl" -ForegroundColor Cyan
Write-Host "Open this on your PHONE / share with the agent (remote):  $publicUrl" -ForegroundColor Cyan
Write-Host ""
try {
    Start-Process $localUrl | Out-Null
    Write-Host "Opened the LOCAL dashboard ($localUrl) in your browser -- use this tab to Connect Robinhood." -ForegroundColor Green
} catch {
    Write-Host "Could not auto-open the browser; visit $localUrl on this computer." -ForegroundColor Yellow
}

# --------------------------------------------------------------------
# 5. Write handle JSON
# --------------------------------------------------------------------
Write-Section "Step 5: Write handle"
New-Item -ItemType Directory -Force -Path $HandleDir | Out-Null
$handle = [ordered]@{
    url        = $publicUrl
    token      = $token
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    host_name  = $env:COMPUTERNAME
    repo_root  = $RepoRoot
}
$handleJson = $handle | ConvertTo-Json
Set-Content -Path $HandlePath -Value $handleJson -Encoding UTF8
Write-Host "Wrote $HandlePath" -ForegroundColor Green

# --------------------------------------------------------------------
# 6. Publish handle to GitHub branch (optional)
# --------------------------------------------------------------------
if (-not $NoPublish) {
    Write-Section "Step 6: Publish handle to GitHub"
    # Git writes progress lines to stderr; PowerShell's `Stop` policy treats
    # those as terminating errors. Temporarily relax it for this block and
    # rely on $LASTEXITCODE for real failure detection.
    $prevErrPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $RepoRoot
    $publishOk = $false
    try {
        $tmpWorktree = Join-Path $env:TEMP ("cockpit-handle-" + [Guid]::NewGuid().ToString("N"))
        $branchName = "cockpit-handle"

        & git worktree add --detach $tmpWorktree *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "git worktree add failed; skipping publish. Handle is local only." -ForegroundColor Yellow
        } else {
            Push-Location $tmpWorktree
            try {
                & git checkout --orphan $branchName *> $null
                & git rm -rf . *> $null
                $remoteHandleDir = Join-Path $tmpWorktree "data\cockpit"
                New-Item -ItemType Directory -Force -Path $remoteHandleDir | Out-Null
                Copy-Item $HandlePath (Join-Path $remoteHandleDir "remote_handle.json") -Force

                & git add data/cockpit/remote_handle.json *> $null
                $commitMsg = "cockpit: publish handle " + (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
                & git -c user.email=devfarinsky@gmail.com -c user.name="Devin Farinsky" commit -m $commitMsg *> $null
                # Orphan single-file state ref, rewritten every launch on purpose.
                & git push --force origin ("HEAD:refs/heads/" + $branchName) *> $null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "Published handle to origin/$branchName" -ForegroundColor Green
                    $publishOk = $true
                } else {
                    Write-Host "git push failed (exit=$LASTEXITCODE); trying GitHub CLI fallback..." -ForegroundColor Yellow
                    $publishOk = Publish-HandleViaGh -HandlePath $HandlePath -Branch $branchName
                    if (-not $publishOk) {
                        Write-Host "Publish to origin failed (git push + gh both unavailable)." -ForegroundColor Yellow
                        Write-Host "Handle is still available locally at $HandlePath" -ForegroundColor Yellow
                        Write-Host "  Agent fallback: paste the Public URL + Token shown below." -ForegroundColor DarkGray
                    }
                }
            } finally {
                Pop-Location
            }
            & git worktree remove --force $tmpWorktree *> $null
        }
    } finally {
        Pop-Location
        $ErrorActionPreference = $prevErrPref
    }
}

# --------------------------------------------------------------------
# 7a. Phone access: write a QR-code page and open it
# --------------------------------------------------------------------
# The tunnel URL IS the phone-accessible dashboard (same FastAPI app),
# so all the phone needs is that URL. We render a scannable QR locally
# in a tiny self-contained HTML page (QR drawn client-side -- no extra
# Python packages, no third-party image API, works offline once open).
Write-Section "Step 7: Phone access"
$qrPage = Join-Path $env:TEMP "cockpit_phone.html"
$qrHtml = @"
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>View AI Trading Bot on your phone</title>
<style>
  body{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#0b0f14;color:#e6edf3;
       margin:0;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;padding:24px}
  h1{font-size:20px;margin:0}
  p{color:#9fb0c0;margin:4px 0;text-align:center;max-width:420px}
  #qr{background:#fff;padding:16px;border-radius:12px}
  code{background:#161b22;padding:6px 10px;border-radius:8px;font-size:13px;word-break:break-all;color:#7ee787}
  a.btn{background:#00e5c4;color:#06231f;text-decoration:none;font-weight:700;padding:10px 18px;border-radius:10px}
</style>
<script src="https://cdn.jsdelivr.net/npm/qrcode@1.5.3/build/qrcode.min.js"></script>
</head><body>
  <h1>📱 Open the bot on your phone</h1>
  <p>Scan this with your phone camera. It opens the live dashboard from anywhere.</p>
  <div id="qr"></div>
  <p>or type this address into your phone browser:</p>
  <code>$publicUrl</code>
  <a class="btn" href="$publicUrl" target="_blank">Open here instead</a>
  <p style="font-size:12px;color:#5b6b7b">This page is safe to leave open. The link works only while the launcher window stays running.</p>
<script>
  QRCode.toCanvas(document.createElement('canvas'), '$publicUrl', {width:240, margin:1},
    function(err, canvas){ if(!err){ document.getElementById('qr').appendChild(canvas); }
      else { document.getElementById('qr').textContent='(QR failed to render \u2014 use the address below)'; } });
</script>
</body></html>
"@
Set-Content -Path $qrPage -Value $qrHtml -Encoding UTF8
try {
    Start-Process $qrPage | Out-Null
    Write-Host "Opened a QR-code page -- scan it with your phone to view the bot." -ForegroundColor Green
} catch {
    Write-Host "QR page saved to $qrPage (open it to scan)." -ForegroundColor Yellow
}

# --------------------------------------------------------------------
# 7b. Status block + block on tunnel
# --------------------------------------------------------------------
Write-Section "Cockpit + tunnel are LIVE"
Write-Host ""
Write-Host "On THIS computer : $CockpitUrl" -ForegroundColor Green
Write-Host "On your PHONE    : $publicUrl" -ForegroundColor Green
Write-Host "  (scan the QR page that just opened, or type that address)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Token      : $token" -ForegroundColor DarkGray
Write-Host "Handle     : $HandlePath" -ForegroundColor DarkGray
if (-not $NoPublish) {
    Write-Host "Published  : origin/cockpit-handle" -ForegroundColor DarkGray
}
Write-Host ""
Write-Host "The agent can now reach your cockpit." -ForegroundColor Yellow
Write-Host "You don't need to tell it the URL -- it will read the published handle." -ForegroundColor Yellow
Write-Host ""
Write-Host "Leave this window open while you want remote access." -ForegroundColor DarkGray
Write-Host "Press Ctrl+C to stop the tunnel." -ForegroundColor DarkGray
Write-Host ""

Wait-Process -Id $proc.Id
