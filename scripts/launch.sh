#!/usr/bin/env bash
# One-click launcher for ai-investing on macOS / Linux.
#
# Mirrors scripts/launch.ps1: activates venv, validates .env, optionally
# pulls latest and starts Docker, then launches the cockpit. Designed for
# parity with the Windows entry point so cross-platform users have one
# muscle memory.
#
# Usage:
#   ./scripts/launch.sh                  # cockpit only
#   ./scripts/launch.sh --with-docker    # plus Postgres/Temporal/Ollama
#   ./scripts/launch.sh --no-pull        # skip git pull
#   ./scripts/launch.sh --port 9000      # custom port

set -euo pipefail

WITH_DOCKER=0
NO_PULL=0
PORT=8765
NO_BROWSER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-docker) WITH_DOCKER=1; shift ;;
    --no-pull)     NO_PULL=1;     shift ;;
    --port)        PORT="$2";     shift 2 ;;
    --no-browser)  NO_BROWSER=1;  shift ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

C_CYAN='\033[36m'; C_GREEN='\033[32m'; C_YELLOW='\033[33m'; C_RED='\033[31m'; C_RESET='\033[0m'
section() { echo; echo -e "${C_CYAN}=== $* ===${C_RESET}"; }
ok()      { echo -e "  ${C_GREEN}[ok]${C_RESET} $*"; }
warn()    { echo -e "  ${C_YELLOW}[warn]${C_RESET} $*"; }
fail()    { echo; echo -e "  ${C_RED}[error]${C_RESET} $*" >&2; exit 1; }

# Locate repo root (one above this script).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

section "Locating install"
[[ -d .git ]] || fail "Not inside an ai-investing checkout: $REPO_ROOT"
ok "Using $REPO_ROOT"

# 2. venv
section "Checking virtual environment"
if [[ ! -x .venv/bin/python ]]; then
  warn ".venv not found - creating one"
  command -v python3 >/dev/null || fail "python3 not on PATH"
  python3 -m venv .venv
  .venv/bin/python -m pip install -e ".[dev]"
  ok "venv created"
else
  ok ".venv ready"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "venv activated"

# 3. .env
section "Validating .env"
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    warn ".env created from .env.example - fill in your Alpaca keys"
  else
    fail ".env not found and no .env.example to seed from"
  fi
fi
if grep -qE '^[[:space:]]*ALPACA_PAPER_KEY_ID[[:space:]]*=[[:space:]]*\S' .env \
   && grep -qE '^[[:space:]]*ALPACA_PAPER_SECRET[[:space:]]*=[[:space:]]*\S' .env; then
  ok "Alpaca paper keys present in .env"
else
  warn "Alpaca paper keys appear blank in .env"
fi

# 4. git pull
if [[ $NO_PULL -eq 0 ]]; then
  section "Syncing latest changes"
  if command -v git >/dev/null; then
    branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    dirty=$(git status --porcelain 2>/dev/null || echo "")
    if [[ -n "$dirty" ]]; then
      warn "Working tree has local changes - skipping pull"
    elif [[ "$branch" != "main" ]]; then
      warn "On branch '$branch' (not main) - skipping pull"
    else
      pre=$(git rev-parse HEAD)
      git fetch origin main --quiet 2>/dev/null || true
      behind=$(git rev-list --count "HEAD..origin/main" 2>/dev/null || echo 0)
      if [[ "$behind" -eq 0 ]]; then
        ok "Already up to date"
      else
        git pull --ff-only origin main --quiet
        post=$(git rev-parse HEAD)
        ok "Updated ${pre:0:7} -> ${post:0:7}"
        if git diff --name-only "$pre..$post" | grep -q '^pyproject\.toml$'; then
          ok "pyproject.toml changed - reinstalling deps"
          .venv/bin/python -m pip install -e ".[dev]" --quiet
        fi
      fi
    fi
  fi
fi

# 5. Docker (optional)
if [[ $WITH_DOCKER -eq 1 ]]; then
  section "Starting Docker stack"
  command -v docker >/dev/null || fail "docker not on PATH"
  if docker info >/dev/null 2>&1; then
    docker compose -f infra/docker/docker-compose.yml --env-file .env up -d
    ok "Docker stack is up"
  else
    warn "Docker daemon not responding - continuing without it"
  fi
fi

# 6. Boot orchestrator (warm up Ollama, pull models, create data dirs, doctor).
# Same code path the Windows launch.ps1 hits — keeps both platforms in lockstep.
# We deliberately omit --quiet so the per-step [ok]/[!!]/[XX] summary always
# lands on the user's screen above any error banner. Tee combined stdout+stderr
# to a permanent log file so the user can paste it into a bug report even if
# the terminal scrollback is gone.
section "Warming up the stack"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
mkdir -p data/cockpit
boot_log="data/cockpit/boot_launcher.log"
set +e
.venv/bin/python -u -m tools.boot 2>&1 | tee "$boot_log"
boot_exit=${PIPESTATUS[0]}
set -e
if [[ $boot_exit -ne 0 ]]; then
  case "$boot_exit" in
    2) hint="One or more boot steps failed (look for [XX] above)." ;;
    3) hint="The boot orchestrator itself crashed (Python traceback above)." ;;
    *) hint="Unexpected exit code from tools.boot." ;;
  esac
  # Best-effort: surface the failed step name from data/cockpit/boot.json.
  failed_step=""
  if [[ -f data/cockpit/boot.json ]] && command -v python3 >/dev/null; then
    failed_step=$(python3 -c "import json,sys
try:
    d=json.load(open('data/cockpit/boot.json'))
    rows=[r for r in d.get('results', []) if r.get('status')=='failed']
    print('; '.join(f\"{r['name']}: {r['message']}\" for r in rows))
except Exception:
    pass" 2>/dev/null || true)
  fi
  # If the orchestrator died before writing boot.json (typical for exit 1),
  # echo the tail of the captured log so the user sees the actual error.
  if [[ -z "$failed_step" && -f "$boot_log" ]]; then
    echo ""
    echo "  --- last 25 lines of $boot_log ---"
    tail -n 25 "$boot_log" || true
    echo "  --- end log ---"
  fi
  if [[ -n "$failed_step" ]]; then
    fail "boot orchestrator failed (exit $boot_exit). $hint
  Failed step(s): $failed_step
  Full log: $boot_log"
  else
    fail "boot orchestrator failed (exit $boot_exit). $hint
  Full log: $boot_log"
  fi
fi

# 7. Cockpit
section "Starting cockpit"
ok "Cockpit will be available at http://127.0.0.1:$PORT"
echo
echo "Press Ctrl+C to stop."
echo
ARGS=(tools/cockpit.py --port "$PORT")
[[ $NO_BROWSER -eq 1 ]] && ARGS+=(--no-browser)
exec .venv/bin/python "${ARGS[@]}"
