#!/usr/bin/env bash
# Safe in-place updater for ai-investing on macOS / Linux.
#
# Pulls latest code, reinstalls Python deps only if pyproject.toml
# changed, warns about new env vars, and re-runs the doctor smoke test.
#
# Usage:
#   cd ~/ai-investing && ./scripts/update.sh
#   # or, run anywhere:
#   curl -fsSL https://raw.githubusercontent.com/boostbar9/ai-investing/main/scripts/update.sh | bash

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-$(pwd)}"
SKIP_DOCTOR="${SKIP_DOCTOR:-0}"
FORCE="${FORCE:-0}"

if [ -t 1 ]; then
  C_CYAN='\033[36m'; C_GREEN='\033[32m'; C_YELLOW='\033[33m'; C_RED='\033[31m'; C_RESET='\033[0m'
else
  C_CYAN=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_RESET=''
fi
section() { printf "\n${C_CYAN}=== %s ===${C_RESET}\n" "$1"; }
ok()      { printf "  ${C_GREEN}[ok]${C_RESET} %s\n" "$1"; }
warn()    { printf "  ${C_YELLOW}[warn]${C_RESET} %s\n" "$1"; }
fail()    { printf "\n  ${C_RED}[error]${C_RESET} %s\n\n" "$1"; exit 1; }

hash_file() {
  if [ -f "$1" ]; then
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "$1" | awk '{print $1}'
    else
      shasum -a 256 "$1" | awk '{print $1}'
    fi
  fi
}

# ----------------------------------------------------------------------
# 1. Locate install
# ----------------------------------------------------------------------
section "Locating install"
if [ ! -d "$INSTALL_DIR/.git" ]; then
  if [ -d "$HOME/ai-investing/.git" ]; then
    INSTALL_DIR="$HOME/ai-investing"
  else
    fail "Not inside an ai-investing checkout and \$HOME/ai-investing doesn't exist. cd to your install first, or run scripts/install.sh."
  fi
fi
cd "$INSTALL_DIR"
ok "Using $INSTALL_DIR"

# ----------------------------------------------------------------------
# 2. Safety checks
# ----------------------------------------------------------------------
section "Safety checks"

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "main" ]; then
  if [ "$FORCE" != "1" ]; then
    fail "On branch '$branch', not 'main'. Switch with 'git switch main' or re-run with FORCE=1."
  fi
  warn "On branch '$branch' (FORCE=1 set, continuing)"
fi

if [ -n "$(git status --porcelain)" ]; then
  if [ "$FORCE" != "1" ]; then
    echo
    echo "  Uncommitted changes detected:"
    git status --short
    fail "Commit, stash, or discard local changes first. Or re-run with FORCE=1."
  fi
  warn "Continuing with uncommitted changes (FORCE=1)"
fi
ok "Working tree clean, on main"

# ----------------------------------------------------------------------
# 3. Snapshot pre-pull state
# ----------------------------------------------------------------------
pre_hash=$(git rev-parse HEAD)
pre_toml=$(hash_file pyproject.toml || true)
pre_env_example=$(hash_file .env.example || true)

# ----------------------------------------------------------------------
# 4. Pull
# ----------------------------------------------------------------------
section "Fetching updates"
git fetch origin main --quiet
behind=$(git rev-list --count HEAD..origin/main)
if [ "$behind" = "0" ]; then
  ok "Already up to date - nothing to pull"
else
  ok "$behind new commit(s) on origin/main"
  git pull --ff-only origin main
fi
post_hash=$(git rev-parse HEAD)

# ----------------------------------------------------------------------
# 5. Update Python deps if pyproject.toml changed
# ----------------------------------------------------------------------
section "Python dependencies"

VENV_PY="$INSTALL_DIR/.venv/bin/python"
VENV_PIP="$INSTALL_DIR/.venv/bin/pip"

if [ ! -x "$VENV_PY" ]; then
  warn ".venv not found - running fresh dependency install"
  python3 -m venv .venv
  "$VENV_PY" -m pip install --upgrade pip --quiet
  "$VENV_PIP" install -e ".[dev]" --quiet
  ok "Dependencies installed"
else
  post_toml=$(hash_file pyproject.toml || true)
  if [ "$pre_toml" != "$post_toml" ]; then
    ok "pyproject.toml changed - reinstalling dependencies"
    "$VENV_PIP" install -e ".[dev]" --quiet
    ok "Dependencies updated"
  else
    ok "pyproject.toml unchanged - skipping pip install"
  fi
fi

# ----------------------------------------------------------------------
# 6. Warn about new env vars
# ----------------------------------------------------------------------
section "Environment variables"
if [ -f .env.example ]; then
  post_env_example=$(hash_file .env.example || true)
  if [ -n "$pre_env_example" ] && [ "$pre_env_example" != "$post_env_example" ]; then
    warn ".env.example changed since your last update. Compare with your .env:"
    echo "    diff .env .env.example"
    echo "    (or 'git diff $pre_hash..$post_hash -- .env.example')"
  else
    ok ".env.example unchanged"
  fi
fi

# ----------------------------------------------------------------------
# 7. Doctor
# ----------------------------------------------------------------------
if [ "$SKIP_DOCTOR" != "1" ] && [ -x "$VENV_PY" ]; then
  section "Running doctor"
  PYTHONPATH=. "$VENV_PY" tools/doctor.py || \
    warn "Doctor reported issues - review the output above."
fi

# ----------------------------------------------------------------------
# 8. Changelog
# ----------------------------------------------------------------------
if [ "$pre_hash" != "$post_hash" ]; then
  section "What changed"
  git log --oneline "$pre_hash..$post_hash"
fi

section "Update complete"
cat <<EOF

${C_CYAN}You can now re-run the nightly paper job:${C_RESET}
  source .venv/bin/activate
  PYTHONPATH=. python tools/paper_trade.py --strategy ensemble

EOF
