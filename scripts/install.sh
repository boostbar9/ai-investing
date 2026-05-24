#!/usr/bin/env bash
# One-command installer for ai-investing on macOS / Linux.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/boostbar9/ai-investing/main/scripts/install.sh | bash
#
# Or, if already cloned:
#   ./scripts/install.sh

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-$HOME/ai-investing}"
REPO_URL="${REPO_URL:-https://github.com/boostbar9/ai-investing.git}"
SKIP_DOCTOR="${SKIP_DOCTOR:-0}"

# Colors (fall back to plain text if not a TTY).
if [ -t 1 ]; then
  C_CYAN='\033[36m'; C_GREEN='\033[32m'; C_YELLOW='\033[33m'; C_RED='\033[31m'; C_RESET='\033[0m'
else
  C_CYAN=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_RESET=''
fi

section() { printf "\n${C_CYAN}=== %s ===${C_RESET}\n" "$1"; }
ok()      { printf "  ${C_GREEN}[ok]${C_RESET} %s\n" "$1"; }
warn()    { printf "  ${C_YELLOW}[warn]${C_RESET} %s\n" "$1"; }
fail()    { printf "\n  ${C_RED}[error]${C_RESET} %s\n\n" "$1"; exit 1; }

# ----------------------------------------------------------------------
# 1. Prereq checks
# ----------------------------------------------------------------------
section "Checking prerequisites"

if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 is not on PATH. Install Python 3.12+ (https://python.org/downloads/ or 'brew install python@3.12')."
fi

PY_VER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 12 ]; }; then
  fail "Need Python 3.12+. Found: $PY_VER."
fi
ok "Python $PY_VER"

if ! command -v git >/dev/null 2>&1; then
  fail "git is not on PATH. Install with your package manager ('brew install git', 'apt install git', etc.)."
fi
ok "$(git --version)"

# ----------------------------------------------------------------------
# 2. Clone or update repo
# ----------------------------------------------------------------------
section "Getting the code"

if [ -d "$INSTALL_DIR/.git" ]; then
  ok "Repo already exists at $INSTALL_DIR - pulling latest"
  git -C "$INSTALL_DIR" pull --ff-only
elif [ -e "$INSTALL_DIR" ]; then
  fail "$INSTALL_DIR exists but is not a git repo. Move or delete it, then re-run."
else
  ok "Cloning into $INSTALL_DIR"
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# ----------------------------------------------------------------------
# 3. Virtual environment
# ----------------------------------------------------------------------
section "Creating virtual environment"

if [ ! -x "$INSTALL_DIR/.venv/bin/python" ]; then
  python3 -m venv .venv
  ok ".venv created"
else
  ok ".venv already exists"
fi

VENV_PY="$INSTALL_DIR/.venv/bin/python"
VENV_PIP="$INSTALL_DIR/.venv/bin/pip"

# ----------------------------------------------------------------------
# 4. Install dependencies
# ----------------------------------------------------------------------
section "Installing Python dependencies (this takes 2-5 minutes)"

"$VENV_PY" -m pip install --upgrade pip --quiet
ok "pip upgraded"

# Drop --quiet here so build errors are visible if they happen.
"$VENV_PIP" install -e ".[dev]"
ok "ai-investing + dev extras installed"

# ----------------------------------------------------------------------
# 5. Environment file
# ----------------------------------------------------------------------
section "Setting up .env"

if [ ! -f .env ]; then
  cp .env.example .env
  ok ".env created from .env.example"
  warn "Edit $INSTALL_DIR/.env and fill in your Alpaca paper keys before running paper trading."
else
  ok ".env already exists (not overwriting)"
fi

# ----------------------------------------------------------------------
# 6. Doctor smoke test
# ----------------------------------------------------------------------
if [ "$SKIP_DOCTOR" != "1" ]; then
  section "Running doctor"
  PYTHONPATH=. "$VENV_PY" tools/doctor.py || \
    warn "Doctor reported issues - review the output above. Most often this means .env still needs Alpaca keys."
fi

# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------
section "Install complete"

cat <<EOF

Project installed at: ${C_YELLOW}$INSTALL_DIR${C_RESET}

${C_CYAN}Next steps:${C_RESET}
  1. Edit .env and add your Alpaca paper keys
     (from https://app.alpaca.markets/paper/dashboard/overview)
  2. Activate the venv:    cd $INSTALL_DIR && source .venv/bin/activate
  3. Download market data: PYTHONPATH=. python -m packages.data.pretrain
  4. First dry-run:        python tools/paper_trade.py --strategy ensemble --dry-run
  5. Open the cockpit GUI: PYTHONPATH=. python tools/cockpit.py
     (then visit http://127.0.0.1:8765 - opens automatically)

EOF
