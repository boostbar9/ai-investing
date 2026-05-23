# Local setup — Windows 11 + AMD GPU

> Target reader: someone on Windows 11 with an AMD Radeon (RX 7900 XT/XTX
> or similar 16-24GB card) and 32GB+ system RAM who wants ai-investing
> running on `http://localhost:3000` with paper trades flowing.
>
> If you're on NVIDIA + WSL2, follow Path B with `--gpu nvidia` substitutions.
> If you're on Mac, see `local-setup-mac.md` (TODO).

There are two supported paths. **Start with Path A.** It's faster to set
up, and the LLM inference loop in this platform is not latency-critical
(decisions pace at minutes, not milliseconds). Move to Path B later if you
want GPU acceleration for the agents.

---

## Prerequisites (both paths)

1. **Windows 11 Home or Pro**, current updates installed.
2. **At least 60 GB free** on whichever SSD will hold Docker volumes + LLM
   weights. The `rx_7900_xt` profile needs ~30 GB of model files; Docker
   volumes + Postgres + Timescale data eats another ~10-20 GB once you're
   running paper trades for a few weeks.
3. **A GitHub account** with SSH or Personal Access Token configured.
4. **WSL2 enabled** (we need this for both Docker Desktop and the
   ai-investing repo tooling):
   ```powershell
   wsl --install
   wsl --set-default-version 2
   ```
   Reboot when prompted.

## One-time tool install

Install in this order, accepting all defaults unless noted:

1. **Git for Windows** — https://git-scm.com/download/win
2. **Docker Desktop for Windows** — https://www.docker.com/products/docker-desktop
   - In Settings → General, check "Use the WSL 2 based engine."
   - In Settings → Resources → WSL Integration, enable your default
     distro.
   - Allocate ≥ 16 GB RAM and ≥ 6 CPUs to Docker (Settings → Resources →
     Advanced).
3. **Node.js 22 LTS** — https://nodejs.org (used by the cockpit)
4. **Python 3.12** — https://www.python.org/downloads/windows/
   - Check "Add python.exe to PATH" during install.
5. **uv** (Python package manager) — open PowerShell:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
6. **pnpm** — open PowerShell:
   ```powershell
   corepack enable
   ```
7. **GitHub CLI** — https://cli.github.com (for the issue/PR workflow)

## Clone the repo

In PowerShell or Windows Terminal:

```powershell
cd $HOME
git clone https://github.com/boostbar9/ai-investing.git
cd ai-investing
copy .env.example .env
```

Open `.env` in Notepad / VS Code and set at minimum:

```
HARDWARE_PROFILE=rx_7900_xt
APP_ENV=dev
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/aiinvesting
DRAGONFLY_URL=redis://localhost:6379
OLLAMA_HOST=http://localhost:11434
ALPACA_PAPER_KEY_ID=     # fill in after creating Alpaca account
ALPACA_PAPER_SECRET=     # ditto
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

You can leave the market-data and notification keys blank for now — the
code degrades gracefully when they're absent.

---

## Path A — Ollama on Windows (CPU inference, easy)

Use this path first. It works regardless of GPU support and lets you
verify the whole stack is healthy before fighting with ROCm.

### A.1 Install Ollama

1. Download from https://ollama.com/download/OllamaSetup.exe
2. Run the installer. It starts a background service on
   `http://localhost:11434`.
3. Confirm it's up:
   ```powershell
   curl http://localhost:11434/api/tags
   ```
   You should get `{"models":[]}`.

### A.2 Pull the models for your profile

For `rx_7900_xt`:

```powershell
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull deepseek-r1:14b
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q4_K_M
```

Total download: ~30 GB. Allow 30-60 minutes on a decent connection.

> If you set `HARDWARE_PROFILE=cpu_only` or `balanced` in `.env`, run
> `make pull-models PROFILE=<that>` from WSL — see the
> [hardware-tiers doc](../hardware-tiers.md) for the full list.

### A.3 Smoke-test inference

```powershell
ollama run qwen2.5:7b-instruct-q4_K_M "Summarize today's SPY action in one sentence."
```

On a Ryzen 7 5700X3D with no GPU acceleration, expect ~5-15 tokens/sec.
Slow but functional.

### A.4 Bring up the rest of the stack

From WSL (open Ubuntu from the Start menu):

```bash
cd /mnt/c/Users/<you>/ai-investing   # or wherever you cloned
make setup                            # one-shot: deps + db migrations + first build
docker compose -f infra/docker/docker-compose.yml up -d
```

Verify:

```bash
curl http://localhost:8000/health/detail | jq
curl http://localhost:3000              # cockpit homepage
```

You're done. Cockpit at http://localhost:3000, API docs at
http://localhost:8000/docs, Grafana at http://localhost:3001 (user
`admin`, password `admin`).

---

## Path B — WSL2 + ROCm (GPU acceleration on RX 7900 XT)

Switch to this path once Path A is working and you want **4-6× faster**
agent decisions.

### B.1 Verify the GPU is visible from WSL

In WSL:

```bash
ls /dev/dxg            # should exist on Win 11 WSLg
sudo apt update && sudo apt install -y radeontop
radeontop                # should show your 7900 XT
```

### B.2 Install ROCm 6.x inside WSL

Follow the AMD official guide for the version current at install time —
**not** the version in this runbook (the URL pattern below is stable but
the package list changes):

> https://rocm.docs.amd.com/projects/install-on-linux/en/latest/

Quick path for Ubuntu 22.04 in WSL2:

```bash
sudo apt update
sudo apt install -y wget gnupg2
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_<VERSION>_all.deb
sudo apt install -y ./amdgpu-install_<VERSION>_all.deb
sudo amdgpu-install --usecase=rocm --no-dkms
sudo usermod -a -G render,video $USER
```

Restart WSL: from PowerShell, `wsl --shutdown` then reopen.

### B.3 Verify ROCm sees the GPU

```bash
rocm-smi                # should list the 7900 XT
```

If `rocm-smi` doesn't see the card, **stop here** and stay on Path A.
ROCm-on-WSL for RDNA3 is supported but fragile; if it doesn't work
out-of-the-box on your kernel + driver combo, the fix is usually a
several-hour deep dive that's not worth it for the latency win.

### B.4 Run Ollama inside WSL with GPU

Stop the Windows Ollama service (System tray → Ollama → Quit), then in
WSL:

```bash
curl -fsSL https://ollama.com/install.sh | sh
HSA_OVERRIDE_GFX_VERSION=11.0.0 ollama serve   # in one terminal
ollama pull deepseek-r1:14b                    # in another
ollama run deepseek-r1:14b "Hello"
```

The `HSA_OVERRIDE_GFX_VERSION=11.0.0` env var is required on RDNA3 to
force the GPU code path. Add it to your `~/.bashrc` once it's working.

Update `.env` to point at the WSL Ollama if needed (usually
`OLLAMA_HOST=http://localhost:11434` works because WSL2 forwards
localhost).

Expect **40-70 tok/s** on the 7900 XT now.

---

## Common issues

### Docker Desktop runs out of memory

Settings → Resources → Advanced → Memory ≥ 16 GB. The combined stack
(Postgres + Timescale + Dragonfly + Temporal + Grafana + n8n + Prometheus
+ otel-collector + api + worker + cockpit) wants ~6-8 GB at idle.

### `make setup` fails on missing build tools

In PowerShell as admin:

```powershell
npm install --global windows-build-tools
```

Then retry in WSL.

### Ollama is slow on CPU

Expected. The model selection in `rx_7900_xt` profile already biases
toward the smaller variants (14B and 8B) for exactly this reason. If you
want faster CPU inference, set `HARDWARE_PROFILE=balanced` in `.env`
which pins everything at 7-8B.

### `pnpm` command not found inside WSL

WSL has its own Node install. Either install pnpm via corepack inside
WSL:

```bash
corepack enable
```

…or run the cockpit build from PowerShell (Windows-side) and only run the
Python services from WSL.

### Postgres complains about disk space

`docker system prune -a` clears dangling images. The Postgres + Timescale
data volume grows ~1 GB/week of tick data once the ingestion job is
running. Plan for ~50 GB after 12 months.

---

## What to do once it's up

1. Create an Alpaca paper account → https://alpaca.markets/
2. Paste the API keys into `.env`, restart the api + worker containers.
3. Open the cockpit at http://localhost:3000.
4. Tap "Register passkey" on the sign-in page once (Touch ID equivalent
   via Windows Hello on a supported device).
5. Approve your first paper trade from the cockpit.
6. Set a calendar reminder 60 trading days from now — that's the §16
   acceptance threshold for graduating to live trading.

If anything is unclear, the [Day-1 operator checklist](./day-1-operator-checklist.md)
is the consolidated "did I forget anything?" view.
