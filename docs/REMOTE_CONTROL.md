# Remote control (Phase 36c)

Let the Perplexity agent observe and steer your locally-running cockpit
from chat. Designed for the "Windows PC at home, always on" case.

## TL;DR — 5-minute setup on Windows

```powershell
cd C:\path\to\ai-investing

# 1. Start the cockpit as you normally do (in its own window):
.\.venv\Scripts\python.exe -m uvicorn packages.cockpit.web.server:app --host 127.0.0.1 --port 8000

# 2. In a SECOND PowerShell window, launch the tunnel:
.\tools\cockpit_tunnel.ps1
```

The script will:

1. Verify the cockpit is up.
2. Generate `COCKPIT_REMOTE_TOKEN` if one isn't set (saved to your User
   env so it survives reboots).
3. Download `cloudflared.exe` to `tools\bin\` if it isn't present.
4. Start a Cloudflare quick-tunnel and print:

```
Public URL : https://random-name.trycloudflare.com
Token      : <32-char hex>

Tell the agent:
  My cockpit is at https://random-name.trycloudflare.com with token <…>
```

5. Paste that one line into the chat and I'll connect.

> **Important:** the quick-tunnel URL changes every time you restart
> the script. If you want a stable hostname, create a Cloudflare named
> tunnel and set `CLOUDFLARED_TUNNEL_NAME=<your-tunnel>` — the script
> will use it automatically.

## Security model

- The entire `/api/remote/*` surface is **fail-closed**: if
  `COCKPIT_REMOTE_TOKEN` is unset or shorter than 16 characters, every
  remote route returns **503 Service Unavailable**. A fresh checkout
  cannot accidentally expose control.
- The token is a 128-bit secret (32 hex chars) generated via the
  Windows CNG RNG (`RandomNumberGenerator.Create`).
- Comparison uses `hmac.compare_digest` to prevent timing leaks.
- The token must be sent on every request as either
  `Authorization: Bearer <token>` or `X-Cockpit-Token: <token>`.
- Liquidation requires an additional confirmation payload
  (`{"confirm": "LIQUIDATE"}`) on top of the token, so a leaked token
  + accidental request cannot drain positions.

If you ever suspect the token has leaked, rotate it:

```powershell
.\tools\cockpit_tunnel.ps1 -GenerateToken
```

…then restart the cockpit so it reloads the env var.

## Surface (what the agent can do)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/remote/health`    | Liveness probe (no auth). |
| `GET`  | `/api/remote/whoami`    | Auth check. |
| `GET`  | `/api/remote/snapshot`  | State + paper-loop status + log tail. |
| `GET`  | `/api/remote/log`       | Log tail (or `?download=1` for full file). |
| `POST` | `/api/remote/pause`     | Pause the bot. |
| `POST` | `/api/remote/resume`    | Resume (unpause). |
| `POST` | `/api/remote/loop/start`| Start paper loop. Body: `{strategy, dry_run}`. |
| `POST` | `/api/remote/loop/stop` | Stop paper loop. |
| `POST` | `/api/remote/liquidate` | Pause + clear intent. Body: `{"confirm":"LIQUIDATE"}`. |

Every mutating route returns the post-action `CockpitState` so the
agent can verify the effect without a follow-up poll.

## What the agent will NOT do without asking

- **Liquidate.** Always confirmed in chat first.
- **Strategy/risk changes.** Confirmed first.
- **Stop the loop** during what looks like an active position. Confirmed first.

The agent freely uses read-only routes (`/snapshot`, `/log`) and the
reversible pause/resume/start/stop pair.

## Troubleshooting

- **`/api/remote/health` returns `{"enabled": false}`** — the cockpit
  process doesn't see `COCKPIT_REMOTE_TOKEN`. Set it in your env and
  restart the cockpit process. The script writes the var at User
  scope; PowerShell windows opened before the script ran won't see it
  until they're reopened.
- **`401 missing remote token`** — request didn't include
  `Authorization: Bearer <token>` or `X-Cockpit-Token`.
- **`403 invalid remote token`** — token mismatch. Most often the
  cockpit was started before the new token was saved.
- **Tunnel URL never appears** — check `%TEMP%\cockpit_tunnel.log` for
  cloudflared errors. The most common cause is a corporate firewall
  blocking outbound to `*.cloudflare.com`.

## Manual smoke test from anywhere

```bash
TOKEN="<your-token>"
URL="https://your-tunnel.trycloudflare.com"

curl -H "Authorization: Bearer $TOKEN" "$URL/api/remote/snapshot" | jq
curl -X POST -H "Authorization: Bearer $TOKEN" "$URL/api/remote/pause"
curl -X POST -H "Authorization: Bearer $TOKEN" "$URL/api/remote/resume"
```
