# Remote control (Phase 36c + 36d)

Let the Perplexity agent observe and steer your locally-running cockpit
from chat. Designed for the "Windows PC at home, always on" case.

## One-command launch (Phase 36d)

```powershell
cd C:\path\to\ai-investing
.\tools\start_cockpit.ps1
```

Or double-click `tools\start_cockpit.cmd` from File Explorer.

The single launcher will:

1. Ensure `COCKPIT_REMOTE_TOKEN` exists (generates one if missing).
2. Start uvicorn (cockpit) in a fresh PowerShell window with the token
   already in its environment.
3. Wait until the cockpit is reachable on 127.0.0.1:8000.
4. Download `cloudflared.exe` to `tools\bin\` if missing.
5. Start a Cloudflare quick-tunnel and parse the public URL.
6. Write the URL+token to `data/cockpit/remote_handle.json`.
7. Force-push that handle to the `cockpit-handle` branch on GitHub so
   the agent can auto-discover the current URL without you pasting
   anything.
8. Block on the tunnel. Ctrl+C in this window stops the tunnel; the
   cockpit keeps running in its own window.

After launch, just say to the agent: *"connect to my cockpit"* — it
will pull the handle from GitHub and dial in.

Flags:

- `-NewToken` rotates the remote token (use if you suspect a leak).
- `-NoPublish` skips the GitHub publish step (URL stays local only).

## Legacy / manual flow (Phase 36c)

If you prefer to start the cockpit yourself:

```powershell
# Window 1 -- cockpit:
.\.venv\Scripts\python.exe -m uvicorn packages.cockpit.web.server:app --host 127.0.0.1 --port 8000

# Window 2 -- tunnel only:
.\tools\cockpit_tunnel.ps1
```

This prints the URL+token for you to paste into chat manually.

> **Note on quick-tunnel URLs:** they change every restart. The
> `start_cockpit.ps1` launcher handles this transparently by publishing
> the new URL to the `cockpit-handle` branch. If you want a *stable*
> hostname instead, create a Cloudflare named tunnel and set
> `CLOUDFLARED_TUNNEL_NAME=<your-tunnel>` — the script will use it.

## Agent-side discovery

From the sandbox (or any machine with `gh` auth to the repo):

```bash
python tools/cockpit_discover.py            # full JSON
python tools/cockpit_discover.py --field url
python tools/cockpit_discover.py --field token
```

The handle is fetched from `refs/heads/cockpit-handle` at
`data/cockpit/remote_handle.json`.

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
.\tools\start_cockpit.ps1 -NewToken
```

This generates a new token AND restarts the cockpit child window with
the new value baked into its env.

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
