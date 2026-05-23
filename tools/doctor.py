"""make doctor — friendly readiness check for the training pipeline.

Usage:
    python -m tools.doctor             # human-readable
    python -m tools.doctor --json      # machine-readable

Exit code is 0 even when sources are missing keys: the goal is to TELL the
operator what's enabled vs gated, not to block them. We only exit non-zero
on hard configuration errors (e.g. Python deps missing).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def check_python_deps() -> tuple[bool, str]:
    missing: list[str] = []
    for mod in ("pandas", "pyarrow", "httpx", "pydantic", "fastapi"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return False, f"missing: {', '.join(missing)} (run `make setup-python`)"
    return True, "ok"


def check_data_sources() -> dict[str, dict]:
    """Report which adapters can actually fetch data right now."""
    return {
        "yfinance": {
            "ok": True,
            "needs": [],
            "note": "free, no key required — primary daily source",
        },
        "alpaca_data": {
            "ok": bool(os.getenv("ALPACA_PAPER_KEY_ID") and os.getenv("ALPACA_PAPER_SECRET")),
            "needs": ["ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET"],
            "note": "free paper account at https://app.alpaca.markets — unlocks intraday bars",
        },
        "fred": {
            "ok": bool(os.getenv("FRED_API_KEY")),
            "needs": ["FRED_API_KEY"],
            "note": "free key at https://fred.stlouisfed.org/docs/api/api_key.html — unlocks macro/regime",
        },
        "sentiment": {
            "ok": True,
            "needs": [],
            "note": "RSS works without keys; REDDIT_CLIENT_ID adds Reddit",
        },
    }


def check_parquet_cache() -> dict[str, int]:
    root = Path(os.getenv("DATA_PARQUET_ROOT", "data/parquet"))
    out = {}
    for sub in ("daily", "intraday", "macro"):
        d = root / sub
        out[sub] = len(list(d.glob("*.parquet"))) if d.exists() else 0
    sent = root / "sentiment" / "latest.json"
    out["sentiment"] = 1 if sent.exists() else 0
    return out


def check_champion_params() -> dict:
    root = Path(os.getenv("DATA_PARAMS_ROOT", "data/params"))
    p = root / "champion.json"
    if not p.exists():
        return {"exists": False, "note": "run `make retune` after `make pretrain`"}
    try:
        return {"exists": True, "params": json.loads(p.read_text())}
    except Exception as e:
        return {"exists": True, "error": str(e)}


def print_human(report: dict) -> None:
    print(f"\n{GREEN}== ai-investing doctor =={RESET}\n")

    ok, msg = report["python_deps"]
    icon = f"{GREEN}\u2713{RESET}" if ok else f"{RED}\u2717{RESET}"
    print(f"{icon} python deps: {msg}")

    print(f"\n{DIM}data sources:{RESET}")
    for name, info in report["data_sources"].items():
        if info["ok"]:
            icon = f"{GREEN}\u2713{RESET}"
            line = f"  {icon} {name:<13} enabled"
        else:
            icon = f"{YELLOW}\u00b7{RESET}"
            line = f"  {icon} {name:<13} gated   (set {', '.join(info['needs'])})"
        print(line)
        if info.get("note"):
            print(f"      {DIM}{info['note']}{RESET}")

    cache = report["parquet_cache"]
    total = sum(cache.values())
    cache_icon = f"{GREEN}\u2713{RESET}" if total > 0 else f"{YELLOW}\u00b7{RESET}"
    print(f"\n{cache_icon} parquet cache: daily={cache['daily']}  intraday={cache['intraday']}  macro={cache['macro']}  sentiment={cache['sentiment']}")
    if total == 0:
        print(f"      {DIM}empty \u2014 run `make pretrain` to fill it{RESET}")

    champ = report["champion_params"]
    if champ["exists"]:
        print(f"{GREEN}\u2713{RESET} champion params: {champ.get('params', 'unreadable')}")
    else:
        print(f"{YELLOW}\u00b7{RESET} champion params: not yet trained")
        print(f"      {DIM}{champ.get('note', '')}{RESET}")

    enabled = [n for n, i in report["data_sources"].items() if i["ok"]]
    print(f"\n{DIM}Bottom line:{RESET} {len(enabled)}/4 data sources enabled. ", end="")
    if total == 0:
        print(f"Next: {GREEN}make pretrain{RESET}")
    elif not champ["exists"]:
        print(f"Next: {GREEN}make retune{RESET}")
    else:
        print(f"You're trained. {GREEN}make dev{RESET} starts the stack.")
    print()


def main() -> int:
    report = {
        "python_deps": check_python_deps(),
        "data_sources": check_data_sources(),
        "parquet_cache": check_parquet_cache(),
        "champion_params": check_champion_params(),
    }
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_human(report)
    return 0 if report["python_deps"][0] else 1


if __name__ == "__main__":
    sys.exit(main())
