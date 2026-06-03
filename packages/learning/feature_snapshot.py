"""Phase 34: candidate feature snapshot sink.

When the autonomy scorer evaluates a candidate, it sees a rich dict of
features (confidence, corroboration, reddit_trust, analyst signals,
insider flow, etc.). Those features are the inputs we want the
supervised ranker to learn from. But they're computed and thrown away
each sweep — only the resulting *score* and the bandit feature labels
survive into the audit trail.

This module persists the **full feature vector** at decision time so
the trainer can later join it against the realised outcomes file and
fit a P(EOD ≥ +0.5%) classifier.

Schema (one row per scored candidate per cycle)::

    {
      "ts":           "2026-06-02T19:43:26+00:00",
      "decision_id":  "...",
      "symbol":       "AAPL",
      "regime":       "bull",
      "features": {
          "confidence":            0.62,
          "corroborated":          1,        # 0/1
          "corroboration_score":   0.55,
          "reddit_trust":          0.71,
          "analyst_mean_rating":   2.1,
          "analyst_num":           14,
          "analyst_action":        "upgrade",   # str (label-encoded at fit time)
          "insider_form4_30d":     4,
          "insider_net_shares":   12500.0,
          "stocktwits_trending":   0,
          "yahoo_news_count":      7,
      },
    }

Join key with ``outcomes.jsonl`` is ``(decision_id, symbol)`` —
identical to ``pick_id`` derivation in the outcome labeler.

Write path is best-effort: a feature-snapshot I/O failure must never
poison a real cycle. Read path is tolerant of malformed lines.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# The canonical feature key set the ranker will train on. Kept in sync
# with ``autonomy._score_candidate``'s feature extraction. Adding a
# new feature: append the key here AND populate it in
# ``extract_features_from_candidate``. The trainer handles missing
# columns gracefully (NaN-fill), so old rows remain usable.
FEATURE_KEYS: tuple[str, ...] = (
    "confidence",
    "corroborated",
    "corroboration_score",
    "reddit_trust",
    "analyst_mean_rating",
    "analyst_num",
    "analyst_action",
    "insider_form4_30d",
    "insider_net_shares",
    "stocktwits_trending",
    "yahoo_news_count",
)

# Categorical features (string-valued). The trainer label-encodes
# these. Everything else is numeric.
CATEGORICAL_KEYS: frozenset[str] = frozenset({"analyst_action"})


def _default_path() -> Path:
    """Env-overridable so tests can isolate. Read at call time."""
    return Path(
        os.environ.get(
            "FEATURE_SNAPSHOT_PATH",
            "data/learning/feature_snapshots.jsonl",
        )
    )


def extract_features_from_candidate(c: Mapping[str, Any]) -> dict[str, Any]:
    """Pluck the model-input subset out of an autonomy candidate dict.

    Coerces types so the JSONL is uniform (bools→int, missing→None).
    Numeric coercion failures fall back to None so the trainer can
    NaN-fill rather than discarding the row.
    """
    def _num(key: str) -> float | None:
        v = c.get(key)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _int(key: str) -> int | None:
        v = c.get(key)
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return {
        "confidence":          _num("confidence"),
        "corroborated":        1 if c.get("corroborated") else 0,
        "corroboration_score": _num("corroboration_score"),
        "reddit_trust":        _num("reddit_trust"),
        "analyst_mean_rating": _num("analyst_mean_rating"),
        "analyst_num":         _int("analyst_num"),
        "analyst_action":      str(c.get("analyst_recent_action") or "") or None,
        "insider_form4_30d":   _int("insider_form4_30d"),
        "insider_net_shares":  _num("insider_net_shares"),
        "stocktwits_trending": 1 if c.get("stocktwits_trending") else 0,
        "yahoo_news_count":    _int("yahoo_news_count"),
    }


def append_snapshots(
    *,
    decision_id: str,
    regime: str,
    rows: Iterable[Mapping[str, Any]],
    ts: str | None = None,
    path: Path | None = None,
) -> int:
    """Append one snapshot per candidate.

    ``rows`` is an iterable of dicts that must carry ``symbol`` and
    the candidate feature keys. Returns count written. Best-effort.
    """
    target = path if path is not None else _default_path()
    ts = ts or datetime.now(UTC).isoformat(timespec="seconds")

    out: list[dict[str, Any]] = []
    for r in rows:
        sym = str(r.get("symbol") or "").upper().strip()
        if not sym:
            continue
        out.append(
            {
                "ts": ts,
                "decision_id": str(decision_id),
                "symbol": sym,
                "regime": str(regime or ""),
                "features": extract_features_from_candidate(r),
            }
        )
    if not out:
        return 0
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            for row in out:
                fh.write(
                    json.dumps(row, separators=(",", ":"), default=str) + "\n"
                )
    except OSError as exc:
        log.warning("feature snapshot write failed: %s", exc)
        return 0
    return len(out)


def iter_snapshots(path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield snapshots oldest-first. Skips malformed lines silently."""
    target = path if path is not None else _default_path()
    if not target.exists():
        return
    try:
        with open(target, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        log.debug("feature snapshot read failed: %s", exc)
        return


def load_snapshots(path: Path | None = None) -> list[dict[str, Any]]:
    """Eagerly load all snapshots into memory. Tiny wrapper for tests."""
    return list(iter_snapshots(path))
