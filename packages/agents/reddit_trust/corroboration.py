"""News-corroboration gate.

The pattern this stops: Reddit lights up about ``$GME`` because one
viral post + a few burner accounts hype it. There is *no* news from a
real outlet to explain the move -- it's purely social.

The gate rules:

  1. Count how many *non-Reddit* news items mention the symbol within
     the corroboration window (default 24h).
  2. If the count is below :data:`MIN_NEWS_HEADLINES`, the symbol is
     uncorroborated.
  3. We grade the corroboration on a 0..1 scale so the cockpit can
     show "weakly corroborated" vs. "well corroborated".

The gate is *advisory* for portfolio candidates (user already owns it,
they should hear about the chatter) and *blocking* for Reddit-only
candidates whose trust weight is below :data:`HIGH_TRUST_FLOOR` -- a
high-trust Reddit-only signal can still pass, but the dashboard tells
the user it's uncorroborated so they can decide.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

# Items below this many corroborating headlines fail the gate.
MIN_NEWS_HEADLINES = 1
# Items at or above this many headlines are "strongly corroborated".
STRONG_NEWS_HEADLINES = 3
# Reddit-only signals with weight at or above this can pass the gate
# uncorroborated (will be tagged for the dashboard).
HIGH_TRUST_FLOOR = 0.75
# Default corroboration window.
DEFAULT_WINDOW_HOURS = 24


@dataclass(frozen=True)
class CorroborationResult:
    """Per-symbol verdict from the gate."""

    symbol: str
    news_headlines: int  # how many non-Reddit headlines matched
    corroboration_score: float  # [0, 1] for the cockpit gauge
    passes: bool  # whether this symbol clears the gate
    reason: str  # short string explaining the verdict


class NewsCorroborationGate:
    """Indexes a batch of news items by symbol and answers per-symbol
    corroboration queries.

    The indexing happens once at construction time so the
    :meth:`check` call is O(1) per symbol.
    """

    def __init__(
        self,
        news_items: Iterable,
        *,
        window_hours: float = DEFAULT_WINDOW_HOURS,
        now: datetime | None = None,
    ) -> None:
        self._window = timedelta(hours=window_hours)
        self._now = now or datetime.now(UTC)
        self._index: dict[str, list] = {}
        for it in news_items:
            # We only count *non-Reddit* sources. A NewsItem.source like
            # 'reddit/wallstreetbets' is filtered out because corroborating
            # Reddit with Reddit defeats the purpose.
            src = getattr(it, "source", "") or ""
            if src.startswith("reddit/"):
                continue
            sym = getattr(it, "symbol", None)
            if not sym:
                continue
            ts = getattr(it, "ts", None)
            if ts is None:
                continue
            # Make sure we compare UTC-aware against UTC-aware.
            if ts.tzinfo is None:
                continue
            if self._now - ts > self._window:
                continue
            self._index.setdefault(sym.upper(), []).append(it)

    def headlines_for(self, symbol: str) -> int:
        return len(self._index.get(symbol.upper(), []))

    def check(
        self,
        symbol: str,
        *,
        reddit_trust_weight: float | None = None,
        is_portfolio: bool = False,
    ) -> CorroborationResult:
        """Decide whether ``symbol`` clears the corroboration gate."""
        sym = symbol.upper()
        n = self.headlines_for(sym)

        # 0..1 score: 0 headlines = 0, STRONG_NEWS_HEADLINES+ = 1.0.
        if STRONG_NEWS_HEADLINES <= 0:  # pragma: no cover - defensive
            score = 1.0 if n > 0 else 0.0
        else:
            score = min(1.0, n / float(STRONG_NEWS_HEADLINES))

        # Portfolio candidates: always pass, the user owns it and
        # deserves to see the chatter. We still report the score.
        if is_portfolio:
            return CorroborationResult(
                symbol=sym,
                news_headlines=n,
                corroboration_score=score,
                passes=True,
                reason=(
                    f"portfolio symbol -- gate is advisory ({n} news)"
                ),
            )

        # Strong corroboration: pass unconditionally.
        if n >= MIN_NEWS_HEADLINES:
            return CorroborationResult(
                symbol=sym,
                news_headlines=n,
                corroboration_score=score,
                passes=True,
                reason=f"{n} non-Reddit headline(s) within window",
            )

        # No corroborating news. Only pass if Reddit-side trust is
        # very high.
        if (
            reddit_trust_weight is not None
            and reddit_trust_weight >= HIGH_TRUST_FLOOR
        ):
            return CorroborationResult(
                symbol=sym,
                news_headlines=n,
                corroboration_score=score,
                passes=True,
                reason=(
                    f"no news but Reddit trust {reddit_trust_weight:.2f} "
                    f">= floor {HIGH_TRUST_FLOOR:.2f}"
                ),
            )

        return CorroborationResult(
            symbol=sym,
            news_headlines=n,
            corroboration_score=score,
            passes=False,
            reason=(
                f"no non-Reddit news in last {self._window.total_seconds() / 3600:.0f}h "
                f"and Reddit trust {reddit_trust_weight or 0:.2f} "
                f"below floor {HIGH_TRUST_FLOOR:.2f}"
            ),
        )
