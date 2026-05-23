"""Plot equity curves for the four strategies on real data."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from packages.backtests.harness import run_backtest
from packages.strategies import (
    MeanReversion,
    SectorRotation,
    SentimentOverlay,
    TrendFollowing,
)
from tools.validate_real_data import load_price_panel


def main() -> None:
    core = ["SPY", "QQQ", "IWM", "DIA"]
    megacaps = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]

    panel_main = load_price_panel(core + megacaps)
    panel_sectors = load_price_panel(sectors)

    runs = [
        ("Trend Following", TrendFollowing(fast=50, slow=200), panel_main),
        ("Sector Rotation", SectorRotation(top_n=3), panel_sectors),
        ("Mean Reversion", MeanReversion(), panel_main),
        (
            "Sentiment Overlay",
            SentimentOverlay(
                base=TrendFollowing(fast=50, slow=200),
                sentiment=dict.fromkeys(panel_main.columns, 1.0),
            ),
            panel_main,
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.flatten()

    for i, (name, strat, panel) in enumerate(runs):
        ax = axes[i]
        bt = run_backtest(strat, panel)
        eq = bt.equity_curve / bt.equity_curve.iloc[0]
        ax.plot(eq.index, eq.values, label=name, linewidth=1.8, color="#1f77b4")

        # Benchmark aligned to this panel
        spy_panel = panel.get("SPY", panel.iloc[:, 0])
        bench = spy_panel / spy_panel.iloc[0]
        ax.plot(bench.index, bench.values, label="SPY buy & hold",
                linewidth=1.2, color="#888888", linestyle="--")

        ax.set_title(
            f"{name}  |  Sharpe {bt.sharpe:.2f}  CAGR {bt.cagr*100:.1f}%  "
            f"MaxDD {bt.max_drawdown*100:.1f}%"
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylabel("Equity (norm)")

    fig.suptitle(
        "Strategy validation on real data — daily Parquet panel\n"
        "(Costs: 6 bps round-trip — see harness DEFAULT_COST_BPS)",
        fontsize=13,
    )
    fig.tight_layout()
    out = Path("docs/validation-curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
