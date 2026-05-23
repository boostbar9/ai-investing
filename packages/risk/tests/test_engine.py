from packages.risk.engine import PER_NAME_CAP, PER_SECTOR_CAP, Candidate, drawdown_halt, size_orders


def test_per_name_cap_enforced():
    out = size_orders(
        [Candidate("SPY", "Index", kelly_fraction=1.0, realized_vol=0.10)],
        regime="bull",
    )
    assert out[0].weight <= PER_NAME_CAP + 1e-9


def test_crisis_halts_sizing():
    out = size_orders(
        [Candidate("SPY", "Index", kelly_fraction=1.0, realized_vol=0.10)],
        regime="crisis",
    )
    assert out[0].weight == 0.0


def test_sector_cap_applied():
    cands = [
        Candidate(s, "Tech", kelly_fraction=1.0, realized_vol=0.10)
        for s in ["AAPL", "MSFT", "NVDA", "GOOG", "META", "AVGO", "TSLA"]
    ]
    out = size_orders(cands, regime="bull")
    total = sum(o.weight for o in out)
    assert total <= PER_SECTOR_CAP + 1e-9


def test_drawdown_halt_triggers():
    assert drawdown_halt([1.0, 1.05, 1.10, 0.99]) is True
    assert drawdown_halt([1.0, 1.02, 1.04, 1.00]) is False
