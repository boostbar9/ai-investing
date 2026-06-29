"""Unit tests for read-only fill-provenance capture at order-record time.

``resolve_fill_provenance`` decides ``fill_price`` / ``filled_qty`` /
``fill_source`` for an order using ONLY values already produced by the
(read-only) submit path: the broker's ``last_fill_meta`` and the last-known
mark. It never places an order and never fabricates a price.
"""
from __future__ import annotations

from tools import paper_trade as pt


def test_broker_fill_takes_priority():
    meta = {"fill_price": 101.25, "filled_qty": 9.5}
    out = pt.resolve_fill_provenance(meta, last_price=100.0, requested_qty=10)
    assert out["fill_source"] == "broker_fill"
    assert out["fill_price"] == 101.25
    assert out["filled_qty"] == 9.5


def test_broker_fill_falls_back_to_requested_qty():
    meta = {"fill_price": 50.0}  # no filled_qty reported
    out = pt.resolve_fill_provenance(meta, last_price=49.0, requested_qty=4)
    assert out["fill_source"] == "broker_fill"
    assert out["fill_price"] == 50.0
    assert out["filled_qty"] == 4.0


def test_mark_estimate_when_no_broker_fill():
    out = pt.resolve_fill_provenance(None, last_price=200.0, requested_qty=3)
    assert out["fill_source"] == "mark_estimate"
    assert out["fill_price"] == 200.0
    assert out["filled_qty"] == 3.0


def test_mark_estimate_when_meta_has_no_usable_price():
    meta = {"pricing_source": "rh_quote"}  # no fill_price
    out = pt.resolve_fill_provenance(meta, last_price=12.5, requested_qty=8)
    assert out["fill_source"] == "mark_estimate"
    assert out["fill_price"] == 12.5


def test_unknown_when_nothing_available():
    out = pt.resolve_fill_provenance(None, last_price=None, requested_qty=10)
    assert out["fill_source"] == "unknown"
    assert out["fill_price"] is None
    assert out["filled_qty"] is None


def test_never_fabricates_from_nonpositive_or_garbage_prices():
    # Zero / negative / non-numeric prices are rejected, never used as a fill.
    assert pt.resolve_fill_provenance({"fill_price": 0.0}, None, 10)["fill_source"] == "unknown"
    assert pt.resolve_fill_provenance({"fill_price": -5.0}, None, 10)["fill_source"] == "unknown"
    assert pt.resolve_fill_provenance({"fill_price": "x"}, None, 10)["fill_source"] == "unknown"
    # A bool must never be mistaken for a price (bool is an int subclass).
    assert pt.resolve_fill_provenance({"fill_price": True}, None, 10)["fill_source"] == "unknown"
