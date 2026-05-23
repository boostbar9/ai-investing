from packages.data.sources import SOURCES


def test_sources_match_spec():
    names = {s.name for s in SOURCES}
    assert {"Polygon.io", "Alpha Vantage", "Finnhub", "SEC EDGAR", "FRED", "Reddit / X"} == names


def test_polygon_hard_cap_present():
    polygon = next(s for s in SOURCES if s.name == "Polygon.io")
    assert "100/sec" in polygon.hard_cap
