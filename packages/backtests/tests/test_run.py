from packages.backtests.run import main


def test_run_smoke(capsys):
    rc = main(["--strategy", "trend-following", "--regime", "bull"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "trend-following" in out
    assert "bull" in out
