"""아침 스냅샷 — 같은 날 중복 기록 방지(최초 시점 보존) 테스트."""

import pandas as pd

from src import snapshot


def test_same_day_second_run_preserves_first_row(monkeypatch, tmp_path):
    path = tmp_path / "market_snapshots.csv"
    monkeypatch.setattr(snapshot, "_snapshot_path", lambda: path)
    monkeypatch.setattr(snapshot, "_fetch_quote", lambda s: 100.0)
    monkeypatch.setattr(snapshot, "load_config",
                        lambda: {"snapshot": {"tickers": {"NQ=F": "nasdaq_fut"}}})

    snapshot.record_market_snapshot()
    first = pd.read_csv(path)
    assert len(first) == 1 and first["nasdaq_fut"].iloc[0] == 100.0

    monkeypatch.setattr(snapshot, "_fetch_quote", lambda s: 999.0)  # 값이 바뀌어도
    snapshot.record_market_snapshot()
    again = pd.read_csv(path)
    assert len(again) == 1                                # 행 추가 없음
    assert again["nasdaq_fut"].iloc[0] == 100.0           # 최초 기록 보존


def test_failed_ticker_recorded_as_nan_not_dropped(monkeypatch, tmp_path):
    path = tmp_path / "market_snapshots.csv"
    monkeypatch.setattr(snapshot, "_snapshot_path", lambda: path)
    monkeypatch.setattr(snapshot, "load_config",
                        lambda: {"snapshot": {"tickers": {"A": "a", "B": "b"}}})

    def flaky(sym):
        if sym == "B":
            raise RuntimeError("source down")
        return 42.0

    monkeypatch.setattr(snapshot, "_fetch_quote", flaky)
    snapshot.record_market_snapshot()
    row = pd.read_csv(path).iloc[0]
    assert row["a"] == 42.0 and pd.isna(row["b"])   # 한 소스 장애가 행을 못 막음
