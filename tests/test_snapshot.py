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


def test_premarket_flag_uses_configured_deadline(monkeypatch):
    """개장 후 기록은 '아침 정보'가 아니다 — 2026-08-10 22:35 행 사례."""
    monkeypatch.setattr(snapshot, "load_config",
                        lambda: {"snapshot": {"premarket_deadline": "09:00"}})
    assert snapshot.is_premarket("2026-08-14 08:40:28")
    assert not snapshot.is_premarket("2026-08-10 22:35:53")
    assert not snapshot.is_premarket("2026-08-10 09:00:00")   # 경계는 개장 후
    assert not snapshot.is_premarket("not-a-time")            # 파싱 실패는 보수적으로


def test_load_snapshots_can_exclude_post_open_rows(monkeypatch, tmp_path):
    path = tmp_path / "market_snapshots.csv"
    monkeypatch.setattr(snapshot, "_snapshot_path", lambda: path)
    monkeypatch.setattr(snapshot, "load_config",
                        lambda: {"snapshot": {"premarket_deadline": "09:00"}})
    pd.DataFrame({
        "date": ["2026-08-10", "2026-08-11"],
        "taken_at": ["2026-08-10 22:35:53", "2026-08-11 08:45:13"],
        "nasdaq_fut": [1.0, 2.0],
    }).to_csv(path, index=False)

    assert len(snapshot.load_snapshots()) == 2                    # 기록은 보존
    kept = snapshot.load_snapshots(premarket_only=True)
    assert len(kept) == 1 and kept["nasdaq_fut"].iloc[0] == 2.0   # 오염 행만 제외


def test_load_snapshots_returns_empty_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "_snapshot_path", lambda: tmp_path / "none.csv")
    assert snapshot.load_snapshots().empty
