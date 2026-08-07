"""ETF 스윙 엔진 테스트 — 진입/청산 시점과 룩어헤드 규칙."""

import pandas as pd
import pytest

from src.etf_swing import rsi, simulate


def _df(n=10, start_price=100.0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(range(n), index=idx, dtype=float) + start_price
    return pd.DataFrame({"Open": close - 0.5, "High": close + 1,
                         "Low": close - 1, "Close": close, "Volume": 100})


def test_entry_at_open_exit_at_close():
    df = _df(10)
    entry = pd.Series(False, index=df.index)
    entry.iloc[2] = True                       # 3번째 날 시가 진입
    exit_ = pd.Series(False, index=df.index)
    exit_.iloc[5] = True                       # 6번째 날 종가 청산
    trades = simulate(df, entry, exit_, max_hold=10, cost=0.001)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_date"] == df.index[2]
    assert t["exit_date"] == df.index[5]
    assert t["hold"] == 3
    expected = df["Close"].iloc[5] / df["Open"].iloc[2] - 1 - 0.001
    assert t["net_ret"] == pytest.approx(expected)


def test_max_hold_forces_exit():
    df = _df(10)
    entry = pd.Series(False, index=df.index)
    entry.iloc[1] = True
    trades = simulate(df, entry, pd.Series(False, index=df.index), max_hold=3, cost=0.0)
    assert len(trades) == 1
    assert trades.iloc[0]["hold"] == 3
    assert trades.iloc[0]["exit_date"] == df.index[4]


def test_no_same_day_exit():
    """진입 당일 청산 신호는 무시 — 최소 1일 보유."""
    df = _df(10)
    entry = pd.Series(False, index=df.index)
    entry.iloc[2] = True
    exit_ = pd.Series(True, index=df.index)    # 항상 청산 신호
    trades = simulate(df, entry, exit_, max_hold=10, cost=0.0)
    assert trades.iloc[0]["hold"] == 1         # 다음날 청산


def test_single_position_no_overlap():
    df = _df(20)
    entry = pd.Series(True, index=df.index)    # 매일 진입 신호
    exit_ = pd.Series(False, index=df.index)
    trades = simulate(df, entry, exit_, max_hold=5, cost=0.0)
    for a, b in zip(trades["exit_date"], trades["entry_date"].iloc[1:], strict=False):
        assert b >= a                           # 청산 후에만 재진입


def test_rsi_bounds():
    df = _df(50)
    r = rsi(df["Close"], 2).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert r.iloc[-1] > 90                      # 연속 상승이면 과매수 근처


def test_trailing_stop_exit():
    """최고 종가 대비 trailing% 이탈 시 그날 종가 청산."""
    idx = pd.bdate_range("2024-01-01", periods=8)
    close = pd.Series([100, 105, 110, 108, 103, 102, 101, 100], index=idx, dtype=float)
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                       "Close": close, "Volume": 100})
    entry = pd.Series(False, index=idx)
    entry.iloc[0] = True
    trades = simulate(df, entry, pd.Series(False, index=idx), max_hold=10,
                      cost=0.0, trailing=0.05)
    t = trades.iloc[0]
    # peak 110 → 110*0.95=104.5 → 첫 이탈 종가는 103 (5번째 봉)
    assert t["exit_date"] == idx[4]
    assert t["net_ret"] == pytest.approx(103 / 100 - 1)


def test_build_flags_consistent_with_raw_signal():
    """백테스트 진입 플래그 = raw 신호의 다음날 — 프리뷰와 정의가 갈라지면 안 된다."""
    from src.etf_swing import build_flags, raw_entry_signal
    idx = pd.bdate_range("2024-01-01", periods=60)
    close = pd.Series(100.0, index=idx) + pd.Series(range(60), index=idx, dtype=float)
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                       "Close": close, "Volume": 100})
    us = pd.Series(0.0, index=idx)
    for strat in ["breakout", "macross", "rsi2"]:
        raw = raw_entry_signal(df, strat, us)
        entry, _, _ = build_flags(df, strat, us)
        pd.testing.assert_series_equal(entry.fillna(False).astype(bool),
                                       raw.shift(1).fillna(False).astype(bool),
                                       check_names=False)
