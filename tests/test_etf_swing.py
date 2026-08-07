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
    idx = pd.bdate_range("2024-01-01", periods=90)
    close = pd.Series(100.0, index=idx) + pd.Series(range(90), index=idx, dtype=float)
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                       "Close": close, "Volume": 100})
    us = pd.Series(0.0, index=idx)
    for strat in ["breakout", "macross", "rsi2", "trend_ride", "pullback"]:
        raw = raw_entry_signal(df, strat, us)
        entry, _, _ = build_flags(df, strat, us)
        pd.testing.assert_series_equal(entry.fillna(False).astype(bool),
                                       raw.shift(1).fillna(False).astype(bool),
                                       check_names=False)


def test_trend_ride_blocks_bear_market_rally():
    """20일 신고가라도 MA20<=MA60(정배열 아님)이면 진입 금지 — 상승장 필터의 핵심."""
    from src.etf_swing import raw_entry_signal
    idx = pd.bdate_range("2024-01-01", periods=75)
    vals = [200.0 - 2 * i for i in range(60)] + [82.0 + 3 * (i + 1) for i in range(15)]
    close = pd.Series(vals, index=idx)
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                       "Close": close, "Volume": 100})
    us = pd.Series(0.0, index=idx)
    assert raw_entry_signal(df, "breakout", us).fillna(False).any()   # 반등이 신고가는 만든다
    assert not raw_entry_signal(df, "trend_ride", us).fillna(False).any()  # 그러나 정배열 아님


def test_pullback_blocked_in_downtrend():
    """하락장 과매도는 매수 금지 — 무필터 rsi2와의 차별점 (MA200 아래면 차단)."""
    from src.etf_swing import raw_entry_signal
    idx = pd.bdate_range("2023-01-01", periods=260)
    close = pd.Series(400.0, index=idx) - pd.Series(range(260), index=idx, dtype=float)
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                       "Close": close, "Volume": 100})
    us = pd.Series(0.0, index=idx)
    assert raw_entry_signal(df, "rsi2", us).fillna(False).any()      # 무필터는 계속 산다
    assert not raw_entry_signal(df, "pullback", us).fillna(False).any()  # 눌림목은 차단


def test_pullback_fires_on_dip_in_uptrend():
    """상승추세(MA200 위) 중 급락 눌림에서는 진입 신호."""
    from src.etf_swing import raw_entry_signal
    idx = pd.bdate_range("2023-01-01", periods=260)
    vals = [100.0 + 0.5 * i for i in range(257)] + [216.0, 212.0, 208.0]  # 3일 급락
    close = pd.Series(vals, index=idx)
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                       "Close": close, "Volume": 100})
    us = pd.Series(0.0, index=idx)
    raw = raw_entry_signal(df, "pullback", us)
    assert raw.iloc[-1]             # 과매도 + MA200 위 → 진입
    ma200 = close.rolling(200).mean().iloc[-1]
    assert close.iloc[-1] > ma200   # 전제 확인: 급락에도 장기 추세는 위


def test_trend_ride_uptrend_signals_and_holds():
    """지속 상승장: 신고가+정배열 → 진입 신호, 추세 유지 중 청산 플래그 없음, 캡 60일."""
    from src.etf_swing import build_flags, raw_entry_signal
    idx = pd.bdate_range("2024-01-01", periods=90)
    close = pd.Series(100.0, index=idx) + pd.Series(range(90), index=idx, dtype=float)
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                       "Close": close, "Volume": 100})
    us = pd.Series(0.0, index=idx)
    assert raw_entry_signal(df, "trend_ride", us).iloc[-1]
    _, exit_, max_hold = build_flags(df, "trend_ride", us)
    assert not exit_.iloc[-1]      # 종가 > MA20 인 동안은 청산 없음
    assert max_hold == 60          # 사전 등록 값 — config 변조 시 테스트가 잡는다
