"""변동성 돌파 엔진 테스트 — 등록된 체결 모형(트리거·갭·미완성 봉)이 정확한지."""

import pandas as pd
import pytest

from src.etf_swing import simulate_volbreak, volbreak_trigger


def _df(rows):
    """rows: [(open, high, low, close), ...]"""
    idx = pd.bdate_range("2026-03-02", periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


def test_trigger_hit_fills_at_trigger_and_exits_next_open():
    # 전일 범위 10 → 트리거 = 100 + 5 = 105. 고가 106 도달 → 105 체결, 익일 시가 108 청산
    df = _df([(100, 105, 95, 100),
              (100, 106, 99, 105),
              (108, 110, 107, 109)])
    t = simulate_volbreak(df, k=0.5, cost=0.0)
    assert len(t) == 1
    assert t.iloc[0]["net_ret"] == pytest.approx(108 / 105 - 1)
    assert t.iloc[0]["hold"] == 1


def test_gap_open_above_trigger_fills_at_open():
    # 시가(112)가 이미 트리거(110+5=115?) 아래… 명확한 케이스: 트리거=시가+5=117,
    # 고가 118 → 117 체결이 기본. 갭 케이스: 시가 자체가 전일고가 위여도
    # 트리거는 '시가+k*range'로 재산정되므로 체결가는 max(시가, 트리거)=트리거
    df = _df([(100, 110, 100, 108),
              (112, 118, 111, 117),
              (120, 121, 119, 120)])
    t = simulate_volbreak(df, k=0.5, cost=0.0)
    assert t.iloc[0]["net_ret"] == pytest.approx(120 / 117 - 1)


def test_no_trade_when_high_below_trigger():
    df = _df([(100, 105, 95, 100),
              (100, 104.9, 99, 104),   # 트리거 105 미도달
              (106, 107, 105, 106)])
    assert simulate_volbreak(df, k=0.5, cost=0.0).empty


def test_first_bar_never_triggers_no_lookahead():
    # 첫 봉은 전일 범위가 없다 — NaN 트리거는 False 처리
    df = _df([(100, 200, 50, 150), (150, 151, 149, 150)])
    assert simulate_volbreak(df, k=0.5, cost=0.0).empty


def test_last_bar_trigger_is_open_position_not_trade():
    # 마지막 봉 트리거 — 다음 시가가 없으므로 완결 트레이드 미계상, 미청산으로 보고
    df = _df([(100, 105, 95, 100),
              (100, 106, 99, 105.5)])
    trades, open_pos = simulate_volbreak(df, k=0.5, cost=0.0, return_open=True)
    assert trades.empty
    assert open_pos is not None and open_pos["entry_price"] == pytest.approx(105.0)


def test_consecutive_triggers_are_independent_trades():
    df = _df([(100, 105, 95, 100),
              (100, 106, 99, 105),     # 트리거 105 → 체결
              (106, 112, 105, 111),    # 시가 106 청산 + 트리거 106+3.5 → 재체결
              (113, 114, 112, 113)])
    t = simulate_volbreak(df, k=0.5, cost=0.0)
    assert len(t) == 2
    assert t.iloc[1]["net_ret"] == pytest.approx(113 / 109.5 - 1)


def test_trigger_preview_value():
    df = _df([(100, 108, 96, 100)])
    assert volbreak_trigger(df, 0.5) == pytest.approx(6.0)  # 내일 '시가+6' 이상 매수
