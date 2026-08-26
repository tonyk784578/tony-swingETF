"""오버나이트 엔진 테스트 — 등록된 규칙(종가 진입·익일 시가 청산·부호 컷)이 정확한지."""

import pandas as pd
import pytest

from src.etf_swing import simulate_overnight


def _df(rows):
    """rows: [(open, high, low, close), ...]"""
    idx = pd.bdate_range("2026-03-02", periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


def test_up_day_enters_close_exits_next_open():
    # 장중 +5% 양봉 → 종가 105 매수, 익일 시가 107 청산
    df = _df([(100, 106, 99, 105),
              (107, 108, 106, 107)])
    t = simulate_overnight(df, entry_above=0.0, cost=0.0)
    assert len(t) == 1
    assert t.iloc[0]["net_ret"] == pytest.approx(107 / 105 - 1)
    assert t.iloc[0]["hold"] == 1


def test_down_or_flat_day_no_trade():
    # 음봉과 보합(종가=시가)은 부호 컷(> 0) 미충족 — 진입 없음
    df = _df([(100, 101, 95, 97),     # 음봉
              (100, 102, 99, 100),    # 보합 (strict > 라 미진입)
              (101, 102, 100, 101)])
    assert simulate_overnight(df, entry_above=0.0, cost=0.0).empty


def test_cost_subtracted():
    df = _df([(100, 106, 99, 105),
              (105, 106, 104, 105)])
    t = simulate_overnight(df, entry_above=0.0, cost=0.001)
    assert t.iloc[0]["net_ret"] == pytest.approx(105 / 105 - 1 - 0.001)


def test_last_bar_signal_is_open_position_not_trade():
    # 마지막 봉 양봉 — 다음 시가가 없으므로 완결 미계상, 미청산으로 보고
    df = _df([(100, 101, 99, 100),
              (100, 106, 99, 105)])
    trades, open_pos = simulate_overnight(df, entry_above=0.0, cost=0.0,
                                          return_open=True)
    assert trades.empty
    assert open_pos is not None and open_pos["entry_price"] == pytest.approx(105)


def test_no_lookahead_future_bar_does_not_change_past_trades():
    # 미래 봉을 바꿔도 그 앞의 완결 트레이드는 불변이어야 한다
    base = [(100, 106, 99, 105), (107, 108, 106, 107), (100, 101, 99, 100)]
    alt = [base[0], base[1], (200, 201, 199, 200)]
    t1 = simulate_overnight(_df(base), entry_above=0.0, cost=0.0)
    t2 = simulate_overnight(_df(alt), entry_above=0.0, cost=0.0)
    pd.testing.assert_frame_equal(t1.iloc[:1], t2.iloc[:1])


def test_consecutive_up_days_are_independent_trades():
    df = _df([(100, 106, 99, 105),
              (106, 112, 105, 111),
              (113, 114, 112, 113)])
    t = simulate_overnight(df, entry_above=0.0, cost=0.0)
    assert len(t) == 2
    assert t.iloc[0]["net_ret"] == pytest.approx(106 / 105 - 1)
    assert t.iloc[1]["net_ret"] == pytest.approx(113 / 111 - 1)
