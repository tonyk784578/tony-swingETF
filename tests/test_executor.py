"""Phase D 실행기 결정 로직 검증 — 주문 없이 순수 함수만 (KIS 미접속).

15:20 창의 진입/청산 판정(_close_action)과 exec_plan 라이프사이클
(mock_positions)이 대상. 시세·주문 경로는 모의 서버 실측이 담당한다.
"""

import pandas as pd

from src.etf_swing import simulate, simulate_overnight, simulate_volbreak
from src.executor import _close_action, mock_positions

TODAY = pd.Timestamp("2026-09-03")


def _frame(rows):
    idx = pd.bdate_range(end=TODAY, periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def _run_volbreak(df):
    return simulate_volbreak(df, k=0.5, cost=0.001, return_open=True)


def test_volbreak_trigger_hit_buys_at_close():
    # 전일 변동폭 10 → 트리거 = 시가 104 + 5 = 109. 당일 고가 110 이 도달
    df = _frame([[100, 110, 100, 105], [104, 110, 100, 106]])
    _, before = _run_volbreak(df.iloc[:-1])
    trades, after = _run_volbreak(df)
    assert _close_action("volbreak", before, trades, after, TODAY) == "buy"
    assert after["entry_price"] == 109   # max(시가, 트리거) — 동결 엔진 체결 모형


def test_volbreak_no_trigger_no_action():
    df = _frame([[100, 110, 100, 105], [104, 106, 100, 101]])  # 고가 106 < 트리거 109
    _, before = _run_volbreak(df.iloc[:-1])
    trades, after = _run_volbreak(df)
    assert _close_action("volbreak", before, trades, after, TODAY) is None


def test_overnight_positive_candle_buys():
    df = _frame([[100, 101, 99, 100], [100, 102, 99, 101]])   # 당일 양봉
    _, before = simulate_overnight(df.iloc[:-1], 0.0, 0.001, return_open=True)
    trades, after = simulate_overnight(df, 0.0, 0.001, return_open=True)
    assert _close_action("overnight", before, trades, after, TODAY) == "buy"


def test_overnight_negative_candle_no_action():
    df = _frame([[100, 101, 99, 100], [100, 101, 98, 99]])    # 당일 음봉
    _, before = simulate_overnight(df.iloc[:-1], 0.0, 0.001, return_open=True)
    trades, after = simulate_overnight(df, 0.0, 0.001, return_open=True)
    assert _close_action("overnight", before, trades, after, TODAY) is None


def _swing(df, max_hold):
    entry = pd.Series(False, index=df.index)
    entry.iloc[1] = True   # 둘째 봉 시가 진입
    exit_ = pd.Series(False, index=df.index)
    return simulate(df, entry, exit_, max_hold, 0.001, return_open=True)


def test_swing_max_hold_reached_sells_at_close():
    rows = [[100, 101, 99, 100]] * 5
    df = _frame(rows)
    _, before = _swing(df.iloc[:-1], max_hold=3)
    trades, after = _swing(df, max_hold=3)
    assert before is not None and after is None
    assert _close_action("tom", before, trades, after, TODAY) == "sell"


def test_swing_still_holding_no_action():
    rows = [[100, 101, 99, 100]] * 4
    df = _frame(rows)
    _, before = _swing(df.iloc[:-1], max_hold=10)
    trades, after = _swing(df, max_hold=10)
    assert before is not None and after is not None
    assert _close_action("trend_ride", before, trades, after, TODAY) is None


def test_one_day_sleeve_never_sells_in_close_window():
    # 어제 진입한 volbreak 포지션 — 잠정 봉에서 오늘 시가 청산 트레이드가 생겨도
    # 종가 창의 매도 대상이 아니다 (익일 시가 매도는 아침 창 담당)
    df = _frame([[100, 110, 100, 105], [104, 106, 100, 101]])
    _, before = _run_volbreak(df.iloc[:-1])
    trades, after = _run_volbreak(df)
    assert _close_action("volbreak", before, trades, after, TODAY) != "sell"


def _log(rows):
    cols = ["date", "generated_at", "mode", "action", "code", "name", "strategy",
            "qty", "note", "order_no"]
    return pd.DataFrame(rows, columns=cols)


def test_mock_positions_lifecycle():
    log = _log([
        ["2026-09-01", "t", "live_mock", "buy_close", "091160", "Semicon",
         "volbreak", 11, "", "0001"],
        ["2026-09-01", "t", "live_mock", "buy_close", "091160", "Semicon",
         "overnight", 11, "", "0002"],
        ["2026-09-02", "t", "live_mock", "sell_open", "091160", "Semicon",
         "volbreak", 11, "", "0003"],
        ["2026-09-02", "t", "live_mock", "buy_open", "132030", "Gold",
         "tom", 54, "", "0004"],
        # 제출 실패(주문번호 없음)와 dry_run 은 보유로 잡히면 안 된다
        ["2026-09-02", "t", "live_mock", "buy_close", "122630", "Lev",
         "volbreak", 9, "", ""],
        ["2026-09-02", "t", "dry_run", "buy_open", "069500", "K200",
         "breakout", 30, "", ""],
    ])
    pos = mock_positions(log)
    assert ("091160", "volbreak") not in pos          # 매도 완료
    assert pos[("091160", "overnight")]["qty"] == 11  # 미청산
    assert pos[("132030", "tom")]["qty"] == 54
    assert ("122630", "volbreak") not in pos
    assert ("069500", "breakout") not in pos
