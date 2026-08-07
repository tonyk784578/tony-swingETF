"""tom(월말월초 효과) 신호 테스트 — 사전 등록 창(월초 1~3거래일)이 정확한지."""

import pandas as pd

from src.etf_swing import build_flags, raw_entry_signal, simulate


def _df(dates):
    idx = pd.DatetimeIndex(dates)
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                         "Close": 100.0, "Volume": 1000}, index=idx)


def test_first_trading_day_detected_including_late_month_start():
    # 5월: 연휴로 첫 거래일이 4일인 경우 — 달력 1일이 아니라 '첫 거래일'이어야 함
    df = _df(["2026-04-28", "2026-04-29", "2026-04-30",
              "2026-05-04", "2026-05-06", "2026-05-07"])
    flag = raw_entry_signal(df, "tom", pd.Series(dtype=float))
    assert flag.tolist() == [False, False, False, True, False, False]


def test_year_boundary():
    df = _df(["2025-12-29", "2025-12-30", "2026-01-02", "2026-01-05"])
    flag = raw_entry_signal(df, "tom", pd.Series(dtype=float))
    assert flag.tolist() == [False, False, True, False]


def test_first_row_never_flags():
    # 데이터 시작일이 월초여도 직전 봉이 없으면 판별 불가 — False
    df = _df(["2026-03-02", "2026-03-03"])
    assert not raw_entry_signal(df, "tom", pd.Series(dtype=float)).iloc[0]


def test_holding_window_is_first_three_trading_days():
    # 진입 = 월초 1일 시가, 청산 = 3거래일 종가 (T+1~T+3) — hold_days=2 매핑 검증
    df = _df(["2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03",
              "2026-07-06", "2026-07-07"])
    entry, exit_, max_hold = build_flags(df, "tom", pd.Series(dtype=float))
    trades = simulate(df, entry, exit_, max_hold, cost=0.0)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_date"] == pd.Timestamp("2026-07-01")   # 월초 1일 시가 진입
    assert t["exit_date"] == pd.Timestamp("2026-07-03")    # 3거래일 종가 청산
    assert t["hold"] == 2
