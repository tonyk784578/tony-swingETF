"""ewy_* (교차상장 신호) — 등록된 신호 정의와 룩어헤드 안전성 테스트."""

import pandas as pd

from src.align import map_us_to_kr
from src.etf_swing import build_flags, raw_entry_signal, simulate


def _kr_df(n=6):
    idx = pd.bdate_range("2026-03-02", periods=n)
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                         "Close": 100.0, "Volume": 1000}, index=idx)


def test_threshold_bands_from_config():
    df = _kr_df()
    mapped = pd.Series([0.005, 0.012, 0.025, -0.01, 0.02, 0.0], index=df.index)
    up1 = raw_entry_signal(df, "ewy_up1", mapped)   # >= +1%
    up2 = raw_entry_signal(df, "ewy_up2", mapped)   # >= +2%
    assert up1.tolist() == [False, True, True, False, True, False]
    assert up2.tolist() == [False, False, True, False, True, False]


def test_mapping_is_strictly_before_kr_date():
    # 미국 금요일 급등이 한국 다음 거래일(월요일) 신호가 되는지 — 기존 정렬 인프라
    kr = pd.DatetimeIndex(["2026-03-06", "2026-03-09"])           # 금, 월
    us_ret = pd.Series([0.03], index=pd.DatetimeIndex(["2026-03-06"]))  # 미국 금요일 +3%
    mapped = map_us_to_kr(us_ret, kr, "ewy")["ewy"]
    assert pd.isna(mapped.loc["2026-03-06"])   # 같은 날 미국 봉은 못 씀 (< D 엄격)
    assert mapped.loc["2026-03-09"] == 0.03    # 월요일이 금요일 봉을 사용


def test_hold_one_day_exits_next_close():
    df = _kr_df()
    mapped = pd.Series([0.0, 0.015, 0.0, 0.0, 0.0, 0.0], index=df.index)
    entry, exit_, max_hold = build_flags(df, "ewy_up1", mapped)
    trades = simulate(df, entry, exit_, max_hold, cost=0.0)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_date"] == df.index[1]      # 신호일 시가 진입
    assert t["exit_date"] == df.index[2]       # 익일 종가 청산 (보유 1일)
    assert t["hold"] == 1
