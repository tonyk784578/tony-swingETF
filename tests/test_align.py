"""날짜 정렬(룩어헤드 방지) 핵심 로직 테스트 — 합성 데이터 사용."""

import numpy as np
import pandas as pd
import pytest

from src.align import build_master, map_us_to_kr, us_returns


def _ohlcv(dates, closes):
    closes = pd.Series(closes, index=dates, dtype=float)
    return pd.DataFrame({
        "Open": closes * 0.99, "High": closes * 1.01, "Low": closes * 0.98,
        "Close": closes, "Volume": 1000,
    })


@pytest.fixture
def synthetic():
    """KR/US 같은 영업일 캘린더 40일 + 알려진 가격 패턴."""
    dates = pd.bdate_range("2024-01-01", periods=40)
    kr = _ohlcv(dates, np.linspace(100, 120, 40))
    kospi = _ohlcv(dates, np.linspace(2000, 2100, 40))
    us = _ohlcv(dates, np.linspace(50, 70, 40))
    return {"STOCK": kr, "KOSPI": kospi, "SOX": us.copy(),
            "NASDAQ": us.copy(), "NVDA": us.copy()}


def test_us_bar_is_strictly_before_kr_date(synthetic):
    m = build_master(synthetic, "STOCK")
    for col in ["sox_ret", "ixic_ret", "nvda_ret"]:
        assert (m[f"{col}_us_date"] < m.index.to_series()).all()


def test_same_calendar_maps_previous_us_day(synthetic):
    """KR 거래일 D에는 미국 D-1 봉 수익률이 매핑돼야 한다."""
    m = build_master(synthetic, "STOCK")
    sox_ret = us_returns(synthetic["SOX"])
    d = m.index[-1]
    prev_us = sox_ret.index[sox_ret.index < d][-1]
    assert m.loc[d, "sox_ret"] == pytest.approx(sox_ret.loc[prev_us])
    assert m.loc[d, "sox_ret_us_date"] == prev_us


def test_us_holiday_reuses_last_confirmed_bar(synthetic):
    """미국 휴장(봉 삭제) 시 그 이전 마지막 봉이 재사용돼야 한다."""
    dates = synthetic["SOX"].index
    holiday = dates[25]
    for k in ["SOX", "NASDAQ", "NVDA"]:
        synthetic[k] = synthetic[k].drop(holiday)
    m = build_master(synthetic, "STOCK")
    kr_day_after = dates[26]
    assert m.loc[kr_day_after, "sox_ret_us_date"] == dates[24]


def test_monday_maps_friday(synthetic):
    m = build_master(synthetic, "STOCK")
    mondays = m.index[m.index.dayofweek == 0]
    d = mondays[-1]
    assert m.loc[d, "sox_ret_us_date"] == d - pd.Timedelta(days=3)


def test_kospi_lagging_one_day_keeps_latest_row(synthetic):
    """KOSPI 캐시가 하루 늦어도 종목 최신일이 마스터에서 탈락하면 안 된다."""
    synthetic["KOSPI"] = synthetic["KOSPI"].iloc[:-1]
    m = build_master(synthetic, "STOCK")
    last = synthetic["STOCK"].index[-1]
    assert last in m.index
    assert not np.isnan(m.loc[last, "kospi_prev_ret"])


def test_derived_columns_have_no_lookahead(synthetic):
    """gap/day_ret/ma20 정의 확인: D 종가는 판단 컬럼에 못 들어간다."""
    m = build_master(synthetic, "STOCK")
    stock = synthetic["STOCK"]
    d = m.index[-1]
    prev = stock.index[stock.index.get_loc(d) - 1]
    assert m.loc[d, "gap"] == pytest.approx(stock.loc[d, "Open"] / stock.loc[prev, "Close"] - 1)
    assert m.loc[d, "day_ret"] == pytest.approx(stock.loc[d, "Close"] / stock.loc[d, "Open"] - 1)
    ma20_prev = stock["Close"].rolling(20).mean().loc[prev]
    assert m.loc[d, "ma20_prev"] == pytest.approx(ma20_prev)


def test_map_us_to_kr_dtype_mismatch():
    """parquet 왕복으로 날짜 해상도가 달라도 merge가 동작해야 한다."""
    us = pd.Series([0.01, 0.02],
                   index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]).astype("datetime64[ms]"))
    kr = pd.DatetimeIndex(["2024-01-04"]).astype("datetime64[us]")
    out = map_us_to_kr(us, kr, "r")
    assert out.iloc[0]["r"] == pytest.approx(0.02)
