"""홀드아웃 검증 모듈 — 데이터 기준 복원과 계열 풀 구성.

이 시험의 가치는 전적으로 '규칙을 안 바꿨다'에 달려 있다. 그래서 검증 대상은
성적이 아니라 배관이다: 가격 기준이 인샘플과 같은가, 계열 풀이 상관 구조를
제대로 눌러 주는가, 미지원 전략이 조용히 빈 신호로 통과하지 않는가.
"""

import numpy as np
import pandas as pd
import pytest

from src.holdout import _pool, candidate_trades, one_sided_t, reconstruct_raw


def _yf_frame(close, adj_close, ratio_open=0.99, ratio_high=1.02, ratio_low=0.98):
    """yfinance 형태(OHLC=배당조정, Adj Close=원가격) 합성 프레임."""
    idx = pd.bdate_range("2010-01-04", periods=len(close))
    close = np.array(close, dtype=float)
    return pd.DataFrame({"Open": close * ratio_open, "High": close * ratio_high,
                         "Low": close * ratio_low, "Close": close,
                         "Adj Close": np.array(adj_close, dtype=float),
                         "Volume": 1000}, index=idx)


# --------------------------------------------------------- 가격 기준 복원

def test_reconstruct_returns_raw_close_exactly():
    """복원된 Close 는 정의상 Adj Close(원가격)와 같아야 한다."""
    y = _yf_frame([100, 110, 121], [95, 104.5, 115.0])
    raw = reconstruct_raw(y)
    assert np.allclose(raw["Close"], y["Adj Close"])


def test_reconstruct_preserves_intraday_ratios():
    """조정계수는 날짜별로 OHLC 전체에 동일 — 일중 비율은 어떤 경우에도 불변."""
    y = _yf_frame([100, 110, 121], [95, 104.5, 116.16])     # 셋째 날 계수 변화
    raw = reconstruct_raw(y)
    assert np.allclose(raw["Open"] / raw["Close"], y["Open"] / y["Close"])
    assert np.allclose((raw["High"] - raw["Low"]) / raw["Close"],
                       (y["High"] - y["Low"]) / y["Close"])


def test_reconstruct_returns_price_return_not_total_return():
    """이 복원의 존재 이유 — 배당락일에 총수익과 가격수익이 갈린다.

    계수가 그대로인 구간은 수익률이 같고, 계수가 바뀐 날(배당 반영)만 달라진다.
    인샘플이 FDR 미조정가(가격수익) 기준이므로 홀드아웃도 가격수익이어야 한다 —
    조정된 총수익을 그대로 쓰면 배당만큼 전략에 유리한 편향이 생긴다.
    """
    y = _yf_frame([100, 110, 121], [95, 104.5, 116.16])
    raw, adj = reconstruct_raw(y)["Close"].pct_change(), y["Close"].pct_change()
    assert raw.iloc[1] == pytest.approx(adj.iloc[1])         # 계수 불변 구간 → 동일
    assert raw.iloc[2] != pytest.approx(adj.iloc[2])         # 배당락일 → 갈림


def test_reconstruct_is_identity_when_no_dividends():
    """무배당 종목(예: 레버리지)은 조정계수 1 — 값이 그대로여야 한다."""
    y = _yf_frame([100, 105], [100, 105])
    raw = reconstruct_raw(y)
    assert np.allclose(raw[["Open", "High", "Low", "Close"]],
                       y[["Open", "High", "Low", "Close"]])


def test_reconstruct_drops_nonpositive_and_missing_rows():
    y = _yf_frame([100, 0, 110], [100, 0, 110])
    y.loc[y.index[2], "Adj Close"] = np.nan
    raw = reconstruct_raw(y)
    assert len(raw) == 1                       # 0원 행과 결측 행 제거


# ------------------------------------------------------------- 계열 풀 구성

def _trades(dates, rets):
    return pd.DataFrame({"exit_date": pd.to_datetime(dates), "net_ret": rets})


def test_daily_pool_collapses_same_day_trades_into_one_observation():
    """동시 트리거 상관을 누르는 장치 — 같은 청산일은 평균 1건으로."""
    a = _trades(["2010-01-04", "2010-01-05"], [0.02, 0.04])
    b = _trades(["2010-01-04", "2010-01-05"], [0.04, 0.00])
    pooled = _pool([a, b], "daily")
    assert len(pooled) == 2                                  # 4트레이드 → 2관측
    assert pooled.iloc[0] == pytest.approx(0.03)


def test_trade_pool_keeps_every_trade():
    a = _trades(["2010-01-04", "2010-01-05"], [0.02, 0.04])
    b = _trades(["2010-01-04"], [0.04])
    assert len(_pool([a, b], "trade")) == 3


def test_daily_pool_lowers_t_when_candidates_move_together():
    """상관이 높을수록 트레이드 풀 t가 부풀고 일 단위가 이를 교정한다."""
    days = pd.bdate_range("2010-01-04", periods=60)
    same = np.random.default_rng(0).normal(0.004, 0.01, 60)
    legs = [_trades(days, same) for _ in range(5)]           # 5후보가 완전 동조
    t_trade = one_sided_t(_pool(legs, "trade"))
    t_daily = one_sided_t(_pool(legs, "daily"))
    assert t_trade > t_daily                                 # 나이브 풀이 부풀림
    assert t_daily == pytest.approx(one_sided_t(pd.Series(same)))


def test_pool_handles_empty_input():
    assert _pool([], "trade").empty
    assert _pool([pd.DataFrame(columns=["exit_date", "net_ret"])], "daily").empty


# ---------------------------------------------------------------- 가드

def test_us_dependent_strategies_are_rejected_not_silently_empty():
    """미국 매핑이 필요한 전략은 조용히 빈 신호로 통과하면 안 된다."""
    df = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0]},
                      index=pd.bdate_range("2010-01-04", periods=1))
    for strat in ("us_dip", "ewy_up1"):
        with pytest.raises(ValueError, match="홀드아웃"):
            candidate_trades({"code": "069500", "name": "X", "strategy": strat}, df, 0.001)


def test_one_sided_t_needs_variation():
    assert np.isnan(one_sided_t(pd.Series([0.01])))
    assert np.isnan(one_sided_t(pd.Series([0.01, 0.01])))     # 분산 0
    assert one_sided_t(pd.Series([0.01, 0.02, 0.03])) > 0
