"""횡단면 프로그램 시점 규율 검증 — 네트워크 없이 합성 데이터로.

핵심: 피처가 신호일 이후 정보를 보지 않는다(룩어헤드), 포트폴리오 산수가
등록 §6 그대로다. 깨지면 등록 위반이므로 버그 수정 외 되돌릴 것.
"""

import numpy as np
import pandas as pd

from src.xsection import FEATURES, rebalance_days, simulate_portfolio, symbol_features


def _symbol(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, n)), index=idx)
    return pd.DataFrame({"Open": c * 0.99, "High": c * 1.01, "Low": c * 0.98,
                         "Close": c, "Volume": 10000.0}, index=idx)


def test_features_no_lookahead():
    df = _symbol()
    s_day = df.index[260]
    base = symbol_features(df).loc[s_day, FEATURES]
    tampered = df.copy()
    tampered.iloc[261:, :] *= 3.0   # 미래를 바꿔도
    after = symbol_features(tampered).loc[s_day, FEATURES]
    pd.testing.assert_series_equal(base, after)   # 신호일 피처는 불변


def test_feature_definitions_match_prereg():
    df = _symbol()
    f = symbol_features(df)
    i = 270
    c = df["Close"]
    assert np.isclose(f["mom_12_1"].iloc[i], c.iloc[i - 21] / c.iloc[i - 252] - 1)
    assert np.isclose(f["rev_21"].iloc[i], c.iloc[i] / c.iloc[i - 21] - 1)
    assert np.isclose(f["high_52w"].iloc[i],
                      c.iloc[i] / df["High"].iloc[i - 251:i + 1].max())


def test_rebalance_days_weekly_first_trading_day():
    cal = pd.bdate_range("2024-01-01", periods=30)
    days = rebalance_days(cal)
    assert all(d.weekday() == 0 for d in days)      # 월요일(휴장 없다면)
    assert days[0] > cal[0]                          # 첫 주(신호일 없음) 제외


def test_portfolio_arithmetic_and_costs():
    cal = pd.bdate_range("2024-01-01", periods=15)
    ks = pd.DataFrame({"Open": 100.0, "Close": 100.0}, index=cal)  # 헤지 0%
    codes = ["A", "B"]
    opens = pd.DataFrame(100.0, index=cal, columns=codes)
    opens.loc[cal[10], "A"] = 110.0   # 2주차 진입가 기준 +10% 청산 확인용
    # 마지막 주는 청산일이 없어 미완성 처리되므로 3주를 넣어 앞 2주를 완결시킨다
    scored = pd.DataFrame({
        "entry_day": [cal[5], cal[5], cal[10], cal[10], cal[13]],
        "code": ["A", "B", "A", "B", "A"],
        "score": [2.0, 1.0, 2.0, 1.0, 1.0],
    })
    port = simulate_portfolio(scored, opens, opens, ks, top_n=1,
                              cost_rt=0.003, hedge_cost=0.0)
    # 1주차: A 진입 100 → 다음 리밸런스일 110 = +10%, 신규 1/1 비용 0.3%
    assert np.isclose(port.iloc[0]["net_ret"], 0.10 - 0.003)
    # 2주차: A 유지 (신규 0) — 비용 0
    assert port.iloc[1]["turnover_new"] == 0


def test_exit_price_fallback_for_dead_symbol():
    # 상폐 종목: 청산일에 시가 없음 → exitpx(마지막 종가 ffill 합성)로 처분
    cal = pd.bdate_range("2024-01-01", periods=15)
    ks = pd.DataFrame({"Open": 100.0, "Close": 100.0}, index=cal)
    opens = pd.DataFrame(100.0, index=cal, columns=["A"])
    opens.loc[cal[8]:, "A"] = np.nan          # 8일째부터 거래 없음
    exitpx = opens.where(opens.notna(), 80.0)  # 폴백 처분가 80
    scored = pd.DataFrame({"entry_day": [cal[5], cal[12]],
                           "code": ["A", "A"], "score": [1.0, 1.0]})
    port = simulate_portfolio(scored, opens, exitpx, ks, top_n=1,
                              cost_rt=0.0, hedge_cost=0.0)
    assert np.isclose(port.iloc[0]["net_ret"], 80.0 / 100.0 - 1)   # -20% 실현
