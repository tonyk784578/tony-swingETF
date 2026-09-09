"""imom 규칙 고정 — 신호 시점(09:30)·진입가(15:00)·청산가(종가) 룩어헤드 방지,
결측 봉 제외, 진짜 직전 거래일 종가, 포워드 장부 멱등."""

import pandas as pd

from src.imom import append_forward, day_table, gate_verdict, simulate_imom


def _daily(closes, start="2026-06-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes},
                        index=idx, dtype=float)


def _bars(day, spec):
    """spec: {"09:25": close, "14:55": close, ...}"""
    idx = pd.to_datetime([f"{day} {t}" for t in spec])
    v = list(spec.values())
    return pd.DataFrame({"Open": v, "High": v, "Low": v, "Close": v}, index=idx, dtype=float)


def test_day_table_uses_prev_trading_day_close_and_named_bars():
    daily = _daily([100.0, 110.0, 120.0])           # 06-01, 06-02, 06-03
    minute = pd.concat([
        _bars("2026-06-02", {"09:00": 99.0, "09:25": 101.0, "14:55": 105.0}),
        _bars("2026-06-03", {"09:00": 99.0, "09:25": 109.0, "14:55": 118.0}),
    ])
    t = day_table(daily, minute)
    assert list(t.index.date) == [pd.Timestamp("2026-06-02").date(),
                                  pd.Timestamp("2026-06-03").date()]
    assert t.loc["2026-06-02", "prev_close"] == 100.0        # 06-01 종가
    assert abs(t.loc["2026-06-02", "r1"] - 0.01) < 1e-12    # 101/100 - 1 (09:25 봉)
    assert abs(t.loc["2026-06-02", "r13"] - (110 / 105 - 1)) < 1e-12   # 종가/14:55


def test_day_table_skips_days_missing_signal_or_entry_bar_and_first_day():
    daily = _daily([100.0, 110.0, 120.0, 130.0])
    minute = pd.concat([
        _bars("2026-06-01", {"09:25": 101.0, "14:55": 105.0}),   # prev_close 없음
        _bars("2026-06-02", {"09:25": 101.0}),                   # 14:55 결측
        _bars("2026-06-03", {"14:55": 118.0}),                   # 09:25 결측
        _bars("2026-06-04", {"09:25": 121.0, "14:55": 125.0}),
    ])
    t = day_table(daily, minute)
    assert len(t) == 1 and t.index[0] == pd.Timestamp("2026-06-04")


def test_simulate_trades_only_positive_r1_with_entry_1500_exit_close():
    daily = _daily([100.0, 110.0, 100.0])
    minute = pd.concat([
        _bars("2026-06-02", {"09:25": 101.0, "14:55": 105.0}),   # r1 +1% → 매수
        _bars("2026-06-03", {"09:25": 100.0, "14:55": 99.0}),    # r1 -9% → 무거래
    ])
    tr = simulate_imom(day_table(daily, minute), cost=0.001)
    assert len(tr) == 1
    assert tr.loc[0, "entry_price"] == 105.0 and tr.loc[0, "exit_price"] == 110.0
    assert abs(tr.loc[0, "net_ret"] - (110 / 105 - 1 - 0.001)) < 1e-12


def test_gate_requires_n_mean_t_and_both_halves():
    good = pd.DataFrame({"net_ret": [0.01] * 20 + [0.012] * 20})
    assert gate_verdict(good, 30, 2.0)["passed"] is True
    bad_half = pd.DataFrame({"net_ret": [0.05] * 20 + [-0.001] * 20})
    assert gate_verdict(bad_half, 30, 2.0)["passed"] is False
    small = pd.DataFrame({"net_ret": [0.01] * 10})
    assert gate_verdict(small, 30, 2.0)["passed"] is False


def test_append_forward_is_idempotent_and_respects_freeze():
    tr = pd.DataFrame({"date": pd.to_datetime(["2026-09-09", "2026-09-10", "2026-09-11"]),
                       "r1": [0.01] * 3, "entry_price": [1.0] * 3, "exit_price": [1.01] * 3,
                       "gross_ret": [0.01] * 3, "cost": [0.001] * 3, "net_ret": [0.009] * 3})
    led = pd.DataFrame(columns=["date", "code", "name", "r1", "entry_price", "exit_price",
                                "gross_ret", "cost", "net_ret"])
    led, k = append_forward(led, tr, "069500", "KODEX200", pd.Timestamp("2026-09-09"))
    assert k == 2 and len(led) == 2                    # freeze 당일 제외 (>)
    led, k = append_forward(led, tr, "069500", "KODEX200", pd.Timestamp("2026-09-09"))
    assert k == 0 and len(led) == 2                    # 멱등
