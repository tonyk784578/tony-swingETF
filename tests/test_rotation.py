"""로테이션 엔진 테스트 — 선택 규칙, 월말 스케줄, 체결·비용 시점."""

import pandas as pd
import pytest

from src.rotation import month_end_days, select_targets, simulate_rotation


def test_month_end_days():
    idx = pd.bdate_range("2024-01-15", "2024-03-10")
    ends = month_end_days(idx)
    assert ends == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"),
                    pd.Timestamp("2024-03-08")]


def test_select_targets_abs_momentum_filter():
    mom = pd.Series({"A": 0.30, "B": 0.10, "C": -0.05, "D": float("nan"), "E": 0.02})
    # 음수(C)·NaN(D) 제외 후 상위 3 → A, B, E
    assert select_targets(mom, 3) == {"A", "B", "E"}
    mom_all_neg = pd.Series({"A": -0.1, "B": -0.2})
    assert select_targets(mom_all_neg, 3) == set()   # 전원 미달 → 현금


def test_simulate_rotation_entry_exit_and_cost():
    """모멘텀 역전 시 다음 리밸런스 시가에 교체, 비용은 청산 시 차감."""
    idx = pd.bdate_range("2024-01-01", periods=280)
    up = pd.Series([100.0 + 0.5 * i for i in range(280)], index=idx)      # 꾸준한 상승
    fade = pd.Series([200.0 - 0.5 * i for i in range(280)], index=idx)    # 꾸준한 하락
    closes = pd.DataFrame({"UP": up, "FADE": fade})
    opens = closes.copy()
    ep, open_pos, daily = simulate_rotation(opens, closes, lookback=252, top_k=1,
                                            cost=0.001)
    # 252일 이후 첫 월말 신호에서 UP만 자격(FADE 모멘텀 음수) → UP 단독 보유 지속
    assert ep.empty                        # 교체 없음 — 완결 에피소드 0
    assert len(open_pos) == 1 and open_pos[0]["name"] == "UP"
    # 진입일은 첫 유효 월말 신호의 다음 거래일
    first_signal = next(d for d in month_end_days(idx)
                        if idx.get_loc(d) >= 252)
    assert open_pos[0]["entry_date"] == idx[idx.get_loc(first_signal) + 1]


def test_simulate_rotation_swap_charges_cost_once():
    """A→B 교체: A 에피소드 순수익 = 시가/시가 - 1 - cost."""
    idx = pd.bdate_range("2024-01-01", periods=300)
    a_vals = [100.0 + 0.5 * i for i in range(260)]               # 고모멘텀 (+110%대)
    a_vals += [a_vals[-1] - 5.0 * (i + 1) for i in range(40)]    # 이후 급락 → 모멘텀 음전
    a = pd.Series(a_vals, index=idx)
    b = pd.Series([50.0 + 0.01 * i for i in range(300)], index=idx)  # 저모멘텀 (+5%대)
    closes = pd.DataFrame({"A": a, "B": b})
    opens = closes.copy()
    ep, open_pos, _ = simulate_rotation(opens, closes, lookback=252, top_k=1, cost=0.001)
    assert len(ep) >= 1
    t = ep.iloc[0]
    assert t["name"] == "A"
    expected = opens.at[t["exit_date"], "A"] / opens.at[t["entry_date"], "A"] - 1 - 0.001
    assert t["net_ret"] == pytest.approx(expected)
    assert open_pos and open_pos[0]["name"] == "B"   # 교체 후 B 보유
