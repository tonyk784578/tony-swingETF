"""판정 절차 동결본 검증 — 실제 판정은 2.3년 뒤라, 합성 장부로 지금 검증한다.

이 테스트가 깨지면 판정 절차가 바뀐 것 — 버그 수정이 아닌 한 되돌릴 것.
"""

import numpy as np
import pandas as pd
import pytest

from src import judge
from src.judge import adjust_costs, first_n_by_completion, one_sided_t


def _ledger(n, mean, sd, name="KODEX200", strategy="breakout", seed=0, start=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-09-01", periods=n + start + 5)
    return pd.DataFrame({
        "entry_date": days[start:start + n],
        "exit_date": days[start + 1:start + n + 1],
        "name": name, "strategy": strategy, "hold": 1,
        "net_ret": rng.normal(mean, sd, n), "preview": "ok",
    })


def test_first_n_by_completion_is_deterministic_and_exit_ordered():
    df = _ledger(10, 0.01, 0.02)
    shuffled = df.sample(frac=1, random_state=1)
    top = first_n_by_completion(shuffled, 5)
    assert list(top["exit_date"]) == sorted(top["exit_date"])
    assert top["exit_date"].max() <= df["exit_date"].sort_values().iloc[4]
    # 실행 시점에 무관: 나중 트레이드가 더 쌓여도 처음 5건은 동일
    more = pd.concat([df, _ledger(5, 0.05, 0.01, start=10)])
    assert first_n_by_completion(more, 5)["net_ret"].tolist() == top["net_ret"].tolist()


def test_adjust_costs_identity_when_judgment_equals_flat():
    df = _ledger(5, 0.01, 0.02)
    adj = adjust_costs(df, 0.001, {"KODEX200": 0.001})
    assert np.allclose(adj, df["net_ret"])


def test_adjust_costs_subtracts_extra_cost_and_keeps_flat_for_unmapped():
    df = _ledger(5, 0.01, 0.02)
    adj = adjust_costs(df, 0.001, {"KODEX200": 0.002})
    assert np.allclose(adj, df["net_ret"] - 0.001)
    assert np.allclose(adjust_costs(df, 0.001, {}), df["net_ret"])  # 매핑 없으면 유지


def test_one_sided_t_matches_scipy():
    from scipy import stats
    r = _ledger(30, 0.01, 0.03)["net_ret"]
    expected = stats.ttest_1samp(r, 0.0).statistic
    assert one_sided_t(r) == pytest.approx(float(expected))


def _run(monkeypatch, capsys, ledger):
    """합성 장부로 run_judge 실제 경로 실행."""
    monkeypatch.setattr("src.etf_paper.load_etf_ledger", lambda: ledger)
    monkeypatch.setattr(judge, "_load_judgment_costs",
                        lambda cfg: {"KODEX200": 0.001})
    judge.run_judge()
    return capsys.readouterr().out


def test_family_verdict_pass_on_synthetic_edge(monkeypatch, capsys):
    # 인샘플급 엣지 (평균 1.3%, sd 5.5%) 120건 → 통과가 기대되는 시나리오
    led = _ledger(120, 0.013, 0.055, seed=42)
    out = _run(monkeypatch, capsys, led)
    assert "계열 판정: 통과" in out
    # 후보별: KODEX200 breakout 120건 중 처음 20건 부호 규칙 적용됨
    assert "처음 20건 평균" in out


def test_family_verdict_reject_on_null_edge(monkeypatch, capsys):
    # 엣지 없음 (평균 0) → 기각이 기대되는 시나리오
    led = _ledger(120, 0.0, 0.055, seed=7)
    out = _run(monkeypatch, capsys, led)
    assert "계열 판정: 기각" in out


def test_not_yet_below_sample(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, _ledger(119, 0.013, 0.055))
    assert "판정 시점 아님 — 완결 119/120건" in out


def test_rotation2_verdict_runs_at_threshold(monkeypatch, capsys):
    led = pd.concat([_ledger(120, 0.013, 0.055),
                     _ledger(20, 0.057, 0.202, name="KODEX_Semicon",
                             strategy="rotation2", seed=3)])
    out = _run(monkeypatch, capsys, led)
    assert ("통과 — 실거래 편입 가능" in out) or ("기각 — 로테이션 프로그램 종료" in out)
