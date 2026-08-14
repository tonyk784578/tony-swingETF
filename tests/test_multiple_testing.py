"""다중검정 보정 모듈 검증 — 합성 데이터로 통과/기각 경로를 모두 태운다.

실제 시험 결과에 의존하지 않는다: 진짜 엣지가 없는 잡음 시험 묶음은 보정을
통과하지 못해야 하고, 진짜 엣지가 하나 심어진 묶음은 통과해야 한다. 이 두
방향이 다 서야 '보정이 작동한다'고 말할 수 있다.
"""

import numpy as np
import pandas as pd
import pytest

from src.multiple_testing import (
    bh_fdr,
    daily_matrix,
    deflated_sharpe,
    effective_trials,
    expected_max_z,
    holm,
    reality_check,
    stationary_bootstrap_idx,
)

BOOT = {"block": 10, "draws": 300, "seed": 3}


def _noise_matrix(n_days=600, k=25, seed=0):
    return np.random.default_rng(seed).normal(0, 0.01, (n_days, k))


# ------------------------------------------------------------ 기대 최대 t

def test_expected_max_z_increases_with_trial_count():
    """시험을 늘릴수록 '엣지 없이 기대되는 최고 t'가 커진다 — 보정의 출발점."""
    vals = [expected_max_z(n) for n in (2, 10, 50, 144, 1000)]
    assert vals == sorted(vals)
    # 144회 탐색이면 t=2 게이트는 잡음 상한 아래 — 이 프로젝트의 핵심 진단
    assert 2.5 < expected_max_z(144) < 2.8
    assert expected_max_z(144) > 2.0


def test_effective_trials_inverts_expected_max_z():
    for n in (12, 60, 144, 500):
        assert effective_trials(expected_max_z(n)) == pytest.approx(n, rel=0.02)


# ------------------------------------------------------ Deflated Sharpe Ratio

def test_dsr_rejects_noise_and_accepts_real_edge():
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 0.01, 2000)
    strong = rng.normal(0.0015, 0.01, 2000)     # 일간 SR 0.15 — 확실한 엣지
    var_sr = 0.0009                              # 전 시험 SR 분산 (sd 0.03 가정)
    assert deflated_sharpe(noise, var_sr, 144)["dsr"] < 0.5
    assert deflated_sharpe(strong, var_sr, 144)["dsr"] > 0.95


def test_dsr_penalises_wider_search():
    """같은 성적이라도 시험을 많이 했으면 DSR이 낮아져야 한다 (선택 편의)."""
    r = np.random.default_rng(7).normal(0.0010, 0.01, 2000)
    few = deflated_sharpe(r, 0.0009, 5)["dsr"]
    many = deflated_sharpe(r, 0.0009, 5000)["dsr"]
    assert few > many
    assert deflated_sharpe(r, 0.0009, 5)["sr0"] < deflated_sharpe(r, 0.0009, 5000)["sr0"]


def test_dsr_handles_degenerate_input():
    assert np.isnan(deflated_sharpe(np.zeros(500), 0.001, 144)["dsr"])
    assert np.isnan(deflated_sharpe(np.array([0.01, 0.02]), 0.001, 144)["dsr"])


# -------------------------------------------------------- 부트스트랩 / RC·SPA

def test_stationary_bootstrap_indices_are_in_range_and_blocky():
    rng = np.random.default_rng(1)
    idx = stationary_bootstrap_idx(400, block=20, draws=50, rng=rng)
    assert idx.shape == (50, 400)
    assert idx.min() >= 0 and idx.max() < 400
    # 블록 보존: 인접 인덱스가 +1로 이어지는 비율이 1-1/block 근처
    cont = ((idx[:, 1:] - idx[:, :-1]) % 400 == 1).mean()
    assert 0.85 < cont < 0.99


def test_reality_check_does_not_flag_pure_noise():
    """엣지가 하나도 없는 25개 시험 — 최고 성적이 좋아 보여도 p가 크게 나와야."""
    res = reality_check(_noise_matrix(), **BOOT)
    assert res["p_rc"] > 0.05
    assert res["p_spa"] > 0.05
    # 귀무 최대 t는 시험 수에 걸맞은 수준 (25개면 대략 2 근처)
    assert 1.5 < res["null_max_t_mean"] < 3.0


def test_reality_check_detects_a_planted_edge():
    mat = _noise_matrix()
    mat[:, 7] += 0.0025                          # 7번 시험에만 진짜 엣지
    res = reality_check(mat, **BOOT)
    assert res["best_idx"] == 7
    assert res["p_spa"] < 0.05
    assert res["p_rc"] < 0.10


def test_spa_is_not_less_powerful_than_reality_check():
    """SPA는 스튜던트화 + 열등 시험 제외로 RC보다 검정력이 높다 (Hansen 2005)."""
    mat = _noise_matrix()
    mat[:, 3] += 0.002
    mat[:, 11] *= 8                              # 변동성만 큰 시험이 RC 귀무를 지배
    res = reality_check(mat, **BOOT)
    assert res["p_spa"] <= res["p_rc"]


def test_reality_check_is_reproducible():
    a = reality_check(_noise_matrix(), **BOOT)
    b = reality_check(_noise_matrix(), **BOOT)
    assert a["p_rc"] == b["p_rc"] and a["p_spa"] == b["p_spa"]


# ------------------------------------------------------------- 보정 p값 규칙

def test_bh_fdr_and_holm_are_ordered_in_strictness():
    p = np.array([1e-9, 1e-5, 0.004, 0.02, 0.3, 0.7])
    bh = bh_fdr(p, 0.10)
    hl = holm(p, 0.05)
    assert bh.sum() >= hl.sum()                  # Holm이 더 엄격
    assert bh[0] and hl[0]
    assert not bh[-1] and not hl[-1]


def test_bh_fdr_rejects_nothing_when_all_p_are_large():
    p = np.linspace(0.2, 0.9, 20)
    assert bh_fdr(p, 0.10).sum() == 0
    assert holm(p, 0.05).sum() == 0


def test_holm_stops_at_first_failure():
    """Holm은 정렬 후 첫 실패에서 멈춘다 — 뒤쪽은 단독 알파를 넘어도 기각 못 한다.

    p=0.04 는 alpha=0.05 단독 기준으로는 유의하지만, 앞순위 0.03 이 0.05/2 문턱을
    넘지 못하는 순간 절차가 멈추므로 함께 탈락한다.
    """
    p = np.array([0.001, 0.03, 0.04])
    assert holm(p, 0.05).tolist() == [True, False, False]


# ------------------------------------------------------------ 일간 행렬 배치

def test_daily_matrix_places_returns_on_exit_dates():
    cal = pd.bdate_range("2026-01-01", periods=10)
    trades = pd.DataFrame({"entry_date": [cal[0], cal[4]],
                           "exit_date": [cal[2], cal[6]],
                           "net_ret": [0.03, -0.01]})
    mat = daily_matrix([{"trades": trades}], cal)
    assert mat.shape == (10, 1)
    assert mat[2, 0] == 0.03 and mat[6, 0] == -0.01
    assert mat.sum() == pytest.approx(0.02)
    assert (mat[[0, 1, 3, 4, 5, 7, 8, 9], 0] == 0).all()   # 무포지션일은 0


def test_daily_matrix_sums_same_day_exits():
    cal = pd.bdate_range("2026-01-01", periods=5)
    trades = pd.DataFrame({"entry_date": [cal[0], cal[1]],
                           "exit_date": [cal[3], cal[3]],
                           "net_ret": [0.01, 0.02]})
    mat = daily_matrix([{"trades": trades}], cal)
    assert mat[3, 0] == pytest.approx(0.03)
