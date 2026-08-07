"""시그널 경계값·조합 열거·통계 계산 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import combo_stats, enumerate_combos
from src.signals import band, build_signals


def test_band_lo_exclusive_hi_inclusive():
    s = pd.Series([0.01, 0.0100001, 0.02, 0.021])
    hit = band(s, 0.01, 0.02)
    assert hit.tolist() == [False, True, True, False]


def test_band_open_ended():
    s = pd.Series([-0.05, 0.0, 0.05])
    assert band(s, 0.01, None).tolist() == [False, False, True]
    assert band(s, None, -0.01).tolist() == [True, False, False]


def _tiny_master(n=10):
    idx = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "sox_ret": rng.normal(0, 0.02, n), "ixic_ret": rng.normal(0, 0.02, n),
        "nvda_ret": rng.normal(0, 0.03, n), "kospi_prev_ret": rng.normal(0, 0.01, n),
        "gap": rng.normal(0, 0.01, n), "above_ma20": rng.random(n) > 0.5,
    }, index=idx)


def test_no_same_category_combos():
    signals = build_signals(_tiny_master())
    for combo in enumerate_combos(signals):
        cats = [signals[name][0] for name in combo]
        assert len(cats) == len(set(cats)), f"same-category combo: {combo}"


def test_signal_categories_are_exhaustive():
    m = _tiny_master()
    signals = build_signals(m)
    # ma20 위/아래는 상호배타·전체포괄이어야 한다
    above = signals["ma20_above"][1]
    below = signals["ma20_below"][1]
    assert ((above ^ below).all())


def test_combo_stats_known_values():
    r = pd.Series([0.01, -0.01, 0.02])
    st = combo_stats(r)
    assert st["n"] == 3
    assert st["mean"] == pytest.approx(r.mean())
    assert st["win_rate"] == pytest.approx(2 / 3)
    equity = (1 + r).cumprod()
    assert st["cum_ret"] == pytest.approx(equity.iloc[-1] - 1)
    # MDD: 고점(1.01) 대비 저점(1.01*0.99)
    assert st["mdd"] == pytest.approx(-0.01)


def test_combo_stats_empty():
    st = combo_stats(pd.Series([], dtype=float))
    assert st["n"] == 0 and np.isnan(st["mean"])
