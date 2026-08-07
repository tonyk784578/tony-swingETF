"""폭락 사례 검출 + MA 게이트 룩어헤드 테스트."""

import pandas as pd
import pytest

from src.crash_study import entry_gate, find_episodes


def test_find_episodes_detects_and_merges():
    idx = pd.bdate_range("2024-01-01", periods=120)
    vals = ([100.0] * 40            # 고점 형성
            + [85.0] * 10           # -15% 이탈 (사례 1)
            + [95.0] * 10           # 일시 회복 (-5%, merge_gap 이내)
            + [82.0] * 10           # 재이탈 → 병합되어야 함
            + [100.0] * 50)         # 회복
    close = pd.Series(vals, index=idx)
    eps = find_episodes(close, lookback=60, drawdown=-0.10, merge_gap=20)
    assert len(eps) == 1                        # 두 이탈이 한 사례로 병합
    assert eps[0]["start"] == idx[40]
    assert eps[0]["end"] == idx[69]
    assert eps[0]["trough_dd"] == pytest.approx(-0.18)


def test_find_episodes_separate_when_gap_large():
    idx = pd.bdate_range("2024-01-01", periods=200)
    vals = ([100.0] * 40 + [85.0] * 10          # 사례 1
            + [101.0] * 80                      # 신고가 회복 (긴 간격)
            + [88.0] * 10 + [101.0] * 60)       # 사례 2
    close = pd.Series(vals, index=idx)
    eps = find_episodes(close, lookback=60, drawdown=-0.10, merge_gap=20)
    assert len(eps) == 2


def test_entry_gate_uses_previous_close():
    """D 진입 게이트는 D-1 종가 기준 — 당일 종가를 쓰면 룩어헤드."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = pd.Series([100.0] * 9 + [200.0], index=idx)   # 마지막 날에만 MA 위로
    gate = entry_gate(close, ma=5)
    assert not gate.iloc[-1]        # 당일 급등은 다음날 진입에나 반영
    close2 = pd.Series([100.0] * 8 + [200.0, 90.0], index=idx)
    gate2 = entry_gate(close2, ma=5)
    assert gate2.iloc[-1]           # 전일 종가가 MA 위였으므로 오늘 진입 허용
