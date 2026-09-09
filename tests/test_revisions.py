"""정정 감시 분류 — 분배락 소급 조정(균일 배율)과 산발 정정을 가른다.

2026-09-09 규명: FDR 일봉은 분배금 조정가라 분배락마다 이전 전 구간이 같은
배율로 재조정된다. 이를 '데이터 정정'으로 경고하면 분기마다 거짓 경보가 뜬다.
"""

import pandas as pd

from src.data_loader import classify_revision


def _close(vals, start="2026-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype=float)


def test_uniform_rescale_before_boundary_is_classified_as_adjustment():
    old = _close([100, 101, 102, 103, 104, 105])
    new = old.copy()
    new.iloc[:4] *= 0.9915            # 경계(4번째 행)까지 -0.85% 일괄, 이후 불변
    k = classify_revision(old, new)
    assert k["uniform"] is True
    assert abs(k["factor"] - 0.9915) < 1e-6
    assert k["boundary"] == old.index[3].date()


def test_scattered_change_is_not_adjustment():
    old = _close([100, 101, 102, 103, 104, 105])
    new = old.copy()
    new.iloc[1] *= 1.02               # 한 행만 정정
    new.iloc[4] *= 0.97
    assert classify_revision(old, new)["uniform"] is False


def test_mixed_factor_is_not_adjustment():
    old = _close([100, 101, 102, 103, 104, 105])
    new = old.copy()
    new.iloc[:2] *= 0.99
    new.iloc[2:4] *= 0.98             # 배율이 둘 — 균일 아님
    assert classify_revision(old, new)["uniform"] is False


def test_no_change_returns_not_uniform():
    old = _close([100, 101, 102])
    assert classify_revision(old, old.copy())["uniform"] is False
