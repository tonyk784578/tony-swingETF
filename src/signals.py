"""조건(불리언 시그널) 정의 — 임계값은 config.yaml에서 로드.

각 시그널은 (이름, 카테고리, boolean Series). 같은 카테고리끼리는 조합하지 않는다.
갭다운 매수(역방향/되돌림)는 gap_down / gap_down_big 조건이 그대로 커버한다.
"""

from __future__ import annotations

import pandas as pd

from .config import load_config


def band(series: pd.Series, lo, hi) -> pd.Series:
    """(lo, hi] 구간 판정. lo 초과, hi 이하. null은 무한대 취급."""
    cond = pd.Series(True, index=series.index)
    if lo is not None:
        cond &= series > lo
    if hi is not None:
        cond &= series <= hi
    return cond


def build_signals(master: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    """{signal_name: (category, bool Series)} 반환."""
    cfg = load_config()["signals"]
    sig: dict[str, tuple[str, pd.Series]] = {}

    band_sources = {
        "sox": master["sox_ret"],
        "nvda": master["nvda_ret"],
        "kospi": master["kospi_prev_ret"],
        "gap": master["gap"],
    }
    for cat, series in band_sources.items():
        for label, (lo, hi) in cfg[cat].items():
            sig[f"{cat}_{label}"] = (cat, band(series, lo, hi))

    for label, want in cfg["ma20"].items():
        sig[f"ma20_{label}"] = ("ma20", master["above_ma20"] == want)

    return sig
