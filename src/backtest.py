"""조건 조합별 open→close 백테스트 및 통계."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .config import load_config
from .signals import build_signals


def combo_stats(net_ret: pd.Series) -> dict:
    """조건 충족일의 순수익률 시리즈 → 통계 dict."""
    n = len(net_ret)
    if n == 0:
        return {"n": 0, "mean": np.nan, "win_rate": np.nan, "std": np.nan,
                "cum_ret": np.nan, "mdd": np.nan, "t_stat": np.nan}
    equity = (1 + net_ret).cumprod()
    peak = equity.cummax()
    mdd = (equity / peak - 1).min()
    std = net_ret.std(ddof=1) if n > 1 else np.nan
    t_stat = net_ret.mean() / (std / np.sqrt(n)) if n > 1 and std > 0 else np.nan
    return {
        "n": n,
        "mean": net_ret.mean(),
        "win_rate": (net_ret > 0).mean(),
        "std": std,
        "cum_ret": equity.iloc[-1] - 1,
        "mdd": mdd,
        "t_stat": t_stat,
    }


def enumerate_combos(signals: dict[str, tuple[str, pd.Series]]) -> list[tuple[str, ...]]:
    """단독 + 2개 조합(같은 카테고리 제외) + config 지정 추가 조합."""
    cfg = load_config()["backtest"]
    names = list(signals)
    combos: list[tuple[str, ...]] = [(n,) for n in names]
    if cfg["max_combo"] >= 2:
        for a, b in combinations(names, 2):
            if signals[a][0] != signals[b][0]:
                combos.append((a, b))
    for extra in cfg.get("extra_combos") or []:
        t = tuple(extra)
        if all(n in signals for n in t) and t not in combos:
            combos.append(t)
    return combos


def run_backtest(master: pd.DataFrame, stock_name: str) -> pd.DataFrame:
    """전체 조건 조합 통계표. 벤치마크(매일 진입) 행 포함."""
    cfg = load_config()
    cost = cfg["cost"]["round_trip"]
    min_n = cfg["backtest"]["min_samples"]

    net = master["day_ret"] - cost
    signals = build_signals(master)

    rows = []
    rows.append({"stock": stock_name, "condition": "ALL_DAYS (benchmark)",
                 **combo_stats(net), "rankable": False})
    for combo in enumerate_combos(signals):
        mask = pd.Series(True, index=master.index)
        for name in combo:
            mask &= signals[name][1]
        st = combo_stats(net[mask])
        rows.append({"stock": stock_name, "condition": " & ".join(combo),
                     **st, "rankable": st["n"] >= min_n})

    df = pd.DataFrame(rows)
    return df.sort_values("mean", ascending=False).reset_index(drop=True)


def condition_mask(master: pd.DataFrame, condition: str) -> pd.Series:
    """'a & b' 형태 조건명 → boolean mask (차트/split 재계산용)."""
    signals = build_signals(master)
    mask = pd.Series(True, index=master.index)
    for name in condition.split(" & "):
        mask &= signals[name][1]
    return mask
