"""로버스트니스 검증: 생존 조건 자동 탐지 → 비용 민감도 + 연도별 분해.

생존 조건 = split 검증에서 후반부 부호 유지(sign_holds) AND 전체 기간 rankable
AND t-stat >= 2 를 모두 충족한 (종목, 조건) 쌍.
"""

from __future__ import annotations

import pandas as pd

from .backtest import combo_stats, condition_mask
from .config import RESULTS_DIR, load_config


def find_survivors(stats: pd.DataFrame, split_df: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    for _, r in split_df[split_df["sign_holds"]].iterrows():
        full = stats[(stats["stock"] == r["stock"]) & (stats["condition"] == r["condition"])]
        if not full.empty and full.iloc[0]["rankable"] and full.iloc[0]["t_stat"] >= 2:
            out.append((r["stock"], r["condition"]))
    return out


def cost_sensitivity(masters: dict[str, pd.DataFrame],
                     survivors: list[tuple[str, str]]) -> pd.DataFrame:
    costs = load_config()["robustness"]["costs"]
    rows = []
    for stock, cond in survivors:
        master = masters[stock]
        mask = condition_mask(master, cond)
        for cost in costs:
            st = combo_stats((master["day_ret"] - cost)[mask])
            rows.append({"stock": stock, "condition": cond, "cost": cost, **st})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(RESULTS_DIR / "robustness_cost.csv", index=False, encoding="utf-8-sig")
    return df


def yearly_decomposition(masters: dict[str, pd.DataFrame],
                         survivors: list[tuple[str, str]]) -> pd.DataFrame:
    cost = load_config()["cost"]["round_trip"]
    rows = []
    for stock, cond in survivors:
        master = masters[stock]
        net = (master["day_ret"] - cost)[condition_mask(master, cond)]
        for year, grp in net.groupby(net.index.year):
            rows.append({"stock": stock, "condition": cond, "year": year, "n": len(grp),
                         "mean": grp.mean(), "win_rate": (grp > 0).mean()})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(RESULTS_DIR / "robustness_yearly.csv", index=False, encoding="utf-8-sig")
    return df


def yearly_summary(yearly: pd.DataFrame) -> pd.DataFrame:
    """생존 조건별 요약: 총 연수, 음수 연수, 최악 연도."""
    rows = []
    for (stock, cond), g in yearly.groupby(["stock", "condition"]):
        worst = g.loc[g["mean"].idxmin()]
        rows.append({"stock": stock, "condition": cond,
                     "years": len(g), "neg_years": int((g["mean"] < 0).sum()),
                     "worst_year": int(worst["year"]), "worst_mean": worst["mean"]})
    return pd.DataFrame(rows)
