"""리포트: 통계 CSV, 누적수익 차트, split 검증, 마크다운 요약."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .backtest import combo_stats, condition_mask, run_backtest
from .config import RESULTS_DIR, load_config
from .robustness import find_survivors, yearly_summary

# 검증된 카테고리 팔레트 (light mode, 인접쌍 CVD-safe 순서)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BENCH_GRAY = "#8a8a85"
GRID_GRAY = "#e5e5e2"


def save_stats(stats: pd.DataFrame) -> None:
    out = stats.copy()
    for col in ["mean", "win_rate", "std", "cum_ret", "mdd"]:
        out[col] = out[col].round(6)
    out["t_stat"] = out["t_stat"].round(2)
    out.to_csv(RESULTS_DIR / "stats_all.csv", index=False, encoding="utf-8-sig")


def _equity(net: pd.Series, mask: pd.Series) -> pd.Series:
    """조건 미충족일은 0% (현금 보유)로 두는 누적수익 곡선."""
    r = net.where(mask, 0.0)
    return (1 + r).cumprod()


def plot_equity_curves(master: pd.DataFrame, stats: pd.DataFrame, stock_name: str) -> str:
    cfg = load_config()
    cost = cfg["cost"]["round_trip"]
    top_n = cfg["report"]["top_n_chart"]
    net = master["day_ret"] - cost

    top = stats[(stats["stock"] == stock_name) & stats["rankable"]].head(top_n)

    fig, ax = plt.subplots(figsize=(11, 6))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID_GRAY)
    ax.grid(axis="y", color=GRID_GRAY, linewidth=0.8)
    ax.set_axisbelow(True)

    for i, (_, row) in enumerate(top.iterrows()):
        eq = _equity(net, condition_mask(master, row["condition"]))
        ax.plot(eq.index, eq, linewidth=1.8, color=SERIES_COLORS[i % len(SERIES_COLORS)],
                label=f"{row['condition']} (n={row['n']})")

    bh = master["close"] / master["close"].iloc[0]
    ax.plot(bh.index, bh, linewidth=1.4, color=BENCH_GRAY, linestyle="--", label="Buy & Hold")
    all_days = _equity(net, pd.Series(True, index=master.index))
    ax.plot(all_days.index, all_days, linewidth=1.4, color=BENCH_GRAY, linestyle=":",
            label="Every day open-to-close (net)")

    ax.set_title(f"{stock_name}: top {len(top)} conditions - cumulative net return",
                 loc="left", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    path = RESULTS_DIR / f"equity_curves_{stock_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def split_validation(masters: dict[str, pd.DataFrame], stats: pd.DataFrame) -> pd.DataFrame:
    """전반부 상위 조건이 후반부에도 유효한지 비교."""
    cfg = load_config()
    cost = cfg["cost"]["round_trip"]
    boundary = pd.Timestamp(cfg["split"]["boundary"])
    top_n = cfg["report"]["top_n_split"]
    min_n = cfg["backtest"]["min_samples"]

    rows = []
    for stock_name, master in masters.items():
        first, second = master[master.index < boundary], master[master.index >= boundary]
        first_stats = run_backtest(first, stock_name)
        first_top = first_stats[first_stats["rankable"]
                                & (first_stats["condition"] != "ALL_DAYS (benchmark)")].head(top_n)
        net2 = second["day_ret"] - cost
        for rank, (_, row) in enumerate(first_top.iterrows(), start=1):
            mask2 = condition_mask(second, row["condition"])
            st2 = combo_stats(net2[mask2])
            rows.append({
                "stock": stock_name,
                "condition": row["condition"],
                "first_rank": rank,
                "first_n": row["n"], "first_mean": row["mean"], "first_win": row["win_rate"],
                "second_n": st2["n"], "second_mean": st2["mean"], "second_win": st2["win_rate"],
                "sign_holds": bool(st2["n"] >= min_n and not np.isnan(st2["mean"])
                                   and np.sign(st2["mean"]) == np.sign(row["mean"])
                                   and row["mean"] > 0),
            })
    df = pd.DataFrame(rows)
    out = df.copy()
    for col in ["first_mean", "first_win", "second_mean", "second_win"]:
        out[col] = out[col].round(6)
    out.to_csv(RESULTS_DIR / "split_validation.csv", index=False, encoding="utf-8-sig")
    return df


def _fmt_stats_table(df: pd.DataFrame, top: int = 15) -> str:
    lines = ["| 조건 | N | 평균순수익 | 승률 | 누적수익 | MDD | t-stat |",
             "|---|---|---|---|---|---|---|"]
    for _, r in df.head(top).iterrows():
        lines.append(
            f"| {r['condition']} | {r['n']} | {r['mean']:+.3%} | {r['win_rate']:.1%} "
            f"| {r['cum_ret']:+.1%} | {r['mdd']:.1%} | {r['t_stat']:.2f} |")
    return "\n".join(lines)


def write_report(stats: pd.DataFrame, split_df: pd.DataFrame,
                 masters: dict[str, pd.DataFrame],
                 cost_df: pd.DataFrame | None = None,
                 yearly_df: pd.DataFrame | None = None) -> str:
    cfg = load_config()
    boundary = cfg["split"]["boundary"]
    min_n = cfg["backtest"]["min_samples"]
    today = pd.Timestamp.today().date()

    parts = [f"""# 미국장 갭 패턴 백테스트 리포트

생성일: {today} | 기간: {cfg['data']['start']} ~ 현재 | 거래비용: 왕복 {cfg['cost']['round_trip']:.1%}

## 방법론

- 전략: 조건 충족 시 **당일 시가 매수 → 당일 종가 매도**, 순수익 = (종가/시가 − 1) − 비용
- 룩어헤드 방지: 미국 수익률은 한국 거래일 D보다 **미국 현지 날짜가 앞선 마지막 봉**만 사용
  (전수 검증 통과). 국내 지표는 D−1까지만, 당일 시가는 갭 계산에만 사용.
- 조건: SOX/NVDA/KOSPI 수익률 구간, 20일 이평 위/아래, 시가 갭 구간 — 단독 + 2개 조합
  + 사전 지정 3개 조합. N < {min_n} 조건은 순위 평가 제외.
- 과최적화 검증: {boundary} 기준 전/후반 분할, 전반부 상위 조건의 후반부 성과 확인.
"""]

    for stock_name in masters:
        s = stats[(stats["stock"] == stock_name)
                  & (stats["rankable"] | (stats["condition"] == "ALL_DAYS (benchmark)"))]
        parts.append(f"## {stock_name} — 상위 조건 (N ≥ {min_n})\n\n"
                     + _fmt_stats_table(s)
                     + f"\n\n![equity](equity_curves_{stock_name}.png)\n")

    parts.append(f"## Split 검증 (전반부 상위 {cfg['report']['top_n_split']}개 → 후반부)\n")
    parts.append("| 종목 | 조건 | 전반 순위 | 전반 평균 | 후반 평균 | 후반 N | 부호 유지 |")
    parts.append("|---|---|---|---|---|---|---|")
    for _, r in split_df.iterrows():
        second_mean = "n/a" if np.isnan(r["second_mean"]) else f"{r['second_mean']:+.3%}"
        parts.append(f"| {r['stock']} | {r['condition']} | {r['first_rank']} "
                     f"| {r['first_mean']:+.3%} | {second_mean} | {r['second_n']} "
                     f"| {'O' if r['sign_holds'] else 'X'} |")

    # 결론: 세 기준 동시 충족 — 전체 기간 rankable & 평균>0 & t>=2, 그리고 후반부 부호 유지
    survivor_keys = find_survivors(stats, split_df)
    strong = [r for _, r in split_df.iterrows()
              if (r["stock"], r["condition"]) in survivor_keys]

    if cost_df is not None and not cost_df.empty:
        parts.append("\n## 로버스트니스 — 생존 조건 비용 민감도\n")
        parts.append("| 종목 | 조건 | 왕복비용 | N | 평균순수익 | 승률 | t-stat |")
        parts.append("|---|---|---|---|---|---|---|")
        for _, r in cost_df.iterrows():
            parts.append(f"| {r['stock']} | {r['condition']} | {r['cost']:.1%} | {r['n']} "
                         f"| {r['mean']:+.3%} | {r['win_rate']:.1%} | {r['t_stat']:.2f} |")

    if yearly_df is not None and not yearly_df.empty:
        parts.append("\n## 로버스트니스 — 생존 조건 연도별 안정성\n")
        parts.append("| 종목 | 조건 | 연수 | 음수 연수 | 최악 연도 | 최악 평균 |")
        parts.append("|---|---|---|---|---|---|")
        for _, r in yearly_summary(yearly_df).iterrows():
            parts.append(f"| {r['stock']} | {r['condition']} | {r['years']} | {r['neg_years']} "
                         f"| {r['worst_year']} | {r['worst_mean']:+.3%} |")
        parts.append("\n연도별 상세는 `robustness_yearly.csv` 참고.")

    parts.append("\n## 결론\n")
    total = len(split_df)
    held = int(split_df["sign_holds"].sum())
    parts.append(f"- 전반부 상위 조건 {total}개 중 후반부에서 양(+)의 평균수익 부호를 유지한 조건: "
                 f"**{held}개** ({held / total:.0%})" if total else "- 평가 가능한 조건 없음")
    if strong:
        parts.append(f"- 이 중 전체 기간 t-stat ≥ 2 (통계적으로 유의미한 수준)인 조건: **{len(strong)}개**")
        for r in strong:
            parts.append(f"  - {r['stock']}: `{r['condition']}` "
                         f"(전반 {r['first_mean']:+.3%} / 후반 {r['second_mean']:+.3%})")
        parts.append("- **판정: 위 조건들은 비용 차감 후에도 표본 수·후반부 유지·유의성 3박자를 "
                     "충족한다. 실전 후보 여부는 위 로버스트니스 표(비용 민감도·연도별 안정성)로 "
                     "판단할 것 — 특히 왕복 0.3%에서 t ≥ 2를 유지하는지가 관건이다.**")
    else:
        parts.append("- 전체 기간 t-stat ≥ 2를 동시에 충족하는 조건은 없음")
        parts.append("- **판정: 비용 차감 후 + 후반부 유지 + 유의성 기준을 모두 만족하는 "
                     "견고한 엣지는 확인되지 않았다. 개별 상위 조건은 과최적화 가능성이 높다.**")

    text = "\n".join(parts) + "\n"
    path = RESULTS_DIR / "report.md"
    path.write_text(text, encoding="utf-8")
    return str(path)
