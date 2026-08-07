"""폭락 구간 사례 연구 + 방어 게이트 실험 (config crash_study — 2026-08-07 사전 등록).

Part 1 (기술적 진단): 후보 ETF별로 '60일 고점 대비 -10% 이탈' 폭락 사례를 검출하고,
섀도와 동일한 룰(iter_candidates, 채택 exit 반영)의 트레이드를
사례 밖 / 폭락 중 진입 / 보유 중 피격으로 분해한다. 여기는 진단이지 룰 선택이 아니다.

Part 2 (확증 실험): MA200 게이트 — 신규 진입은 전일 종가 > 200일 이평일 때만.
판정 기준은 refine 레짐 필터와 동일(전/후반 각각 t 유지·개선 AND MDD 개선).
가설이 Part 1 진단과 같은 인샘플에서 나왔으므로 채택돼도 최종 판정은 포워드다.
ETF별 사례는 12~24건이지만 독립된 거시 사건으로는 10건 남짓(같은 폭락이 여러 ETF에
동시 반영) — 폭락 전용 룰은 표본이 작다는 한계를 항상 명시.
"""

from __future__ import annotations

import pandas as pd

from .config import RESULTS_DIR, load_config
from .data_loader import confirmed_cutoff
from .etf_swing import iter_candidates, simulate
from .refinements import _improved, _split_stats


def find_episodes(close: pd.Series, lookback: int, drawdown: float,
                  merge_gap: int) -> list[dict]:
    """60일 고점 대비 drawdown 이하 구간 목록. 인접 구간(merge_gap 영업일 내)은 병합."""
    dd = close / close.rolling(lookback, min_periods=1).max() - 1
    below = dd < drawdown
    runs = (below != below.shift()).cumsum()
    spans = [[g.index[0], g.index[-1]] for _, g in below.groupby(runs) if g.iloc[0]]
    merged: list[list] = []
    for s, e in spans:
        if merged and (close.index.get_loc(s)
                       - close.index.get_loc(merged[-1][1])) <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [{"start": s, "end": e, "trough": dd.loc[s:e].idxmin(),
             "trough_dd": dd.loc[s:e].min()} for s, e in merged]


def entry_gate(close: pd.Series, ma: int) -> pd.Series:
    """MA 게이트 — D 진입(D-1 신호)이므로 D-1 종가 기준으로 shift(1)."""
    return (close > close.rolling(ma).mean()).shift(1).fillna(False).astype(bool)


def _classify(t: pd.Series, episodes: list[dict]) -> str:
    for ep in episodes:
        if ep["start"] <= t["entry_date"] <= ep["end"]:
            return "entered_in"
        if t["entry_date"] < ep["start"] and t["exit_date"] >= ep["start"]:
            return "held_into"
    return "outside"


def run_case_study(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """사례 목록 + 후보별 손익 분해. 반환: (episodes_df, decomposition_df)."""
    cfg = load_config()
    p = cfg["crash_study"]["episode"]
    cost = cfg["etf"]["cost_round_trip"]

    ep_rows, dec_rows = [], []
    seen_codes = set()
    for cand, df, entry, exit_, max_hold, trailing in iter_candidates(
            force, cutoff=confirmed_cutoff()):
        eps = find_episodes(df["Close"], p["lookback"], p["drawdown"], p["merge_gap"])
        if cand["code"] not in seen_codes:
            seen_codes.add(cand["code"])
            for ep in eps:
                ep_rows.append({"etf": cand["name"], "start": ep["start"].date(),
                                "end": ep["end"].date(), "trough": ep["trough"].date(),
                                "trough_dd": ep["trough_dd"]})
        trades = simulate(df, entry, exit_, max_hold, cost, trailing=trailing)
        if trades.empty:
            continue
        kinds = trades.apply(_classify, axis=1, episodes=eps)
        row = {"name": cand["name"], "strategy": cand["strategy"],
               "n": len(trades), "total": trades["net_ret"].sum()}
        for kind in ["outside", "entered_in", "held_into"]:
            sub = trades[kinds == kind]
            row[f"{kind}_n"] = len(sub)
            row[f"{kind}_sum"] = sub["net_ret"].sum()
        in_ep = trades[kinds != "outside"]
        row["worst_in_ep"] = in_ep["net_ret"].min() if len(in_ep) else float("nan")
        dec_rows.append(row)
    return pd.DataFrame(ep_rows), pd.DataFrame(dec_rows)


def run_gate_test(force: bool = False) -> pd.DataFrame:
    """MA200 게이트 1회 판정 — baseline은 섀도가 실제로 쓰는 룰(채택 exit 포함)."""
    cfg = load_config()
    ma = cfg["crash_study"]["gate_ma"]
    cost = cfg["etf"]["cost_round_trip"]
    boundary = pd.Timestamp(cfg["split"]["boundary"])

    rows = []
    for cand, df, entry, exit_, max_hold, trailing in iter_candidates(
            force, cutoff=confirmed_cutoff()):
        gate = entry_gate(df["Close"], ma)
        entry_b = entry.fillna(False).astype(bool)
        base = _split_stats(
            simulate(df, entry_b, exit_, max_hold, cost, trailing=trailing), boundary)
        gated = _split_stats(
            simulate(df, entry_b & gate, exit_, max_hold, cost, trailing=trailing),
            boundary)
        rows.append({"name": cand["name"], "strategy": cand["strategy"],
                     "base_n": base["full"]["n"], "gate_n": gated["full"]["n"],
                     "base_mean": base["full"]["mean"], "gate_mean": gated["full"]["mean"],
                     "base_mdd": base["full"]["mdd"], "gate_mdd": gated["full"]["mdd"],
                     "base_t": base["full"]["t_stat"], "gate_t": gated["full"]["t_stat"],
                     "adopted": _improved(base, gated, check_mdd=True)})
    return pd.DataFrame(rows)


def run_crash_study(force: bool = False) -> dict:
    episodes, dec = run_case_study(force)
    gate = run_gate_test(force)
    episodes.to_csv(RESULTS_DIR / "crash_episodes.csv", index=False, encoding="utf-8-sig")
    dec.to_csv(RESULTS_DIR / "crash_decomposition.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(RESULTS_DIR / "crash_gate.csv", index=False, encoding="utf-8-sig")
    _write_report(episodes, dec, gate)
    return {"episodes": episodes, "decomposition": dec, "gate": gate}


def _write_report(episodes: pd.DataFrame, dec: pd.DataFrame,
                  gate: pd.DataFrame) -> None:
    cfg = load_config()["crash_study"]
    dd_def = f"{cfg['episode']['drawdown']:.0%}"
    lines = [f"""# 폭락 구간 사례 연구 + 방어 게이트 실험 (사전 등록 2026-08-07)

생성일: {pd.Timestamp.today().date()} | 사례 정의: 60일 고점 대비 {dd_def} 이하
(20영업일 내 재이탈 병합) | 게이트: 신규 진입 시 전일 종가 > MA{cfg['gate_ma']}

## 1. 폭락 사례 목록 (ETF 자체 가격 기준)

| ETF | 구간 | 저점 | 저점 낙폭 |
|---|---|---|---|"""]
    for _, r in episodes.iterrows():
        lines.append(f"| {r['etf']} | {r['start']} ~ {r['end']} | {r['trough']} "
                     f"| {r['trough_dd']:.1%} |")

    lines.append("""
## 2. 손익 분해 — 폭락이 어디서 아픈가 (섀도와 동일 룰)

| 후보 | N | 합산 | 사례 밖 | 폭락 중 진입 | 보유 중 피격 | 사례 내 최악 |
|---|---|---|---|---|---|---|""")
    for _, r in dec.iterrows():
        lines.append(
            f"| {r['name']} {r['strategy']} | {r['n']} | {r['total']:+.1%} "
            f"| {r['outside_sum']:+.1%} ({r['outside_n']}건) "
            f"| {r['entered_in_sum']:+.1%} ({r['entered_in_n']}건) "
            f"| {r['held_into_sum']:+.1%} ({r['held_into_n']}건) "
            f"| {r['worst_in_ep']:+.1%} |")

    lines.append("""
## 3. MA200 게이트 판정 (채택 기준: 전/후반 각각 t 유지·개선 AND MDD 개선)

| 후보 | N(전→후) | 평균(전→후) | MDD(전→후) | t(전→후) | 판정 |
|---|---|---|---|---|---|""")
    for _, r in gate.iterrows():
        lines.append(f"| {r['name']} {r['strategy']} | {r['base_n']}→{r['gate_n']} "
                     f"| {r['base_mean']:+.3%}→{r['gate_mean']:+.3%} "
                     f"| {r['base_mdd']:.1%}→{r['gate_mdd']:.1%} "
                     f"| {r['base_t']:.2f}→{r['gate_t']:.2f} "
                     f"| {'채택' if r['adopted'] else '기각'} |")

    n_adopt = int(gate["adopted"].sum())
    lines.append(f"""
**게이트 판정 통과: {n_adopt}건.**

주의사항 (사전 등록 시점에 기록):
- ETF별 사례는 12~24건이나 독립 거시 사건은 10건 남짓(동일 폭락의 동시 반영) —
  폭락 전용 룰은 어떤 것이든 소표본 위에 서 있다.
- 게이트 가설은 사례 진단과 같은 인샘플에서 나왔다. 채택되더라도 인샘플 확증일 뿐,
  최종 판정은 포워드(섀도)에서 한다. 기각이면 그대로 종료 — 임계값 변경 재실험 금지.
- 기존 방어층: trailing -5%(채택 2후보), trend_ride 정배열 게이트,
  포트폴리오 총노출 0.53(-8% MDD 캘리브레이션), 동일 그룹 중복 금지.
""")
    (RESULTS_DIR / "crash_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
