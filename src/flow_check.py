"""flow 스크리닝 부속 진단 — 사전 등록(PREREG_flow.md §5)이 약속한 표.

1. **일 단위 교정 t** — 같은 시장 페어(KODEX200/Lev)는 신호가 동일해 트레이드
   풀 나이브 t가 부푼다 (volbreak/overnight 선례). 같은 청산일 평균 후 검정.
2. **가격 모멘텀 중복 진단** — 외국인 순매수는 가격과 동시상관이 있어 수급
   신호가 사실상 trend 계열과 겹칠 수 있다 (등록된 우려). 같은 코드의 Stage 2
   trend 후보와 보유일 중복률 + 슬리브 상관 병기. |ρ|>0.5 면 같은 리스크 그룹.
3. **홀드아웃 방향 확인** — 게이트 통과분 한정. KOSPI 계열(069500/122630)만
   가능 — KOSDAQ150 2종은 2015 상장이라 '확인 불가 → 편입 보류' (등록 규칙).

측정·기록 전용 — 신호·게이트·판정 기준을 바꾸지 않는다. 재실행 안전.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .backtest import combo_stats
from .config import RESULTS_DIR, load_config
from .etf_swing import (
    build_flags,
    gate_mask,
    gate_row,
    iter_candidates,
    iter_screen_trades,
    simulate,
)
from .flow_data import load_flow
from .holdout import load_holdout_symbol, one_sided_t


def _held_days(trades: pd.DataFrame, index: pd.DatetimeIndex) -> set:
    days: set = set()
    for _, t in trades.iterrows():
        days.update(index[(index >= t["entry_date"]) & (index <= t["exit_date"])])
    return days


def run_flow_check(screen: pd.DataFrame, force: bool = False) -> dict:
    cfg = load_config()
    ecfg = cfg["etf"]
    fcfg = ecfg["strategies"]["flow"]
    cost = ecfg["cost_round_trip"]
    uni = {c: ecfg["universe"][c] for c in fcfg["universe"]}

    items = {i["etf"]: i for i in iter_screen_trades(uni, ["flow"], force)}

    # 1. 풀 t — 나이브 vs 일 단위 교정 (같은 시장 페어의 동일 신호 상관 제거)
    all_tr = pd.concat([i["trades"] for i in items.values() if not i["trades"].empty],
                       ignore_index=True)
    naive_t = one_sided_t(all_tr["net_ret"]) if not all_tr.empty else float("nan")
    daily_pool = (all_tr.groupby("exit_date")["net_ret"].mean()
                  if not all_tr.empty else pd.Series(dtype=float))
    daily_t = one_sided_t(daily_pool)

    # 2. 같은 코드 Stage 2 trend 후보와의 중복 (breakout 등 — 등록된 우려 진단)
    overlap_rows = []
    trend_daily: list[pd.Series] = []
    for cand, df, entry, exit_, max_hold, trailing in iter_candidates(force=False):
        code = str(cand["code"])
        if code not in fcfg["universe"] or cand.get("family", "trend") != "trend":
            continue
        name = ecfg["universe"][code]
        if name not in items or items[name]["trades"].empty:
            continue
        t_tr = simulate(df, entry, exit_, max_hold, cost, trailing=trailing)
        f_days = _held_days(items[name]["trades"], df.index)
        t_days = _held_days(t_tr, df.index)
        overlap_rows.append({
            "etf": name, "trend_strategy": cand["strategy"],
            "flow_days": len(f_days), "trend_days": len(t_days),
            "both": len(f_days & t_days),
            "overlap": len(f_days & t_days) / len(f_days) if f_days else float("nan"),
        })
        if not t_tr.empty:
            trend_daily.append(t_tr.groupby("exit_date")["net_ret"].mean())
    overlap = pd.DataFrame(overlap_rows)

    corr = float("nan")
    if trend_daily and not daily_pool.empty:
        trend_sleeve = (pd.concat(trend_daily, axis=1).mean(axis=1))
        union = daily_pool.index.union(trend_sleeve.index)
        corr = float(daily_pool.reindex(union).fillna(0.0)
                     .corr(trend_sleeve.reindex(union).fillna(0.0)))
    same_group = bool(np.isfinite(corr) and abs(corr) > 0.5)

    # 3. 게이트 통과분 홀드아웃 방향 확인 (KOSPI 계열만 가능 — 등록 규칙)
    mask = screen["strategy"].eq("flow") & gate_mask(screen)
    passers = screen[mask]
    code_by_name = {v: k for k, v in ecfg["universe"].items()}
    hold_rows = []
    for _, r in passers.iterrows():
        code = str(code_by_name[r["etf"]])
        market = fcfg["markets"][code]
        dfh = load_holdout_symbol(code, force=False)
        if dfh.empty or len(dfh) < 120:
            hold_rows.append({"etf": r["etf"], "n": 0, "mean": np.nan, "t_stat": np.nan,
                              "verdict": "확인 불가 (구간 데이터 없음) — 편입 보류"})
            continue
        flow = load_flow(market)["foreign"]
        entry, exit_, max_hold = build_flags(dfh, "flow", flow)
        ht = simulate(dfh, entry, exit_, max_hold, cost)
        st = combo_stats(ht["net_ret"]) if not ht.empty else combo_stats(pd.Series(dtype=float))
        ok = st["n"] > 0 and st["mean"] > 0
        hold_rows.append({"etf": r["etf"], "n": st["n"], "mean": st["mean"],
                          "t_stat": st["t_stat"],
                          "verdict": "양수 — 편입 가능" if ok else "음수 — Stage 2 편입 보류"})
    holdout_df = pd.DataFrame(hold_rows)

    res = {"screen": screen[screen["strategy"].eq("flow")].copy(),
           "naive_t": naive_t, "daily_t": daily_t, "daily_n": len(daily_pool),
           "overlap": overlap, "corr": corr, "same_group": same_group,
           "passers": passers, "holdout": holdout_df}
    _write_report(res, cfg)
    return res


def _write_report(res: dict, cfg: dict) -> None:
    sc = res["screen"].sort_values("t_stat", ascending=False)
    lines = [f"""# flow 스크리닝 진단 (사전 등록 부속표)

생성일: {pd.Timestamp.today().date()} | 등록: PREREG_flow.md +
config `etf.strategies.flow` (2026-08-26) | 본표: etf_screening.csv

## 전 조합 결과 (4테스트 — 게이트: N>=30, 전/후반 양수, t>=2)

| ETF | N | 평균보유 | 평균 | 승률 | 누적 | MDD | t | 전반 | 후반 | 게이트 |
|---|---|---|---|---|---|---|---|---|---|---|"""]
    for _, r in sc.iterrows():
        ok = gate_row(r)
        lines.append(f"| {r['etf']} | {r['n']} | {r['avg_hold']:.1f}일 | {r['mean']:+.3%} "
                     f"| {r['win']:.1%} | {r['cum']:+.1%} | {r['mdd']:.1%} "
                     f"| {r['t_stat']:.2f} | {r['first_mean']:+.3%} "
                     f"| {r['second_mean']:+.3%} | {'**통과**' if ok else '미달'} |")

    lines.append(f"""
## 일 단위 교정 t (같은 시장 페어의 동일 신호 상관 제거)

- 트레이드 풀 나이브 단측 t: **{res['naive_t']:.2f}** (부풀려진 값 — 참고만)
- 같은 청산일 평균 후 단측 t: **{res['daily_t']:.2f}** (관측 {res['daily_n']}일)

## 가격 모멘텀(trend 후보) 중복 진단 (등록된 우려)

- flow 슬리브 vs trend 슬리브 일간 수익 상관(비활동=0): **ρ={res['corr']:.2f}**
- 판정: {'**|ρ|>0.5 — 운영상 같은 리스크 그룹**' if res['same_group']
         else '|ρ|<=0.5 — 수급 신호는 가격 모멘텀과 구분되는 정보'}

| ETF | trend 전략 | flow 보유일 | trend 보유일 | 중복일 | 중복률(flow 기준) |
|---|---|---|---|---|---|""")
    for _, r in res["overlap"].iterrows():
        lines.append(f"| {r['etf']} | {r['trend_strategy']} | {r['flow_days']} "
                     f"| {r['trend_days']} | {r['both']} | {r['overlap']:.1%} |")
    if res["overlap"].empty:
        lines.append("| (대상 없음) | | | | | |")

    lines.append(f"""
## 게이트 통과분 홀드아웃 방향 확인 ({cfg['holdout']['start']} ~ {cfg['holdout']['end']})

비용은 스크리닝과 동일한 일괄 왕복 {cfg['etf']['cost_round_trip']:.2%}.
KOSDAQ150 계열은 2015 상장이라 확인 불가 → 통과해도 편입 보류 (등록 규칙 —
독립 재현 없는 단독 통과는 약하다는 tom의 교훈).
""")
    if res["passers"].empty:
        lines.append("- 게이트 통과 후보 없음 → 홀드아웃 확인 대상 없음. "
                     "**계열 종결** (변형 재시험 금지).")
    else:
        lines += ["| ETF | N | 평균 | t | 결론 |", "|---|---|---|---|---|"]
        for _, r in res["holdout"].iterrows():
            mean = "-" if not np.isfinite(r["mean"]) else f"{r['mean']:+.3%}"
            ts = "-" if not np.isfinite(r["t_stat"]) else f"{r['t_stat']:.2f}"
            lines.append(f"| {r['etf']} | {int(r['n'])} | {mean} | {ts} | {r['verdict']} |")

    lines.append("""
측정 전용 — 이 리포트는 신호·게이트·판정 기준을 바꾸지 않는다.
결과 해석과 Stage 2 편입 여부 기록은 CLAUDE.md 참조.""")
    out = RESULTS_DIR / "flow_screening.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out}", file=sys.stderr)
