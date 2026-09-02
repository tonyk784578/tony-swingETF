"""overnight 스크리닝 부속 진단 — 사전 등록(PREREG_overnight.md)이 약속한 표.

스크리닝 본표(etf_screening.csv)에 overnight 12조합이 포함되지만, 등록문이
약속한 세 가지는 본표 형식이 담지 못해 여기서 산출한다:

1. **일 단위 교정 t** — 12종이 같은 날 동시 신호가 나는 구조라 트레이드 풀
   나이브 t는 부풀려진다 (volbreak 선례: 나이브 8.75 → 일 단위 2.38).
   같은 청산일의 트레이드를 평균해 관측치 1개로 만든 뒤 검정 (holdout._pool 관례).
2. **volbreak 중복 진단** — 두 전략 모두 야간을 보유하므로 (a) 코드별 신호일
   중복률, (b) 슬리브 일간 수익 상관을 병기. |ρ|>0.5 면 운영상 같은 리스크
   그룹으로 취급한다 (등록된 약속 — 노출 합산 캡).
3. **홀드아웃 방향 확인** — 게이트 통과 후보에 한해 홀드아웃 구간(config
   holdout, 2009~2014)에 같은 규칙을 1회 적용, 평균 부호 확인. 음수면
   Stage 2 편입 보류 (우연 통과 가능성 우세로 기록).

측정·기록 전용 — 신호·게이트·판정 기준을 바꾸지 않는다. 계산은 결정적이라
재실행 안전 (리포트 재생성 용도).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .backtest import combo_stats
from .config import RESULTS_DIR, load_config
from .etf_swing import gate_mask, gate_row, iter_screen_trades, simulate_overnight
from .holdout import load_holdout_symbol, one_sided_t


def _sleeve_daily(trades_by_etf: dict[str, pd.DataFrame]) -> pd.Series:
    """슬리브 일간 수익 — 같은 청산일 트레이드의 평균 (동일비중 슬리브 근사)."""
    frames = [t for t in trades_by_etf.values() if not t.empty]
    if not frames:
        return pd.Series(dtype=float)
    allt = pd.concat(frames, ignore_index=True)
    return allt.groupby("exit_date")["net_ret"].mean()


def run_overnight_check(screen: pd.DataFrame, force: bool = False) -> dict:
    cfg = load_config()
    ecfg = cfg["etf"]
    cost = ecfg["cost_round_trip"]
    entry_above = ecfg["strategies"]["overnight"]["entry_above"]

    # 트레이드 재산출 — 스크리닝 통계와 반드시 같은 경로(iter_screen_trades)
    on_trades: dict[str, pd.DataFrame] = {}
    vb_trades: dict[str, pd.DataFrame] = {}
    for item in iter_screen_trades(ecfg["universe"], ["overnight", "volbreak"], force):
        dest = on_trades if item["strategy"] == "overnight" else vb_trades
        dest[item["etf"]] = item["trades"]

    # 1. 풀 t — 나이브 vs 일 단위 교정 (같은 날 12종 동시 신호 상관 제거)
    all_on = pd.concat([t for t in on_trades.values() if not t.empty],
                       ignore_index=True)
    naive_t = one_sided_t(all_on["net_ret"]) if not all_on.empty else float("nan")
    daily_pool = (all_on.groupby("exit_date")["net_ret"].mean()
                  if not all_on.empty else pd.Series(dtype=float))
    daily_t = one_sided_t(daily_pool)

    # 2. volbreak 중복 — 코드별 신호일 중복률 + 슬리브 일간 수익 상관
    overlap_rows = []
    for etf in on_trades:
        on_days = set(on_trades[etf]["entry_date"]) if not on_trades[etf].empty else set()
        vb_days = (set(vb_trades[etf]["entry_date"])
                   if etf in vb_trades and not vb_trades[etf].empty else set())
        if on_days:
            overlap_rows.append({
                "etf": etf, "on_days": len(on_days), "vb_days": len(vb_days),
                "both": len(on_days & vb_days),
                "overlap": len(on_days & vb_days) / len(on_days),
            })
    overlap = pd.DataFrame(overlap_rows)

    on_sleeve = _sleeve_daily(on_trades)
    vb_sleeve = _sleeve_daily(vb_trades)
    union = on_sleeve.index.union(vb_sleeve.index)
    # 슬리브 관점(비활동일=현금 0%) 상관 + 동시 활동일만의 상관을 함께 병기
    corr_union = float(on_sleeve.reindex(union).fillna(0.0)
                       .corr(vb_sleeve.reindex(union).fillna(0.0)))
    co_active = on_sleeve.index.intersection(vb_sleeve.index)
    corr_active = (float(on_sleeve.loc[co_active].corr(vb_sleeve.loc[co_active]))
                   if len(co_active) >= 2 else float("nan"))
    # 등록 규칙은 |ρ|>0.5 (부호 무관) — abs 없이 비교하면 음의 상관을 놓친다
    same_group = abs(corr_union) > 0.5 or abs(corr_active) > 0.5

    # 3. 게이트 통과분 홀드아웃 방향 확인 (사전 등록 — 통과 후보에 한해 1회)
    mask = screen["strategy"].eq("overnight") & gate_mask(screen)
    passers = screen[mask]
    code_by_name = {v: k for k, v in ecfg["universe"].items()}
    hold_rows = []
    for _, r in passers.iterrows():
        df = load_holdout_symbol(str(code_by_name[r["etf"]]), force=False)
        if df.empty or len(df) < 120:
            hold_rows.append({"etf": r["etf"], "n": 0, "mean": np.nan,
                              "t_stat": np.nan,
                              "verdict": f"구간 데이터 {len(df)}일 — 확인 불가"})
            continue
        ht = simulate_overnight(df, entry_above, cost)
        st = combo_stats(ht["net_ret"]) if not ht.empty else combo_stats(pd.Series(dtype=float))
        ok = st["n"] > 0 and st["mean"] > 0
        hold_rows.append({"etf": r["etf"], "n": st["n"], "mean": st["mean"],
                          "t_stat": st["t_stat"],
                          "verdict": "양수 — 편입 가능" if ok else "음수 — Stage 2 편입 보류"})
    holdout_df = pd.DataFrame(hold_rows)

    res = {"screen": screen[screen["strategy"].eq("overnight")].copy(),
           "naive_t": naive_t, "daily_t": daily_t, "daily_n": len(daily_pool),
           "overlap": overlap, "corr_union": corr_union, "corr_active": corr_active,
           "same_group": same_group, "passers": passers, "holdout": holdout_df}
    _write_report(res, cfg)
    return res


def _write_report(res: dict, cfg: dict) -> None:
    sc = res["screen"].sort_values("t_stat", ascending=False)
    lines = [f"""# overnight 스크리닝 진단 (사전 등록 부속표)

생성일: {pd.Timestamp.today().date()} | 등록: PREREG_overnight.md +
config `etf.strategies.overnight` (2026-08-26) | 본표: etf_screening.csv

## 전 조합 결과 (12테스트 — 게이트: N>=30, 전/후반 양수, t>=2)

| ETF | N | 평균 | 승률 | 누적 | MDD | t | 전반 | 후반 | 게이트 |
|---|---|---|---|---|---|---|---|---|---|"""]
    for _, r in sc.iterrows():
        ok = gate_row(r)
        lines.append(f"| {r['etf']} | {r['n']} | {r['mean']:+.3%} | {r['win']:.1%} "
                     f"| {r['cum']:+.1%} | {r['mdd']:.1%} | {r['t_stat']:.2f} "
                     f"| {r['first_mean']:+.3%} | {r['second_mean']:+.3%} "
                     f"| {'**통과**' if ok else '미달'} |")

    lines.append(f"""
## 일 단위 교정 t (등록된 병기 — 12종 동시 신호 상관 제거)

- 트레이드 풀 나이브 단측 t: **{res['naive_t']:.2f}** (부풀려진 값 — 참고만)
- 같은 청산일 평균 후 단측 t: **{res['daily_t']:.2f}** (관측 {res['daily_n']}일)

## volbreak 중복 진단 (등록된 약속)

- 슬리브 일간 수익 상관: 전 거래일(비활동=0) **ρ={res['corr_union']:.2f}**,
  동시 활동일만 ρ={res['corr_active']:.2f}
- 판정: {'**|ρ|>0.5 — 운영상 같은 리스크 그룹** (노출 합산 캡 적용 약속)'
         if res['same_group'] else '|ρ|<=0.5 — 별도 슬리브 운용 가능'}

| ETF | overnight 신호일 | volbreak 트리거일 | 중복일 | 중복률 |
|---|---|---|---|---|""")
    for _, r in res["overlap"].sort_values("overlap", ascending=False).iterrows():
        lines.append(f"| {r['etf']} | {r['on_days']} | {r['vb_days']} "
                     f"| {r['both']} | {r['overlap']:.1%} |")

    lines.append(f"""
## 게이트 통과분 홀드아웃 방향 확인 ({cfg['holdout']['start']} ~ {cfg['holdout']['end']})

비용은 스크리닝과 동일한 일괄 왕복 {cfg['etf']['cost_round_trip']:.2%}.
평균 음수면 Stage 2 편입 보류 — 등록 시 못 박은 규칙이다.
""")
    if res["passers"].empty:
        lines.append("- 게이트 통과 후보 없음 → 홀드아웃 확인 대상 없음. "
                     "**계열 종결** (등록된 예상 시나리오 — 컷 변경 재시험 금지).")
    else:
        lines += ["| ETF | N | 평균 | t | 결론 |", "|---|---|---|---|---|"]
        for _, r in res["holdout"].iterrows():
            mean = "-" if not np.isfinite(r["mean"]) else f"{r['mean']:+.3%}"
            ts = "-" if not np.isfinite(r["t_stat"]) else f"{r['t_stat']:.2f}"
            lines.append(f"| {r['etf']} | {int(r['n'])} | {mean} | {ts} | {r['verdict']} |")

    lines.append("""
측정 전용 — 이 리포트는 신호·게이트·판정 기준을 바꾸지 않는다.
결과 해석과 Stage 2 편입 여부 기록은 CLAUDE.md 참조.""")
    out = RESULTS_DIR / "overnight_screening.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out}", file=sys.stderr)
