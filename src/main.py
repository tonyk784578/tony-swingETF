"""CLI: python -m src.main {download|verify|backtest|report|all} [--force]"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .align import build_master, verify_alignment
from .backtest import run_backtest
from .config import RESULTS_DIR, ensure_dirs, load_config
from .data_loader import load_all, summarize
from .paper import forward_summary, preview, update_ledger
from .report import plot_equity_curves, save_stats, split_validation, write_report
from .robustness import cost_sensitivity, find_survivors, yearly_decomposition, yearly_summary


def _load_masters(force: bool = False):
    data = load_all(force)
    stocks = list(load_config()["data"]["kr_stocks"].values())
    return data, {name: build_master(data, name) for name in stocks}


def cmd_download(force: bool) -> None:
    from .data_loader import load_symbol

    data = load_all(force)
    cfg = load_config()
    # ETF Stage 2 후보 + 확장 로테이션 유니버스도 함께 갱신 — 저녁 cron의
    # ETF 장부·로테이션 리플레이가 최신 봉을 보도록
    codes = {str(cand["code"]) for cand in cfg["etf_paper"]["candidates"]}
    if cfg.get("etf_rotation2", {}).get("freeze"):
        from .rotation import rotation2_universe

        codes |= set(rotation2_universe())
    for code in sorted(codes):
        load_symbol(code, "kr", force)
    # 외국인 수급(flow)은 기각 종결(2026-08-26) — 일일 수집 안 함 (ewy 관례:
    # 소급 가능 데이터는 새 가설 등록 시점에 받는다). 캐시는 재현용 잔존,
    # 백필은 flow_data.update_flow_caches() 수동 호출로 언제든 복원 가능.
    print(summarize(data))


def cmd_verify() -> None:
    _, masters = _load_masters()
    ok = all(verify_alignment(m, name) for name, m in masters.items())
    if not ok:
        sys.exit("alignment verification FAILED — backtest must not proceed")
    print("\nAll alignment checks passed.")


def cmd_backtest() -> pd.DataFrame:
    _, masters = _load_masters()
    stats = pd.concat([run_backtest(m, name) for name, m in masters.items()],
                      ignore_index=True)
    save_stats(stats)
    pd.set_option("display.width", 200)
    for name in masters:
        s = stats[(stats["stock"] == name) & stats["rankable"]].head(10)
        print(f"\n=== {name}: top 10 rankable conditions ===")
        print(s[["condition", "n", "mean", "win_rate", "cum_ret", "mdd", "t_stat"]]
              .to_string(index=False,
                         formatters={"mean": "{:+.3%}".format, "win_rate": "{:.1%}".format,
                                     "cum_ret": "{:+.1%}".format, "mdd": "{:.1%}".format,
                                     "t_stat": "{:.2f}".format}))
    return stats


def _run_robustness(masters, stats, split_df):
    survivors = find_survivors(stats, split_df)
    if not survivors:
        print("no surviving conditions — robustness step skipped")
        return None, None
    cost_df = cost_sensitivity(masters, survivors)
    yearly_df = yearly_decomposition(masters, survivors)
    print(f"\n=== robustness: {len(survivors)} surviving condition(s) ===")
    print(cost_df[["stock", "condition", "cost", "n", "mean", "win_rate", "t_stat"]]
          .to_string(index=False,
                     formatters={"cost": "{:.1%}".format, "mean": "{:+.3%}".format,
                                 "win_rate": "{:.1%}".format, "t_stat": "{:.2f}".format}))
    print()
    print(yearly_summary(yearly_df)
          .to_string(index=False, formatters={"worst_mean": "{:+.3%}".format}))
    return cost_df, yearly_df


def cmd_robustness() -> None:
    _, masters = _load_masters()
    stats = pd.concat([run_backtest(m, name) for name, m in masters.items()],
                      ignore_index=True)
    split_df = split_validation(masters, stats)
    _run_robustness(masters, stats, split_df)


def cmd_paper(show_preview: bool, force: bool) -> None:
    from .etf_paper import etf_forward_summary, preview_etf, update_etf_ledger

    data, masters = _load_masters(force)
    if show_preview:
        preview(data)
        preview_etf(force)
        from .snapshot import record_market_snapshot

        record_market_snapshot()   # 개장 전 세계 상태 축적 (소급 불가 데이터)
        from .brief import STATUS_PATH, write_status

        write_status("morning", notify=True)
        print(f"\nbrief: {STATUS_PATH}")
        return
    ledger, added = update_ledger(masters)
    print(f"ledger: {added} new row(s) appended "
          f"({int(ledger['triggered'].sum())} trades / {len(ledger)} rows total)")
    trades = ledger[ledger["triggered"] == True]  # noqa: E712
    if not trades.empty:
        print("\n=== recent trades ===")
        print(trades.tail(10)[["date", "stock", "condition", "gap", "net_ret"]]
              .to_string(index=False))
    summary = forward_summary(masters, ledger)
    print("\n=== forward vs in-sample ===")
    print(summary.to_string(index=False,
                            formatters={"fwd_mean": "{:+.3%}".format,
                                        "fwd_sized_mean": "{:+.3%}".format,
                                        "fwd_win": "{:.1%}".format,
                                        "fwd_cum": "{:+.2%}".format,
                                        "insample_mean": "{:+.3%}".format,
                                        "insample_win": "{:.1%}".format}))

    etf_ledger, etf_added, status = update_etf_ledger(force)
    print(f"\n[ETF] ledger: {etf_added} new closed trade(s), {len(etf_ledger)} total")
    if etf_added:
        print(etf_ledger.tail(etf_added).to_string(index=False))
    print("\n=== [ETF] forward vs in-sample ===")
    print(etf_forward_summary(etf_ledger, status)
          .to_string(index=False, formatters={"fwd_mean": "{:+.3%}".format,
                                              "fwd_cum": "{:+.2%}".format,
                                              "ok_mean": "{:+.3%}".format,
                                              "insample_mean": "{:+.3%}".format}))
    from .daily_settle import update_daily_settlement

    update_daily_settlement(etf_ledger, status)

    from .brief import STATUS_PATH, write_status

    write_status("evening")
    print(f"\nbrief: {STATUS_PATH}")


def cmd_daily() -> None:
    """일별 정산 갱신·조회 — 장부 리플레이(멱등) 후 가상 계좌 일별 CSV 재생성."""
    from .daily_settle import update_daily_settlement
    from .etf_paper import update_etf_ledger

    ledger, added, status = update_etf_ledger(False)
    if added:
        print(f"[ETF] ledger: {added} new closed trade(s) appended")
    update_daily_settlement(ledger, status, show_tail=15)


def cmd_ml() -> None:
    from .ml_experiment import run_experiment

    _, masters = _load_masters()
    res = run_experiment(masters)
    fmt = {"mean": "{:+.3%}".format, "win_rate": "{:.1%}".format,
           "cum_ret": "{:+.1%}".format, "mdd": "{:.1%}".format, "t_stat": "{:.2f}".format}
    rows = pd.DataFrame([{"strategy": k, **v} for k, v in
                         [("ML", res["ml"]), ("rule", res["rule"]), ("all_days", res["all"])]])
    print(rows[["strategy", "n", "mean", "win_rate", "cum_ret", "mdd", "t_stat"]]
          .to_string(index=False, formatters=fmt))
    print("\nfeature importance:", ", ".join(f"{k} {v:.0%}" for k, v in res["importance"].items()))
    print("gates:", res["gates"])
    print("verdict:", "채택 — ML이 룰을 이김" if res["verdict"] else "기각 — 수동 룰 유지")
    print("report:", RESULTS_DIR / "ml_report.md")


def cmd_analysis(force: bool) -> None:
    from .analysis import (
        strength_analysis,
        transfer_validation,
        write_analysis_report,
        year_diagnosis,
    )

    data, masters = _load_masters()
    pct = "{:+.3%}".format
    strength = strength_analysis(masters)
    print("=== signal strength (NVDA bins) ===")
    print(strength[["nvda_bin", "n", "mean", "win_rate", "t_stat"]]
          .to_string(index=False, formatters={"mean": pct, "win_rate": "{:.1%}".format,
                                              "t_stat": "{:.2f}".format}))
    diag = year_diagnosis(masters)
    print("\n=== loss-year diagnosis ===")
    print(diag.to_string(formatters={"mean_net": pct, "win": "{:.1%}".format,
                                     "avg_gap": pct, "avg_nvda": pct,
                                     "avg_kospi_prev": pct, "avg_vol20": "{:.2%}".format}))
    transfer = transfer_validation(data, force)
    print("\n=== transfer validation ===")
    print(transfer[["stock", "n", "mean", "win", "t_stat", "first_mean", "second_mean",
                    "sign_holds"]]
          .to_string(index=False, formatters={"mean": pct, "win": "{:.1%}".format,
                                              "t_stat": "{:.2f}".format,
                                              "first_mean": pct, "second_mean": pct}))
    print("\nreport:", write_analysis_report(strength, diag, transfer))


def cmd_sizing() -> None:
    from .sizing import run_sizing_experiment

    _, masters = _load_masters()
    res = run_sizing_experiment(masters)
    print(res["table"].to_string(index=False,
                                 formatters={"ret_on_capital": "{:+.3%}".format,
                                             "mean_port": "{:+.3%}".format,
                                             "sharpe_trade": "{:.3f}".format,
                                             "cum_ret": "{:+.1%}".format,
                                             "mdd": "{:.1%}".format}))
    print("\n역사적 지지 (전/후반 각각 tiered Sharpe > flat):",
          "지지" if res["supported"] else "기각")
    print("최종 판정은 섀도 포워드 성과(fwd_sized_mean)로 —", RESULTS_DIR / "sizing_experiment.md")


def cmd_minute() -> None:
    from .minute_data import collect_minute, coverage

    df = collect_minute()
    days = coverage(df)
    span = f"{days.min().date()} ~ {days.max().date()}" if len(days) else "없음"
    print(f"minute bars: {len(df)} rows, {len(days)} trading days ({span})")


def cmd_stoploss() -> None:
    from .stoploss import run_stoploss_study

    _, masters = _load_masters()
    res = run_stoploss_study(masters)
    print(f"분봉 확보 발동일: {res['n_days']}건 — "
          f"{'판정 가능' if res['ready'] else '표본 부족(판정 보류)'}")
    if not res["summary"].empty:
        print(res["summary"].to_string(
            formatters={"mean": "{:+.3%}".format, "min": "{:+.2%}".format,
                        "win": "{:.0%}".format}))
    print("report:", RESULTS_DIR / "stoploss.md")


def cmd_etf(force: bool) -> None:
    from .etf_swing import run_screening

    df = run_screening(force)
    pct = "{:+.3%}".format
    show = df[df["rankable"]].head(15)
    print(f"=== ETF swing screening: top 15 of {len(df)} (rankable, by t-stat) ===")
    print(show[["etf", "strategy", "n", "avg_hold", "mean", "win", "cum", "mdd",
                "t_stat", "sign_holds"]]
          .to_string(index=False,
                     formatters={"avg_hold": "{:.1f}".format, "mean": pct,
                                 "win": "{:.1%}".format, "cum": "{:+.1%}".format,
                                 "mdd": "{:.1%}".format, "t_stat": "{:.2f}".format}))
    from .etf_swing import gate_mask, run_ext_screening, write_strategy_compare

    passed = df[gate_mask(df)]
    print(f"\nStage-2 후보 (N>=30, 전/후반 양수, t>=2): {len(passed)}개")
    print("report:", RESULTS_DIR / "etf_screening.md")

    ext = run_ext_screening(force)
    if ext is not None:
        print(f"\n=== 확장 유니버스 스크리닝 (신자산 x 생존 계열, {len(ext)}조합) ===")
        print(ext[ext["rankable"]][["etf", "strategy", "n", "avg_hold", "mean", "win",
                                    "cum", "mdd", "t_stat", "sign_holds"]]
              .to_string(index=False,
                         formatters={"avg_hold": "{:.1f}".format, "mean": pct,
                                     "win": "{:.1%}".format, "cum": "{:+.1%}".format,
                                     "mdd": "{:.1%}".format, "t_stat": "{:.2f}".format}))

        ext_pass = ext[gate_mask(ext)]
        print(f"확장 통과 (게이트 동일): {len(ext_pass)}개")
        print("report:", RESULTS_DIR / "etf_ext_screening.md")

    compare = write_strategy_compare()
    print("\n=== 전략별 최종 성과 비교 ===")
    print(compare.to_string(index=False,
                            formatters={"통과율": "{:.0%}".format,
                                        "평균수익(전조합)": "{:+.2%}".format}))
    print("report:", RESULTS_DIR / "strategy_compare.md")

    # overnight 부속 진단 (사전 등록 약속 — 교정 t·volbreak 중복·홀드아웃 확인).
    # 데이터는 직전 run_screening 이 이미 갱신했으므로 재다운로드하지 않는다.
    from .overnight_check import run_overnight_check

    oc = run_overnight_check(df, force=False)
    print("\n=== overnight 진단 (사전 등록 부속표) ===")
    print(f"풀 t: 나이브 {oc['naive_t']:.2f} → 일 단위 교정 {oc['daily_t']:.2f} "
          f"({oc['daily_n']}일) | volbreak 슬리브 상관 ρ={oc['corr_union']:.2f}"
          + (" — 같은 리스크 그룹 취급" if oc["same_group"] else ""))
    if oc["passers"].empty:
        print("게이트 통과 0건 — 홀드아웃 확인 대상 없음 (등록된 예상 시나리오)")
    else:
        print(f"게이트 통과 {len(oc['passers'])}건 → 홀드아웃 방향 확인:")
        for _, r in oc["holdout"].iterrows():
            print(f"  {r['etf']}: {r['verdict']}")
    print("report:", RESULTS_DIR / "overnight_screening.md")

    # flow 부속 진단 (사전 등록 약속 — 교정 t·trend 중복·홀드아웃 확인)
    from .flow_check import run_flow_check

    fc = run_flow_check(df, force=False)
    print("\n=== flow 진단 (사전 등록 부속표) ===")
    print(f"풀 t: 나이브 {fc['naive_t']:.2f} → 일 단위 교정 {fc['daily_t']:.2f} "
          f"({fc['daily_n']}일) | trend 슬리브 상관 ρ={fc['corr']:.2f}"
          + (" — 같은 리스크 그룹 취급" if fc["same_group"] else ""))
    if fc["passers"].empty:
        print("게이트 통과 0건 — 계열 종결 (변형 재시험 금지)")
    else:
        print(f"게이트 통과 {len(fc['passers'])}건 → 홀드아웃 방향 확인:")
        for _, r in fc["holdout"].iterrows():
            print(f"  {r['etf']}: {r['verdict']}")
    print("report:", RESULTS_DIR / "flow_screening.md")


def cmd_portfolio(force: bool) -> None:
    from .portfolio import run_portfolio

    res = run_portfolio(force)
    for label, st in [("full(1.0)", res["full"]), (f"calibrated({res['e_final']:.2f})",
                                                   res["calibrated"])]:
        print(f"{label:18s} cum {st['cum']:+.1%}  CAGR {st['cagr']:+.2%}  "
              f"MDD {st['mdd']:.1%}  Sharpe {st['sharpe']:.2f}")
    print("entries:", res["diag"])
    print("report:", RESULTS_DIR / "portfolio.md")


def cmd_refine(force: bool) -> None:
    from .refinements import run_refinements

    res = run_refinements(force)
    pct = "{:+.3%}".format
    print("=== regime filter (변동성비>1.5 진입금지) ===")
    print(res["regime"][["name", "strategy", "base_n", "filt_n", "base_mean", "filt_mean",
                         "base_mdd", "filt_mdd", "base_t", "filt_t", "adopted"]]
          .to_string(index=False,
                     formatters={c: pct for c in ["base_mean", "filt_mean"]}
                     | {c: "{:.1%}".format for c in ["base_mdd", "filt_mdd"]}
                     | {c: "{:.2f}".format for c in ["base_t", "filt_t"]}))
    print("\n=== exit rules (trailing -5%) ===")
    print(res["exits"][["name", "strategy", "variant", "n", "mean", "base_mean",
                        "mdd", "base_mdd", "t", "base_t", "adopted"]]
          .to_string(index=False,
                     formatters={c: pct for c in ["mean", "base_mean"]}
                     | {c: "{:.1%}".format for c in ["mdd", "base_mdd"]}
                     | {c: "{:.2f}".format for c in ["t", "base_t"]}))
    print("\nreport:", RESULTS_DIR / "refinements.md")


def _print_rotation(tag: str, res: dict) -> None:
    v = res["verdict"]
    for label, st in [("전체", v["full"]), ("전반", v["first"]), ("후반", v["second"])]:
        print(f"  {label}: N={st['n']}  평균 {st['mean']:+.3%}  "
              f"승률 {st['win_rate']:.1%}  t {st['t_stat']:.2f}")
    print(f"  판정(Stage 1 기준): {'통과' if v['passed'] else '기각'}")
    print(f"  계좌: CAGR {res['stats']['cagr']:+.2%}  MDD {res['stats']['mdd']:.1%}  "
          f"Sharpe {res['stats']['sharpe']:.2f}  "
          f"(KODEX200 B&H: {res['bench']['cagr']:+.2%} / {res['bench']['mdd']:.1%} / "
          f"{res['bench']['sharpe']:.2f})")
    print("  현재 시그널:", ", ".join(res["now_target"]) or "현금")
    print("  report:", RESULTS_DIR / f"{tag}.md")


def cmd_rotation(force: bool) -> None:
    from .rotation import run_rotation, run_rotation2

    print("=== 실험 1: 기존 12종 (2026-08-07 기각 — 재현) ===")
    _print_rotation("rotation", run_rotation(force))
    print("\n=== 실험 2: 확장 유니버스 — 자산군 다변화, 파생형 제외 ===")
    _print_rotation("rotation2", run_rotation2(force))


def cmd_crash(force: bool) -> None:
    from .crash_study import run_crash_study

    res = run_crash_study(force)
    pct = "{:+.1%}".format
    print(f"=== 폭락 사례 (60일 고점 -10% 이탈, ETF별) — {len(res['episodes'])}건 ===")
    print(res["episodes"].to_string(index=False,
                                    formatters={"trough_dd": "{:.1%}".format}))
    print("\n=== 손익 분해 (사례 밖 / 폭락 중 진입 / 보유 중 피격) ===")
    print(res["decomposition"][["name", "strategy", "n", "total", "outside_sum",
                                "entered_in_n", "entered_in_sum",
                                "held_into_n", "held_into_sum", "worst_in_ep"]]
          .to_string(index=False,
                     formatters={c: pct for c in ["total", "outside_sum", "entered_in_sum",
                                                  "held_into_sum", "worst_in_ep"]}))
    print("\n=== MA200 게이트 판정 ===")
    print(res["gate"][["name", "strategy", "base_n", "gate_n", "base_mean", "gate_mean",
                       "base_mdd", "gate_mdd", "base_t", "gate_t", "adopted"]]
          .to_string(index=False,
                     formatters={c: "{:+.3%}".format for c in ["base_mean", "gate_mean"]}
                     | {c: "{:.1%}".format for c in ["base_mdd", "gate_mdd"]}
                     | {c: "{:.2f}".format for c in ["base_t", "gate_t"]}))
    print("\nreport:", RESULTS_DIR / "crash_study.md")


def cmd_brief() -> None:
    from .brief import STATUS_PATH, write_status

    write_status("manual")
    print(f"brief: {STATUS_PATH}")


def cmd_health() -> None:
    from .health import run_health

    run_health()


def cmd_cost() -> None:
    from .etf_costs import run_costs

    run_costs()


def cmd_judge() -> None:
    from .judge import run_judge

    run_judge()


def cmd_distcheck() -> None:
    from .dist_check import run_dist_check

    run_dist_check()


def cmd_fillcheck() -> None:
    from .fill_check import run_fill_check

    run_fill_check()


def cmd_holdout(force: bool) -> None:
    from .holdout import run_holdout

    res = run_holdout(force)
    t, v = res["trials"], res["verdicts"]
    print(f"=== 홀드아웃 검증 {res['start']} ~ {res['end']} (사전 등록 1회 실행) ===")
    print("  ※ 2008 금융위기 미포함 — yfinance 2007~08 데이터 공백 (등록 시 명시)")
    print("\n--- 후보별 (진단용) ---")
    if t.empty:
        print("  대상 없음")
    else:
        print(t[["name", "strategy", "years", "n", "mean", "win", "mdd", "t_stat",
                 "stress_t"]]
              .to_string(index=False,
                         formatters={"mean": "{:+.3%}".format, "win": "{:.1%}".format,
                                     "mdd": "{:.1%}".format, "t_stat": "{:.2f}".format,
                                     "stress_t": "{:.2f}".format}))
    print("\n--- 계열 판정 (사전 등록 규칙) ---")
    for _, r in v.iterrows():
        mean = "-" if not pd.notna(r["mean"]) else f"{r['mean']:+.3%}"
        ts = "-" if not pd.notna(r["t_stat"]) else f"{r['t_stat']:.2f}"
        print(f"  [{r['family']:9s}] {r['pool']:5s} 풀 N={int(r['n']):4d} "
              f"평균 {mean} t={ts} (임계 {r['crit']}) → {r['outcome']}")
    for label, why in res["skipped"]:
        print(f"  (제외) {label}: {why}")
    print("\n판정 기준·후보·파라미터는 이 결과로 바꾸지 않는다 (사후 기준 변경 금지).")
    print("report:", RESULTS_DIR / "holdout.md")


def cmd_mtest(force: bool) -> None:
    from .multiple_testing import run_multiple_testing

    res = run_multiple_testing(force)
    tri, rc = res["trials"], res["rc"]
    print(f"=== 다중검정 보정 (재현 시험 {len(tri)}개 / 선언 누적 탐색 {res['declared']}회) ===")
    print(f"  [1] 기대 최대 t (엣지 0인데도 나오는 최고 t, 독립 가정): "
          f"{res['emax_declared']:.2f}  ← Stage 1 게이트는 t>=2")
    print(f"  [2] Reality Check p={rc['p_rc']:.3f} | SPA p={rc['p_spa']:.3f} "
          f"(최고: {res['best_label']}, 일간 t={rc['best_t']:.2f})")
    print(f"      상관 반영 유효 독립 시험 수 ≈ {res['n_eff']:.0f}/{len(tri)}개 "
          f"(독립성 {res['indep_ratio']:.0%}) → 보정 기대 최대 t {res['emax_adjusted']:.2f}")
    print("  [3] 후보별 DSR (선택편의+비정규성 반영, >=0.95 유의):")
    for _, r in res["dsr"].iterrows():
        mark = "통과" if r["dsr"] >= 0.95 else "미달"
        print(f"      {r['label']:34s} N={int(r['n']):5d}  DSR={r['dsr']:.3f}  {mark}")
    raw2 = int((tri["t_stat"] >= 2).sum())
    print(f"  [4] 보정 후 생존: BH-FDR {int(tri['bh_pass'].sum())}개 / "
          f"Holm {int(tri['holm_pass'].sum())}개 (무보정 t>=2 는 {raw2}개)")
    print("\n측정 전용 — 판정 기준(etf_paper.judgment)은 동결 유지, 이 결과로 바꾸지 않는다.")
    print("report:", RESULTS_DIR / "multiple_testing.md")


def cmd_sleeves(force: bool) -> None:
    from .sleeve_report import run_sleeves

    res = run_sleeves(force)
    pos = res["positions"]
    print(f"=== 통합 슬리브 뷰 (측정 전용) — 보유 {len(pos)}건 ===")
    if not res["dup"].empty:
        for name in sorted(res["dup"]["name"].unique()):
            strats = ", ".join(res["dup"][res["dup"]["name"] == name]["strategy"])
            print(f"  ⚠ 동일 ETF 중복 보유: {name} ({strats}) — 명목 노출 배증")
    if not res["corr"].empty:
        print("  슬리브 상관 (비활동=0):")
        print(res["corr"].round(2).to_string())


def cmd_xmarket(force: bool) -> None:
    from .crossmarket import run_crossmarket, run_crossmarket_volbreak

    run_crossmarket(force)          # trend 1회 시험 재현 (2026-08-26 완료분)
    run_crossmarket_volbreak(force)  # volbreak 복제 (PREREG_xmarket2)


def cmd_trade(live_mock: bool, liquidate_legacy: bool, auto: bool = False,
              close_window: bool = False) -> None:
    from .executor import run_trade

    run_trade(live_mock=live_mock, liquidate_legacy=liquidate_legacy, auto=auto,
              close_window=close_window)


def cmd_xdata(force: bool) -> None:
    from .xsection_data import backfill, coverage_report

    backfill(force)
    coverage_report()


def cmd_xsection() -> None:
    from .xsection import run_xsection

    run_xsection()


def cmd_report() -> None:
    _, masters = _load_masters()
    stats = pd.concat([run_backtest(m, name) for name, m in masters.items()],
                      ignore_index=True)
    save_stats(stats)
    for name, m in masters.items():
        print("chart:", plot_equity_curves(m, stats, name))
    split_df = split_validation(masters, stats)
    cost_df, yearly_df = _run_robustness(masters, stats, split_df)
    print("report:", write_report(stats, split_df, masters, cost_df, yearly_df))


def main() -> None:
    parser = argparse.ArgumentParser(description="US gap pattern backtest")
    parser.add_argument("step",
                        choices=["download", "verify", "backtest", "robustness", "report",
                                 "paper", "ml", "analysis", "sizing", "minute", "stoploss",
                                 "etf", "portfolio", "refine", "crash", "rotation",
                                 "cost", "judge", "distcheck", "fillcheck", "mtest", "holdout",
                                 "sleeves", "xmarket", "trade", "daily", "brief",
                                 "xdata", "xsection", "health", "all"])
    parser.add_argument("--force", action="store_true", help="ignore cache, re-download")
    parser.add_argument("--preview", action="store_true",
                        help="paper: 다음 거래일 조건 발동 여부 미리보기 (장부 기록 없음)")
    parser.add_argument("--live-mock", action="store_true",
                        help="trade: KIS 모의계좌에 실제 주문 제출 (기본은 dry-run)")
    parser.add_argument("--liquidate-legacy", action="store_true",
                        help="trade: SwingETF 잔여 보유 전량 시장가 청산 (--live-mock 필요)")
    parser.add_argument("--auto", action="store_true",
                        help="trade: cron 경로 — 모드를 config ops.trade_mode에서 읽음")
    parser.add_argument("--close-window", action="store_true",
                        help="trade: 15:20 실행 창 — 잠정 당일 봉으로 종가 매수/청산")
    args = parser.parse_args()

    ensure_dirs()
    commands = {
        "download": lambda: cmd_download(args.force),
        "verify": cmd_verify,
        "backtest": cmd_backtest,
        "robustness": cmd_robustness,
        "paper": lambda: cmd_paper(args.preview, args.force),
        "ml": cmd_ml,
        "analysis": lambda: cmd_analysis(args.force),
        "sizing": cmd_sizing,
        "minute": cmd_minute,
        "stoploss": cmd_stoploss,
        "etf": lambda: cmd_etf(args.force),
        "portfolio": lambda: cmd_portfolio(args.force),
        "refine": lambda: cmd_refine(args.force),
        "crash": lambda: cmd_crash(args.force),
        "rotation": lambda: cmd_rotation(args.force),
        "cost": cmd_cost,
        "judge": cmd_judge,
        "distcheck": cmd_distcheck,
        "fillcheck": cmd_fillcheck,
        "mtest": lambda: cmd_mtest(args.force),
        "holdout": lambda: cmd_holdout(args.force),
        "sleeves": lambda: cmd_sleeves(args.force),
        "xmarket": lambda: cmd_xmarket(args.force),
        "trade": lambda: cmd_trade(args.live_mock, args.liquidate_legacy, args.auto,
                                   args.close_window),
        "daily": cmd_daily,
        "brief": cmd_brief,
        "xdata": lambda: cmd_xdata(args.force),
        "xsection": cmd_xsection,
        "health": cmd_health,
    }
    if args.step == "all":
        for step in ["download", "verify"]:
            commands[step]()
        cmd_report()
    elif args.step == "report":
        cmd_report()
    else:
        commands[args.step]()


if __name__ == "__main__":
    main()
