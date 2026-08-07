"""사전 등록 개선 실험 2건 (config refine — 동결): 레짐 필터, 청산 규칙.

대상: Stage 2 후보 4개. 판정 기준은 config 주석에 동결된 그대로.
설정을 바꿔 재실행하면 다중검정으로 실험 무효.
"""

from __future__ import annotations

import pandas as pd

from .backtest import combo_stats
from .config import RESULTS_DIR, load_config
from .data_loader import confirmed_cutoff
from .etf_swing import iter_candidates, simulate


def _candidates(force: bool = False):
    """baseline 플래그(채택 exit 미적용) — 실험 재현의 기준점."""
    for cand, df, entry, exit_, max_hold, _ in iter_candidates(
            force, cutoff=confirmed_cutoff(), adopted_exits=False):
        yield cand, df, entry, exit_, max_hold


def _split_stats(trades: pd.DataFrame, boundary: pd.Timestamp) -> dict:
    r = trades.set_index("entry_date")["net_ret"] if len(trades) else pd.Series(dtype=float)
    return {"full": combo_stats(r),
            "first": combo_stats(r[r.index < boundary]),
            "second": combo_stats(r[r.index >= boundary])}


def _improved(base: dict, alt: dict, check_mdd: bool, strict: bool = False) -> bool:
    """전/후반 각각 t-stat 유지·개선(strict=True면 엄격 개선) (+옵션: MDD 개선)."""
    for half in ["first", "second"]:
        b, a = base[half]["t_stat"], alt[half]["t_stat"]
        if not (a > b if strict else a >= b):
            return False
        if check_mdd and not (alt[half]["mdd"] > base[half]["mdd"]):
            return False
    return True


def run_regime_filter(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    p = cfg["refine"]["regime_filter"]
    cost = cfg["etf"]["cost_round_trip"]
    boundary = pd.Timestamp(cfg["split"]["boundary"])

    rows = []
    for cand, df, entry, exit_, max_hold in _candidates(force):
        ret = df["Close"].pct_change()
        ratio = (ret.rolling(p["vol_fast"]).std()
                 / ret.rolling(p["vol_slow"]).std())
        # D 진입 판단은 D-1까지 정보. bool로 명시 캐스팅 — object dtype에서
        # `~`가 정수 반전(-1/-2)이 되어 &가 무의미해지는 함정 방지
        blocked = (ratio > p["block_above"]).shift(1).fillna(False).astype(bool)
        entry_b = entry.fillna(False).astype(bool)
        base = _split_stats(simulate(df, entry_b, exit_, max_hold, cost), boundary)
        filt = _split_stats(simulate(df, entry_b & ~blocked, exit_, max_hold, cost),
                            boundary)
        rows.append({"name": cand["name"], "strategy": cand["strategy"],
                     "base_n": base["full"]["n"], "filt_n": filt["full"]["n"],
                     "base_mean": base["full"]["mean"], "filt_mean": filt["full"]["mean"],
                     "base_mdd": base["full"]["mdd"], "filt_mdd": filt["full"]["mdd"],
                     "base_t": base["full"]["t_stat"], "filt_t": filt["full"]["t_stat"],
                     "adopted": _improved(base, filt, check_mdd=True)})
    return pd.DataFrame(rows)


def run_exit_rules(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    ts = cfg["refine"]["exit_rules"]["trailing_stop"]
    cost = cfg["etf"]["cost_round_trip"]
    boundary = pd.Timestamp(cfg["split"]["boundary"])

    rows = []
    for cand, df, entry, exit_, max_hold in _candidates(force):
        no_exit = pd.Series(False, index=df.index)
        variants = {
            "baseline": simulate(df, entry, exit_, max_hold, cost),
            "trail_only": simulate(df, entry, no_exit, max_hold, cost, trailing=ts),
            "ma_plus_trail": simulate(df, entry, exit_, max_hold, cost, trailing=ts),
        }
        stats = {k: _split_stats(v, boundary) for k, v in variants.items()}
        for variant in ["trail_only", "ma_plus_trail"]:
            s = stats[variant]["full"]
            b = stats["baseline"]["full"]
            rows.append({"name": cand["name"], "strategy": cand["strategy"],
                         "variant": variant, "n": s["n"],
                         "mean": s["mean"], "base_mean": b["mean"],
                         "mdd": s["mdd"], "base_mdd": b["mdd"],
                         "t": s["t_stat"], "base_t": b["t_stat"],
                         "adopted": _improved(stats["baseline"], stats[variant],
                                              check_mdd=False, strict=True)})
    return pd.DataFrame(rows)


def run_refinements(force: bool = False) -> dict:
    regime = run_regime_filter(force)
    exits = run_exit_rules(force)
    regime.to_csv(RESULTS_DIR / "regime_filter.csv", index=False, encoding="utf-8-sig")
    exits.to_csv(RESULTS_DIR / "exit_rules.csv", index=False, encoding="utf-8-sig")

    lines = ["""# 개선 실험 리포트 (사전 등록 2026-08-06)

## 1. 레짐 필터 — 변동성비(20/60일) > 1.5 이면 신규 진입 금지

채택 기준: 전/후반 각각 t-stat 유지·개선 AND MDD 개선

| 후보 | N(전→후) | 평균(전→후) | MDD(전→후) | t(전→후) | 채택 |
|---|---|---|---|---|---|"""]
    for _, r in regime.iterrows():
        lines.append(f"| {r['name']} {r['strategy']} | {r['base_n']}→{r['filt_n']} "
                     f"| {r['base_mean']:+.3%}→{r['filt_mean']:+.3%} "
                     f"| {r['base_mdd']:.1%}→{r['filt_mdd']:.1%} "
                     f"| {r['base_t']:.2f}→{r['filt_t']:.2f} "
                     f"| {'채택' if r['adopted'] else '기각'} |")

    lines.append("""
## 2. 청산 규칙 — 트레일링 -5% (단독 / MA 결합)

채택 기준: 전/후반 각각 t-stat 개선 (baseline = MA 이탈 청산)

| 후보 | 변형 | N | 평균 (base) | MDD (base) | t (base) | 채택 |
|---|---|---|---|---|---|---|""")
    for _, r in exits.iterrows():
        lines.append(f"| {r['name']} {r['strategy']} | {r['variant']} | {r['n']} "
                     f"| {r['mean']:+.3%} ({r['base_mean']:+.3%}) "
                     f"| {r['mdd']:.1%} ({r['base_mdd']:.1%}) "
                     f"| {r['t']:.2f} ({r['base_t']:.2f}) "
                     f"| {'채택' if r['adopted'] else '기각'} |")

    n_adopt = int(regime["adopted"].sum() + exits["adopted"].sum())
    lines.append(f"\n**요약: 채택 기준 통과 {n_adopt}건.** 통과 항목이 있으면 config에 반영해"
                 " Stage 2 후보 정의를 갱신하고 freeze를 다시 찍는다 (기존 ledger 보존).")
    (RESULTS_DIR / "refinements.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"regime": regime, "exits": exits}
