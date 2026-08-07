"""리서치 분석: 신호 세기(사이징 근거), 손실 연도 진단, 종목 전이 검증.

전이 검증 원칙: 이미 확정한 조건을 **그대로** 새 종목에 적용한다.
새 종목에서 조건을 다시 찾으면 다중검정이 되므로 금지.
"""

from __future__ import annotations

import pandas as pd

from .align import build_master
from .backtest import combo_stats, condition_mask
from .config import RESULTS_DIR, load_config
from .data_loader import load_symbol


def strength_analysis(masters: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """주력 조건 발동일을 NVDA 하락폭 구간으로 분해 — 세기-수익 단조성 확인."""
    cfg = load_config()
    acfg = cfg["analysis"]
    cost = cfg["cost"]["round_trip"]
    stock = cfg["main_stock"]
    master = masters[stock]

    base = condition_mask(master, acfg["strength_condition"])
    net = master["day_ret"] - cost
    edges = sorted(acfg["strength_bins"])  # 오름차순 (예: -10% ~ -2%)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = base & (master["nvda_ret"] > lo) & (master["nvda_ret"] <= hi)
        st = combo_stats(net[mask])
        rows.append({"stock": stock, "nvda_bin": f"{lo:+.0%} ~ {hi:+.0%}", **st})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "signal_strength.csv", index=False, encoding="utf-8-sig")
    return df


def year_diagnosis(masters: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """손실 연도의 주력 조건 발동일이 다른 해와 무엇이 달랐는지 비교."""
    cfg = load_config()
    acfg = cfg["analysis"]
    cost = cfg["cost"]["round_trip"]
    stock = cfg["main_stock"]
    year = acfg["diagnose_year"]
    master = masters[stock]

    mask = condition_mask(master, cfg["paper"]["conditions"][0]["condition"])
    trig = master[mask].copy()
    trig["net_ret"] = trig["day_ret"] - cost
    trig["kospi_vol20"] = (masters[stock]["kospi_prev_ret"].rolling(20).std()
                           .reindex(trig.index))
    trig["group"] = trig.index.year.map(lambda y: str(year) if y == year else "other_years")

    out = (trig.groupby("group")
           .agg(n=("net_ret", "count"), mean_net=("net_ret", "mean"),
                win=("net_ret", lambda s: (s > 0).mean()),
                avg_gap=("gap", "mean"), avg_nvda=("nvda_ret", "mean"),
                avg_kospi_prev=("kospi_prev_ret", "mean"),
                avg_vol20=("kospi_vol20", "mean")))
    out.to_csv(RESULTS_DIR / "year_diagnosis.csv", encoding="utf-8-sig")
    return out


def transfer_validation(data: dict[str, pd.DataFrame], force: bool = False) -> pd.DataFrame:
    """확정 조건을 사전 지정 종목 목록에 그대로 적용 (전 기간 + 전/후반)."""
    cfg = load_config()
    acfg = cfg["analysis"]
    cost = cfg["cost"]["round_trip"]
    cond = acfg["transfer_condition"]
    boundary = pd.Timestamp(cfg["split"]["boundary"])

    rows = []
    for code, name in acfg["transfer_stocks"].items():
        d = dict(data)
        d[name] = load_symbol(str(code), "kr", force)
        master = build_master(d, name)
        net = master["day_ret"] - cost
        mask = condition_mask(master, cond)
        st_full = combo_stats(net[mask])
        st_first = combo_stats(net[mask & (master.index < boundary)])
        st_second = combo_stats(net[mask & (master.index >= boundary)])
        rows.append({
            "stock": name, "code": code, "condition": cond,
            "start": master.index.min().date(),
            "n": st_full["n"], "mean": st_full["mean"], "win": st_full["win_rate"],
            "t_stat": st_full["t_stat"],
            "first_mean": st_first["mean"], "second_mean": st_second["mean"],
            "sign_holds": bool(st_first["n"] and st_second["n"]
                               and st_first["mean"] > 0 and st_second["mean"] > 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "transfer_validation.csv", index=False, encoding="utf-8-sig")
    return df


def write_analysis_report(strength: pd.DataFrame, diag: pd.DataFrame,
                          transfer: pd.DataFrame) -> str:
    cfg = load_config()
    acfg = cfg["analysis"]
    stock = cfg["main_stock"]
    year = acfg["diagnose_year"]

    lines = [f"""# 리서치 분석 리포트

생성일: {pd.Timestamp.today().date()} | 기준 종목: {stock} | 비용: {cfg['cost']['round_trip']:.1%}

## 1. 신호 세기 분석 (사이징 근거)

`{acfg['strength_condition']}` 고정 + NVDA 전일 수익률 구간별 (당일 순수익):

| NVDA 구간 | N | 평균 | 승률 | t-stat |
|---|---|---|---|---|"""]
    for _, r in strength.iterrows():
        lines.append(f"| {r['nvda_bin']} | {r['n']} | {r['mean']:+.3%} "
                     f"| {r['win_rate']:.1%} | {r['t_stat']:.2f} |")

    lines.append(f"""
## 2. 손실 연도 진단 ({year})

주력 조건 발동일의 특성 비교:

| 구분 | N | 평균순수익 | 승률 | 평균갭 | 평균NVDA | 평균KOSPI전일 | KOSPI 20일변동성 |
|---|---|---|---|---|---|---|---|""")
    for g, r in diag.iterrows():
        lines.append(f"| {g} | {int(r['n'])} | {r['mean_net']:+.3%} | {r['win']:.1%} "
                     f"| {r['avg_gap']:+.2%} | {r['avg_nvda']:+.2%} "
                     f"| {r['avg_kospi_prev']:+.2%} | {r['avg_vol20']:.2%} |")

    lines.append(f"""
## 3. 전이 검증 — `{acfg['transfer_condition']}` 를 새 종목에 그대로 적용

| 종목 | N | 평균 | 승률 | t-stat | 전반 평균 | 후반 평균 | 양쪽 양수 |
|---|---|---|---|---|---|---|---|""")
    for _, r in transfer.iterrows():
        lines.append(f"| {r['stock']} ({r['code']}) | {r['n']} | {r['mean']:+.3%} "
                     f"| {r['win']:.1%} | {r['t_stat']:.2f} | {r['first_mean']:+.3%} "
                     f"| {r['second_mean']:+.3%} | {'O' if r['sign_holds'] else 'X'} |")

    lines.append("\n(전이 검증은 사후 확인용 참고치 — 채택 여부는 SKHynix 주력 조건과 별개)")
    path = RESULTS_DIR / "analysis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
