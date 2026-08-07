"""포트폴리오 시뮬레이션 — Stage 2 후보 전체를 한 계좌로 운용할 때.

규칙 (config portfolio):
- 후보별 트레이드는 etf_swing 엔진 그대로. 진입 시점의 점수 = 그 후보의
  직전 완결 트레이드(최근 score_window건) 승률 (표본 < min이면 0.5 중립).
- 같은 exposure group(예: KODEX200/레버리지 = 같은 KOSPI200)은 동시 보유 금지 —
  같은 날 신호가 겹치면 점수 높은 쪽만 진입, 이미 보유 중이면 신규 진입 스킵.
- 진입 비중 = (max_gross/3) x (점수/0.5), 남은 노출 예산/레버리지로 클립.
  레버리지 ETF는 노출 2배 계상. 보유 중 리밸런스 없음, 현금 수익 0.
- MDD -8% 목표: 1차 패스(총노출 1.0)의 MDD로 총노출을 축소 캘리브레이션 후
  2차 패스로 확정. (리스크 조정이며 신호 튜닝 아님)
"""

from __future__ import annotations

import pandas as pd

from .config import RESULTS_DIR, load_config
from .data_loader import confirmed_cutoff
from .etf_swing import iter_candidates, simulate


def _candidate_trades(force: bool = False) -> tuple[list[dict], pd.DatetimeIndex]:
    """후보별 트레이드 + 일별 보유수익 시계열 + 진입시점 점수 + 전체 거래일 캘린더."""
    cfg = load_config()
    pcfg = cfg["portfolio"]
    cost = cfg["etf"]["cost_round_trip"]

    out = []
    calendar: set = set()
    for cand, df, entry, exit_, max_hold, trailing in iter_candidates(
            force, cutoff=confirmed_cutoff()):
        code = str(cand["code"])
        calendar.update(df.index)
        trades = simulate(df, entry, exit_, max_hold, cost, trailing=trailing)
        group = next((g for g, codes in pcfg["exposure_groups"].items() if code in codes),
                     code)
        lev = pcfg["leverage"].get(code, 1)

        wins_hist: list[bool] = []
        for _, t in trades.iterrows():
            recent = wins_hist[-pcfg["score_window"]:]
            score = (sum(recent) / len(recent)
                     if len(recent) >= pcfg["score_min_trades"] else 0.5)
            # 일별 보유수익 (진입일: 종가/시가, 이후: 종가/전일종가, 청산일 비용 차감)
            i0 = df.index.get_loc(t["entry_date"])
            i1 = df.index.get_loc(t["exit_date"])
            rets = {}
            for i in range(i0, i1 + 1):
                r = (df["Close"].iloc[i] / df["Open"].iloc[i] - 1 if i == i0
                     else df["Close"].iloc[i] / df["Close"].iloc[i - 1] - 1)
                if i == i1:
                    r -= cost
                rets[df.index[i]] = r
            out.append({"name": cand["name"], "group": group, "lev": lev,
                        "score": score, "entry": t["entry_date"], "exit": t["exit_date"],
                        "net_ret": t["net_ret"], "daily": rets})
            wins_hist.append(t["net_ret"] > 0)
    return out, pd.DatetimeIndex(sorted(calendar))


def _simulate_portfolio(trades: list[dict], e_max: float,
                        calendar: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """일별 포트폴리오 수익 시뮬레이션. 반환: (daily returns, 진단 카운터).

    calendar는 전체 거래일 — 무포지션 날도 0%로 포함해야 Sharpe 기준이 일관된다.
    """
    entries: dict[pd.Timestamp, list[dict]] = {}
    for t in trades:
        entries.setdefault(t["entry"], []).append(t)

    active: list[dict] = []
    diag = {"taken": 0, "skip_group": 0, "skip_budget": 0}
    daily = {}
    for d in calendar:
        for t in sorted(entries.get(d, []), key=lambda x: -x["score"]):
            if any(a["group"] == t["group"] for a in active):
                diag["skip_group"] += 1
                continue
            budget = e_max - sum(a["cap"] * a["lev"] for a in active)
            cap = min((e_max / 3) * (t["score"] / 0.5), budget / t["lev"])
            if cap <= 0:
                diag["skip_budget"] += 1
                continue
            active.append({**t, "cap": cap})
            diag["taken"] += 1
        daily[d] = sum(a["cap"] * a["daily"].get(d, 0.0) for a in active)
        active = [a for a in active if a["exit"] > d]
    return pd.Series(daily).sort_index(), diag


def _bootstrap_mdd_q(port: pd.Series, bcfg: dict) -> tuple[float, dict]:
    """블록 부트스트랩 MDD 분위수. 반환: (하위 quantile MDD, 분포 요약).

    실현 MDD는 단일 경로의 극값이라 추정 분산이 가장 큰 통계량이다 —
    실측 -8.0%의 부트스트랩 분포는 5%~95% 구간이 [-11.9%, -5.0%]에 이른다.
    점추정 대신 하위 분위수(보수 쪽 꼬리)에 노출을 맞춘다. 블록 샘플링은
    장기 레짐 지속성을 끊으므로 실제 꼬리는 이보다 더 나쁠 수 있다(하한 아님).
    """
    import numpy as np

    r = port.to_numpy()
    block = int(bcfg["block_days"])
    n_blocks = int(np.ceil(len(r) / block))
    rng = np.random.default_rng(int(bcfg["seed"]))
    mdds = np.empty(int(bcfg["draws"]))
    for k in range(len(mdds)):
        idx = rng.integers(0, len(r) - block, n_blocks)
        path = np.concatenate([r[i:i + block] for i in idx])
        eq = np.cumprod(1 + path)
        mdds[k] = float((eq / np.maximum.accumulate(eq) - 1).min())
    q = float(np.quantile(mdds, bcfg["quantile"]))
    dist = {p: float(np.quantile(mdds, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return q, dist


def _stats(port: pd.Series) -> dict:
    equity = (1 + port).cumprod()
    mdd = (equity / equity.cummax() - 1).min()
    years = (port.index[-1] - port.index[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else float("nan")
    ann_sharpe = (port.mean() / port.std() * (252 ** 0.5)) if port.std() > 0 else float("nan")
    return {"cum": equity.iloc[-1] - 1, "cagr": cagr, "mdd": mdd,
            "sharpe": ann_sharpe, "equity": equity}


def run_portfolio(force: bool = False) -> dict:
    pcfg = load_config()["portfolio"]
    trades, calendar = _candidate_trades(force)

    port1, _ = _simulate_portfolio(trades, pcfg["max_gross_exposure"], calendar)
    st1 = _stats(port1)
    # 1단계 — 실현 MDD 기준 반복 캘리브레이션 (2026-08-06 등록 방식, 비교용 유지):
    # 복리 비선형성 때문에 1-pass 스케일링은 목표를 살짝 초과할 수 있어,
    # MDD가 목표 이하로 들어올 때까지 축소(최대 4회)
    e_point = min(pcfg["max_gross_exposure"],
                  pcfg["max_gross_exposure"] * abs(pcfg["target_mdd"]) / abs(st1["mdd"]))
    for _ in range(4):
        port2, diag = _simulate_portfolio(trades, e_point, calendar)
        st2 = _stats(port2)
        if st2["mdd"] >= pcfg["target_mdd"]:
            break
        e_point *= abs(pcfg["target_mdd"]) / abs(st2["mdd"])

    # 2단계 — 부트스트랩 분위수 기준 (2026-08-07 개정, 권고 노출):
    # 실현 경로 하나의 MDD가 아니라 같은 수익률 분포에서 뽑은 경로들의
    # 하위 분위수 MDD를 목표에 맞춘다. 실현 -8.0%는 분포의 ~30퍼센타일에
    # 불과했다 — 점추정 캘리브레이션은 노출을 과대 산정한다.
    bcfg = pcfg["bootstrap"]
    e_final = e_point
    for _ in range(6):
        port3, diag3 = _simulate_portfolio(trades, e_final, calendar)
        st3 = _stats(port3)
        q_mdd, dist = _bootstrap_mdd_q(port3, bcfg)
        if q_mdd >= pcfg["target_mdd"] * 1.001:
            break
        e_final *= abs(pcfg["target_mdd"]) / abs(q_mdd)

    pd.DataFrame({"full_exposure": st1["equity"], "calibrated": st2["equity"],
                  "quantile_calibrated": st3["equity"]}) \
        .to_csv(RESULTS_DIR / "portfolio_equity.csv", encoding="utf-8-sig")
    _plot(st1["equity"], st3["equity"], e_final)
    _write_report(st1, st2, st3, e_point, e_final, q_mdd, dist, diag3, pcfg)
    return {"full": st1, "point": st2, "calibrated": st3,
            "e_point": e_point, "e_final": e_final, "q_mdd": q_mdd, "diag": diag3}


def _plot(eq1: pd.Series, eq2: pd.Series, e_final: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot(eq1.index, eq1, linewidth=1.6, color="#2a78d6",
            label="Full exposure (1.0)")
    ax.plot(eq2.index, eq2, linewidth=1.8, color="#eb6834",
            label=f"MDD-calibrated (exposure {e_final:.2f})")
    ax.set_yscale("log")
    ax.set_title("ETF swing portfolio - cumulative (log)", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "portfolio_equity.png", dpi=150)
    plt.close(fig)


def _write_report(st1: dict, st2: dict, st3: dict, e_point: float, e_final: float,
                  q_mdd: float, dist: dict, diag: dict, pcfg: dict) -> None:
    bcfg = pcfg["bootstrap"]
    dist_line = "  ".join(f"{int(p*100)}%={v:.1%}" for p, v in sorted(dist.items()))
    lines = f"""# ETF 스윙 포트폴리오 시뮬레이션

생성일: {pd.Timestamp.today().date()} | 규칙: config `portfolio` (점수=직전 승률,
동일 그룹 중복 금지, 비중=슬롯x(점수/0.5), 레버리지 2배 계상, 리밸런스 없음)

| 구성 | 총노출 | 누적 | CAGR | MDD | Sharpe(연) |
|---|---|---|---|---|---|
| Full | {pcfg['max_gross_exposure']:.2f} | {st1['cum']:+.1%} | {st1['cagr']:+.2%} | {st1['mdd']:.1%} | {st1['sharpe']:.2f} |
| 실현 MDD 캘리브레이션 (구방식) | {e_point:.2f} | {st2['cum']:+.1%} | {st2['cagr']:+.2%} | {st2['mdd']:.1%} | {st2['sharpe']:.2f} |
| **부트스트랩 q{int(bcfg['quantile']*100)} 캘리브레이션 (권고)** | **{e_final:.2f}** | {st3['cum']:+.1%} | {st3['cagr']:+.2%} | {st3['mdd']:.1%} | {st3['sharpe']:.2f} |

- 목표: 블록 부트스트랩(블록 {bcfg['block_days']}일, {bcfg['draws']}회) MDD
  하위 {int(bcfg['quantile']*100)}퍼센타일 >= {pcfg['target_mdd']:.0%}
  → 권고 노출에서 q{int(bcfg['quantile']*100)} MDD {q_mdd:.1%}
- 권고 노출의 부트스트랩 MDD 분포: {dist_line}
- 실현 MDD는 단일 경로 극값이라 분산이 크다 — 점추정 캘리브레이션(구방식
  {e_point:.2f})은 같은 분포에서 5% 확률로 -12% 급 낙폭을 허용했다.
- 진입 {diag['taken']}건 / 그룹 중복 스킵 {diag['skip_group']}건 / 예산 부족 스킵 {diag['skip_budget']}건
- 주의: 블록 부트스트랩은 장기 레짐 지속성을 끊으므로 실제 꼬리는 이보다 나쁠 수
  있다(하한 아님). 그룹 정의는 실측 상관 |rho|>0.5 기준 — config `exposure_groups`.
"""
    (RESULTS_DIR / "portfolio.md").write_text(lines, encoding="utf-8")
