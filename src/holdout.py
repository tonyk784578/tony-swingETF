"""홀드아웃 검증 — 동결된 후보 규칙을 '개발에 안 쓴 과거 구간'에 1회 적용.

사전 등록 전문은 config `holdout` (2026-08-14, 결과 관측 전 확정). 요약:
- 가설: etf_paper.candidates 의 동결 규칙을 2009-04-17~2014-12-31 에 **그대로**
  적용하면 인샘플과 같은 방향(평균>0)이 재현된다.
- 이 구간은 인샘플(2015-01-01~)과 겹치지 않고, 개발·튜닝에 한 번도 쓰이지 않았다.
- 2008 금융위기는 **포함되지 않는다** — yfinance 한국 ETF 데이터가 2007~2008 에
  사실상 비어 있다(2007년 10일, 2008년 2일). 대신 2011 유럽 재정위기와
  2012~2014 장기 박스권이 들어간다. 박스권은 추세추종에 불리한 레짐이다.

**이 파일이 지키는 규칙 (어기면 홀드아웃이 아니라 2차 탐색이 된다)**
1. 신호·청산 계산은 기존 동결 엔진(`candidate_flags`/`simulate`/`simulate_volbreak`)을
   그대로 호출한다. 여기서 규칙을 재구현하지 않는다.
2. 후보를 성적으로 고르지 않는다 — 구간 데이터가 있는 등록 후보 전수.
3. 판정 임계·계열 구분은 config 에서만 읽는다.
4. 1회 실행. 구간·비용·파라미터를 바꿔 재실행하지 않는다.

데이터 기준: yfinance 한국 ETF는 OHLC 가 배당조정, Adj Close 가 원가격이다
(통상과 반대 — 실측 확인). 인샘플 FDR 미조정가와 기준을 맞추려고 역산한다:
원가격 OHLC = OHLC x (Adj Close / Close). 2015+ 겹침 2,820일에서 일간수익률
중앙오차 0.2~1.0bp 로 일치 확인. 캐시는 라이브와 분리(data/holdout_{code}.parquet).

실행: python -m src.main holdout
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .backtest import combo_stats
from .config import DATA_DIR, RESULTS_DIR, load_config
from .etf_swing import candidate_flags, simulate, simulate_overnight, simulate_volbreak

# us_dip / ewy_* 는 미국 일봉 매핑이 필요하다. 등록 후보에는 없지만, 나중에
# 그런 후보가 추가된 채로 이 단계를 돌리면 조용히 빈 신호가 되므로 막는다.
_NEEDS_US = ("us_dip", "ewy_")


def _holdout_path(code: str):
    return DATA_DIR / f"holdout_{code}.parquet"


def reconstruct_raw(y: pd.DataFrame) -> pd.DataFrame:
    """yfinance 프레임 → 원가격(미조정) OHLC.

    yfinance 한국 ETF는 OHLC 가 배당조정, Adj Close 가 원가격이다(통상과 반대).
    조정계수는 날짜별로 OHLC 전체에 동일하게 걸려 있으므로 Adj Close/Close 를
    곱하면 네 값이 모두 원가격으로 돌아온다 — 일중 비율(수익률)은 불변이고
    수준만 인샘플(FDR 미조정가) 기준에 맞춰진다.
    """
    factor = y["Adj Close"] / y["Close"]
    df = y[["Open", "High", "Low", "Close"]].mul(factor, axis=0)
    if "Volume" in y.columns:
        df["Volume"] = y["Volume"]
    return df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)].dropna()


def load_holdout_symbol(code: str, force: bool = False) -> pd.DataFrame:
    """홀드아웃 구간 일봉 (원가격 기준 복원). 라이브 캐시와 별도 파일."""
    cfg = load_config()["holdout"]
    path = _holdout_path(code)
    if path.exists() and not force:
        return pd.read_parquet(path)

    import yfinance as yf

    end = pd.Timestamp(cfg["end"]) + pd.Timedelta(days=1)   # yfinance end 는 배타적
    y = yf.Ticker(f"{code}.KS").history(start="2000-01-01", end=end.strftime("%Y-%m-%d"),
                                        auto_adjust=False)
    if y.empty:
        return pd.DataFrame()
    y.index = pd.to_datetime(y.index.date)
    df = reconstruct_raw(y)
    df = df[(df.index >= pd.Timestamp(cfg["start"])) & (df.index <= pd.Timestamp(cfg["end"]))]
    df.index.name = "Date"
    if not df.empty:
        df.to_parquet(path)
    return df


def _judgment_costs() -> dict[str, float]:
    """판정용 동결 비용표 — judge 와 같은 표를 쓴다 (관례 일치)."""
    from .judge import _load_judgment_costs

    return _load_judgment_costs(load_config())


def candidate_trades(cand: dict, df: pd.DataFrame, cost: float) -> pd.DataFrame:
    """동결 규칙 그대로의 트레이드 목록 — 규칙 재구현 없음."""
    strategy = cand["strategy"]
    if any(strategy.startswith(p) or strategy == p for p in _NEEDS_US):
        raise ValueError(f"{strategy}: 홀드아웃은 미국 일봉 매핑 미지원 — 대상에서 제외할 것")
    if strategy == "volbreak":
        k = load_config()["etf"]["strategies"]["volbreak"]["k"]
        return simulate_volbreak(df, k, cost)
    if strategy == "overnight":
        p = load_config()["etf"]["strategies"]["overnight"]
        return simulate_overnight(df, p["entry_above"], cost)
    # 가격 전략은 us 매핑을 참조하지 않는다 (_NEEDS_US 가드로 보장)
    empty = pd.Series(np.nan, index=df.index)
    entry, exit_, max_hold, trailing = candidate_flags(cand, df, empty)
    return simulate(df, entry, exit_, max_hold, cost, trailing=trailing)


def one_sided_t(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    if len(r) < 2 or not r.std(ddof=1) > 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(len(r)))


def _pool(trades: list[pd.DataFrame], how: str) -> pd.Series:
    """계열 풀 구성. trade=전 트레이드, daily=같은 청산일을 평균해 관측치 1개.

    daily 가 필요한 이유는 volbreak 처럼 여러 후보가 같은 날 동시 트리거되는
    구조 때문 — 트레이드를 그대로 풀면 상관 때문에 t 가 부풀어 오른다.
    """
    if not trades:
        return pd.Series(dtype=float)
    allt = pd.concat(trades, ignore_index=True)
    if allt.empty:
        return pd.Series(dtype=float)
    if how == "daily":
        return allt.groupby("exit_date")["net_ret"].mean()
    return allt["net_ret"].reset_index(drop=True)


def run_holdout(force: bool = False) -> dict:
    cfg = load_config()
    hcfg = cfg["holdout"]
    cost_map = _judgment_costs()
    flat = cfg["etf"]["cost_round_trip"]
    stress = float(hcfg["cost_stress"])

    rows, by_family, skipped = [], {}, []
    for cand in cfg["etf_paper"]["candidates"]:
        code, name = str(cand["code"]), cand["name"]
        strategy = cand["strategy"]
        fam = cand.get("family", "trend")
        if fam not in hcfg["families"]:
            # 홀드아웃 등록(08-14) 이후 편입된 계열 — 여기 넣으면 동결된 1회
            # 시험의 재현이 깨진다. 신규 계열의 홀드아웃 확인은 자기 등록 절차가
            # 담당한다 (예: overnight — overnight_check, PREREG_overnight.md)
            skipped.append((f"{name} {strategy}", "홀드아웃 등록 후 편입 계열 — 대상 외"))
            continue
        df = load_holdout_symbol(code, force)
        if df.empty or len(df) < 120:      # 워밍업(MA60·20일 신고가)도 못 채우는 표본
            skipped.append((f"{name} {strategy}", f"구간 데이터 {len(df)}일"))
            continue
        cost = cost_map.get(name, flat)
        trades = candidate_trades(cand, df, cost)
        if trades.empty:
            skipped.append((f"{name} {strategy}", "구간 내 트레이드 0건"))
            continue
        st = combo_stats(trades["net_ret"])
        # 비용 2배 민감도 — 2026 실측 비용을 2009~14 에 쓰는 것이 유리한 방향이라
        stressed = trades["net_ret"] - cost * (stress - 1)
        rows.append({
            "name": name, "strategy": strategy, "family": fam,
            "start": df.index.min().date(), "end": df.index.max().date(),
            "years": round(len(df) / 247, 1), "cost": cost,
            "n": st["n"], "mean": st["mean"], "win": st["win_rate"],
            "cum": st["cum_ret"], "mdd": st["mdd"], "t_stat": st["t_stat"],
            "stress_mean": stressed.mean(), "stress_t": one_sided_t(stressed),
        })
        by_family.setdefault(fam, []).append(trades)

    trials = pd.DataFrame(rows)
    verdicts = []
    for fam, rule in hcfg["families"].items():
        pooled = _pool(by_family.get(fam, []), rule["pool"])
        t = one_sided_t(pooled)
        mean = float(pooled.mean()) if len(pooled) else float("nan")
        if not len(pooled):
            outcome = "표본 없음"
        elif mean > 0 and np.isfinite(t) and t >= rule["t"]:
            outcome = "통과 — 다른 레짐에서도 재현됨 (인샘플 근거 보강)"
        elif mean > 0:
            outcome = "미달 — 판별 불가 (표본/레짐 한계)"
        else:
            outcome = "실패 — 다른 레짐에서 재현되지 않음 (실거래 결정 시 반증 자료로 병기)"
        verdicts.append({"family": fam, "pool": rule["pool"], "n": len(pooled),
                         "mean": mean, "t_stat": t, "crit": rule["t"],
                         "outcome": outcome})

    res = {"trials": trials, "verdicts": pd.DataFrame(verdicts), "skipped": skipped,
           "start": hcfg["start"], "end": hcfg["end"], "stress": stress}
    if not trials.empty:
        trials.to_csv(RESULTS_DIR / "holdout.csv", index=False, encoding="utf-8-sig")
    _write_report(res)
    return res


def _write_report(res: dict) -> None:
    t, v = res["trials"], res["verdicts"]
    lines = [f"""# 홀드아웃 검증 — 동결 규칙을 안 본 과거 구간에 적용 (1회)

생성일: {pd.Timestamp.today().date()} | 구간 **{res['start']} ~ {res['end']}**
| 사전 등록: config `holdout` (2026-08-14, 결과 관측 전 확정)

인샘플(2015-01-01~)과 **겹치지 않고 개발·튜닝에 한 번도 쓰이지 않은** 구간에
동결된 후보 규칙을 그대로 적용했다. 후보를 성적으로 고르지 않았다 — 구간
데이터가 있는 등록 후보 전수.

**2008 금융위기는 포함되지 않는다.** yfinance 한국 ETF 일봉이 2007~2008 에
사실상 비어 있어(2007년 10일, 2008년 2일) 연속 데이터가 2009-04-17 부터다.
대신 2011 유럽 재정위기와 2012~2014 장기 박스권이 들어간다 — 박스권은
추세추종에 구조적으로 불리하므로 이 시험은 관대하지 않다.

## 계열 판정 (사전 등록 규칙 — 계열별 1회)

| 계열 | 풀 | 관측수 | 평균 | 단측 t | 임계 | 결론 |
|---|---|---|---|---|---|---|"""]
    for _, r in v.iterrows():
        mean = "-" if not np.isfinite(r["mean"]) else f"{r['mean']:+.3%}"
        ts = "-" if not np.isfinite(r["t_stat"]) else f"**{r['t_stat']:.2f}**"
        lines.append(f"| {r['family']} | {r['pool']} | {int(r['n'])} | {mean} | {ts} "
                     f"| {r['crit']} | {r['outcome']} |")

    lines.append(f"""
volbreak 만 '일 단위' 풀인 이유: 후보들이 같은 날 동시 트리거되는 고상관 구조라
트레이드를 그대로 풀면 t 가 부풀기 때문(등록 시 나이브 t=8.75 → 일 단위 2.38).
같은 청산일의 트레이드를 평균해 관측치 1개로 만든 뒤 검정했다.

## 후보별 상세 (진단용 — 채택 근거 아님)

| 후보 | 전략 | 구간 | 년 | N | 평균 | 승률 | 누적 | MDD | t | 비용x{res['stress']:.0f} 평균 | 비용x{res['stress']:.0f} t |
|---|---|---|---|---|---|---|---|---|---|---|---|""")
    for _, r in t.iterrows():
        lines.append(f"| {r['name']} | {r['strategy']} | {r['start']}~{r['end']} "
                     f"| {r['years']} | {int(r['n'])} | {r['mean']:+.3%} | {r['win']:.1%} "
                     f"| {r['cum']:+.1%} | {r['mdd']:.1%} | {r['t_stat']:.2f} "
                     f"| {r['stress_mean']:+.3%} | {r['stress_t']:.2f} |")
    if t.empty:
        lines.append("| (대상 없음) | | | | | | | | | | | |")

    if res["skipped"]:
        lines.append("\n### 제외된 후보 (사전 등록대로 — 성적과 무관)\n")
        for label, why in res["skipped"]:
            lines.append(f"- {label}: {why}")

    lines.append(f"""
## 해석 규칙 (사전 등록 — 결과를 보고 만든 것이 아님)

- 통과 → "다른 레짐에서도 재현됨 — 인샘플 근거 보강"
- 미달(평균>0, t 미달) → "판별 불가 — 표본/레짐 한계"
- 실패(평균<=0) → "인샘플 성적이 다른 레짐에서 재현되지 않음". 포워드 판정을
  통과하더라도 실거래 편입 결정에 반증 자료로 병기한다.

**어느 결과든 config 의 판정 기준·후보 목록·파라미터는 바꾸지 않는다.** 동결된
포워드 판정 절차를 새 시험 결과로 고치면 그게 사후 기준 변경이다. 이 시험이
바꾸는 것은 기록되는 결론뿐이고, 그 결론은 실행 전에 위와 같이 못 박혀 있었다.

비용은 판정용 동결 실측표(results/etf_costs.csv)를 썼다. 2026 실측치를
2009~2014 에 적용하는 것이라 당시 실제 비용보다 낮을 수 있어 **전략에 유리한
방향**이다 — 그래서 비용 {res['stress']:.0f}배 민감도를 함께 실었다.

이 시험은 계열 검정 {len(v)}건이다. 인샘플 전략 탐색이 아니라 동결 규칙의 재현
시험이므로 `multiple_testing.declared_trials`(144)에는 더하지 않는다.
**1회 실행이며 구간·비용·후보를 바꿔 재실행하지 않는다.**
""")
    out = RESULTS_DIR / "holdout.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out}", file=sys.stderr)
