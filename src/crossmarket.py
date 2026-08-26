"""교차 시장 복제 — trend 계열 동결 정의를 미국 지수 ETF 2000~2014에 적용.

사전 등록 전문: PREREG_xmarket.md + config `xmarket` (2026-08-26, 결과 관측 전
확정). 홀드아웃과 같은 '동결 규칙의 재현 시험' — 신호는 `raw_entry_signal`/
`build_flags`/`simulate`를 그대로 호출하고 여기서 재구현하지 않는다.

데이터: yfinance auto_adjust=False 원가격 OHLC (미국은 통상 방향 — OHLC가
원가격, Adj Close 가 조정가. 한국 ETF의 반대 사례와 다름을 실측 확인 후 사용).
분배금 낙폭이 가격에 남으므로 전략에 불리한 방향 = 보수적 시험.

실행: python -m src.main xmarket  (1회 판정 — 재실행은 리포트 재생성만, 캐시 결정적)
"""

from __future__ import annotations

import sys

import pandas as pd

from .backtest import combo_stats
from .config import DATA_DIR, RESULTS_DIR, load_config
from .etf_swing import build_flags, simulate
from .holdout import one_sided_t


def load_xmarket_symbol(sym: str, force: bool = False) -> pd.DataFrame:
    cfg = load_config()["xmarket"]
    path = DATA_DIR / f"xmarket_{sym}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    import yfinance as yf

    end = pd.Timestamp(cfg["end"]) + pd.Timedelta(days=1)
    y = yf.Ticker(sym).history(start="1999-01-01", end=end.strftime("%Y-%m-%d"),
                               auto_adjust=False)
    if y.empty:
        return pd.DataFrame()
    y.index = pd.to_datetime(y.index.date)
    df = y[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[(df.index >= pd.Timestamp(cfg["start"])) & (df.index <= pd.Timestamp(cfg["end"]))]
    df.index.name = "Date"
    if not df.empty:
        df.to_parquet(path)
    return df


def run_crossmarket(force: bool = False) -> dict:
    cfg = load_config()
    xcfg = cfg["xmarket"]
    cost = cfg["etf"]["cost_round_trip"]

    rows, all_trades = [], []
    for sym, label in xcfg["symbols"].items():
        df = load_xmarket_symbol(sym, force)
        if df.empty or len(df) < 250:
            rows.append({"sym": sym, "label": label, "strategy": "-", "n": 0,
                         "note": f"데이터 부족 ({len(df)}일)"})
            continue
        empty = pd.Series(dtype=float)
        for strat in xcfg["strategies"]:
            entry, exit_, max_hold = build_flags(df, strat, empty)
            trades = simulate(df, entry, exit_, max_hold, cost)
            if trades.empty:
                continue
            st = combo_stats(trades["net_ret"])
            rows.append({"sym": sym, "label": label, "strategy": strat,
                         "n": st["n"], "mean": st["mean"], "win": st["win_rate"],
                         "cum": st["cum_ret"], "mdd": st["mdd"], "t_stat": st["t_stat"],
                         "avg_hold": trades["hold"].mean()})
            all_trades.append(trades)

    pooled = (pd.concat(all_trades, ignore_index=True)["net_ret"]
              if all_trades else pd.Series(dtype=float))
    t = one_sided_t(pooled)
    mean = float(pooled.mean()) if len(pooled) else float("nan")
    crit = float(xcfg["family_t"])
    if not len(pooled):
        outcome = "표본 없음"
    elif mean > 0 and t >= crit:
        outcome = "통과 — 추세추종 메커니즘이 다른 시장·시대에서 재현 (인샘플 근거 보강)"
    elif mean > 0:
        outcome = "미달 — 판별 불가 (포워드 판정만이 답, 현상 유지)"
    else:
        outcome = "실패 — 재현 안 됨 (포워드 판정 통과 시에도 반증 자료로 병기)"

    res = {"rows": pd.DataFrame(rows), "n": len(pooled), "mean": mean,
           "t": t, "crit": crit, "outcome": outcome,
           "start": xcfg["start"], "end": xcfg["end"]}
    _write_report(res)
    print(f"=== 교차 시장 복제 (trend 계열, {res['start']}~{res['end']}) ===")
    print(f"  계열 풀: N={res['n']}, 평균 {mean:+.3%}, 단측 t={t:.2f} (임계 {crit})")
    print(f"  ▶ {outcome}")
    return res


def _write_report(res: dict) -> None:
    r = res["rows"]
    lines = [f"""# 교차 시장 복제 — trend 계열을 미국 지수 ETF 2000~2014에 적용 (1회)

생성일: {pd.Timestamp.today().date()} | 구간 **{res['start']} ~ {res['end']}**
(닷컴 붕괴 + 2008 금융위기 포함, 한국 인샘플 2015+와 시대 불겹침)
| 사전 등록: PREREG_xmarket.md (결과 관측 전 확정)

## 계열 판정 (등록 규칙 — 1회)

| 풀 | N | 평균 | 단측 t | 임계 | 결론 |
|---|---|---|---|---|---|
| trend 12조합 트레이드 | {res['n']} | {res['mean']:+.3%} | **{res['t']:.2f}** \
| {res['crit']} | {res['outcome']} |

## 조합별 상세 (진단용 — 선별에 쓰지 않는다)

| ETF | 전략 | N | 평균보유 | 평균 | 승률 | 누적 | MDD | t |
|---|---|---|---|---|---|---|---|---|"""]
    for _, x in r.iterrows():
        if x.get("n", 0) == 0:
            lines.append(f"| {x['sym']} | - | 0 | | | | | | {x.get('note','')} |")
            continue
        lines.append(f"| {x['sym']} | {x['strategy']} | {x['n']} | {x['avg_hold']:.1f}일 "
                     f"| {x['mean']:+.3%} | {x['win']:.1%} | {x['cum']:+.1%} "
                     f"| {x['mdd']:.1%} | {x['t_stat']:.2f} |")
    lines.append("""
- 데이터는 미조정 원가격 — 배당 낙폭이 전략에 불리한 방향(보수적).
- exit 는 기본형(ma) — 한국 후보에서 채택된 trail 변형은 선택의 산물이라 배제.
- 비용 왕복 0.1% 유지 — 미국 실제(1~2bp 스프레드)보다 크게 보수적.
- **어느 결과든 config 판정 기준·후보·파라미터는 불변** — 이 시험이 바꾸는 것은
  기록되는 결론뿐이며 그 결론은 실행 전에 등록돼 있었다 (홀드아웃 관례).
- 동결 규칙 재현 시험이므로 declared_trials(160)에 가산하지 않는다. 1회 실행.""")
    out = RESULTS_DIR / "xmarket.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out}", file=sys.stderr)
