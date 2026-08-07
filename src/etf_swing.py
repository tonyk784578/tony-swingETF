"""메인 트랙: ETF 스윙 스크리닝 백테스트 (Stage 1).

룩어헤드 규칙:
- entry_flag(D) = True 면 D **시가** 매수. 플래그는 D 개장 전에 알 수 있는
  정보로만 산출한다 (가격 전략은 D-1 종가까지 → shift(1),
  us_dip은 '미국 날짜 < D' 매핑이라 그 자체로 안전).
- exit_flag(D) = True 면 D **종가** 청산 (마감 동시호가 근사 — D 종가 정보로
  판단해 그 종가에 파는 가정). 최소 1일 보유 후부터 청산 가능.
- 최대 보유일 도달 시 그날 종가 청산. 비용은 청산 시 왕복 일괄 차감.

Stage 1은 탐색(스크리닝)이다 — 여기서 나온 상위 후보를 그대로 믿지 말고
Stage 2(사전 등록 + 섀도)로 넘겨 검증한다.
"""

from __future__ import annotations

import pandas as pd

from .align import map_us_to_kr, us_returns
from .backtest import combo_stats
from .config import RESULTS_DIR, load_config
from .data_loader import load_symbol


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def simulate(df: pd.DataFrame, entry_flag: pd.Series, exit_flag: pd.Series,
             max_hold: int, cost: float, return_open: bool = False,
             trailing: float | None = None):
    """단일 포지션 스윙 엔진. 반환: 트레이드 목록 (entry/exit date, hold, net_ret).

    return_open=True 면 (trades, open_position) 튜플 — 마지막 봉 기준 미청산
    포지션 정보(dict) 또는 None. 섀도 상태 표시용.
    trailing=0.05 면 보유 중 최고 종가 대비 -5% 이탈 종가에 청산 (exit_flag와 OR).
    """
    entry_flag = entry_flag.fillna(False).astype(bool)
    exit_flag = exit_flag.fillna(False).astype(bool)
    trades = []
    pos_i = None
    entry_price = None
    peak = None
    for i in range(len(df)):
        if pos_i is None:
            if entry_flag.iloc[i]:
                pos_i, entry_price = i, df["Open"].iloc[i]
                peak = df["Close"].iloc[i]
        else:
            held = i - pos_i
            peak = max(peak, df["Close"].iloc[i])
            trail_hit = (trailing is not None
                         and df["Close"].iloc[i] < peak * (1 - trailing))
            if held >= 1 and (exit_flag.iloc[i] or trail_hit or held >= max_hold):
                trades.append({
                    "entry_date": df.index[pos_i], "exit_date": df.index[i], "hold": held,
                    "net_ret": df["Close"].iloc[i] / entry_price - 1 - cost,
                })
                pos_i = None
    result = pd.DataFrame(trades)
    if not return_open:
        return result
    open_pos = None
    if pos_i is not None:
        open_pos = {"entry_date": df.index[pos_i], "entry_price": entry_price,
                    "hold": len(df) - 1 - pos_i,
                    "unrealized": df["Close"].iloc[-1] / entry_price - 1}
    return result, open_pos


def raw_entry_signal(df: pd.DataFrame, strategy: str, us_ret_mapped: pd.Series) -> pd.Series:
    """전략의 원신호 — 마지막 값이 True면 '다음 개장 시가 진입'.

    build_flags(백테스트)와 preview(아침 프리뷰)가 반드시 이 하나를 공유한다.
    가격 전략은 D 종가 기준 신호(진입은 D+1 시가 → build_flags에서 shift(1)),
    us_dip은 매핑 자체가 'D 이전 미국 봉'이라 그대로 D 진입 플래그다.
    """
    p = load_config()["etf"]["strategies"][strategy]
    close = df["Close"]
    if strategy == "breakout":
        return close > close.rolling(p["lookback"]).max().shift(1)
    if strategy == "macross":
        fast = close.rolling(p["fast"]).mean()
        slow = close.rolling(p["slow"]).mean()
        return (fast > slow) & (fast.shift(1) <= slow.shift(1))
    if strategy == "rsi2":
        return rsi(close, p["period"]) < p["entry_below"]
    if strategy == "us_dip":
        return us_ret_mapped.reindex(df.index) <= p["threshold"]
    if strategy == "trend_ride":
        fast = close.rolling(p["trend_fast"]).mean()
        slow = close.rolling(p["trend_slow"]).mean()
        return (close > close.rolling(p["lookback"]).max().shift(1)) & (fast > slow)
    raise ValueError(strategy)


def build_flags(df: pd.DataFrame, strategy: str,
                us_ret_mapped: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """전략별 (entry_flag, exit_flag, max_hold). 플래그 시점 규칙은 모듈 docstring."""
    cfg = load_config()["etf"]
    p = cfg["strategies"][strategy]
    close = df["Close"]
    raw = raw_entry_signal(df, strategy, us_ret_mapped)

    if strategy == "breakout":
        return raw.shift(1), close < close.rolling(p["exit_ma"]).mean(), cfg["max_hold"]
    if strategy == "macross":
        fast = close.rolling(p["fast"]).mean()
        slow = close.rolling(p["slow"]).mean()
        return raw.shift(1), fast < slow, cfg["max_hold"]
    if strategy == "rsi2":
        return raw.shift(1), close > close.rolling(p["exit_ma"]).mean(), cfg["max_hold"]
    if strategy == "trend_ride":
        # 상승장 추세 라이더: 전략별 max_hold(60)가 기본 10일 캡을 대체 —
        # 추세 이탈(종가<MA20) 전까지 보유를 연장하는 것이 가설의 핵심
        return raw.shift(1), close < close.rolling(p["exit_ma"]).mean(), p["max_hold"]
    # us_dip: raw 가 곧 진입 플래그, 고정 보유일 청산
    return raw, pd.Series(False, index=df.index), p["hold_days"]


def candidate_flags(cand: dict, df: pd.DataFrame, us_ret_mapped: pd.Series):
    """Stage 2 후보의 (entry, exit, max_hold, trailing) — config exit 모드 반영."""
    entry, exit_, max_hold = build_flags(df, cand["strategy"], us_ret_mapped)
    mode = cand.get("exit", "ma")
    ts = load_config()["etf_paper"]["trailing_stop"]
    if mode == "trail_only":
        return entry, pd.Series(False, index=df.index), max_hold, ts
    if mode == "ma_plus_trail":
        return entry, exit_, max_hold, ts
    return entry, exit_, max_hold, None


def iter_candidates(force: bool = False, cutoff: pd.Timestamp | None = None,
                    adopted_exits: bool = True):
    """Stage 2 후보 공용 반복자: (cand, df, entry, exit, max_hold, trailing).

    adopted_exits=False 면 채택된 exit 변형을 무시한 baseline(build_flags) —
    refine 실험 재현용. cutoff 지정 시 확정 봉까지로 절단.
    """
    cfg = load_config()
    nasdaq = load_symbol("^IXIC", "us", force)
    for cand in cfg["etf_paper"]["candidates"]:
        df = load_symbol(str(cand["code"]), "kr", force)
        if cutoff is not None:
            df = df[df.index <= cutoff]
        us_mapped = map_us_to_kr(us_returns(nasdaq), df.index, "ixic")["ixic"]
        if adopted_exits:
            entry, exit_, max_hold, trailing = candidate_flags(cand, df, us_mapped)
        else:
            entry, exit_, max_hold = build_flags(df, cand["strategy"], us_mapped)
            trailing = None
        yield cand, df, entry, exit_, max_hold, trailing


def run_screening(force: bool = False) -> pd.DataFrame:
    from .data_loader import confirmed_cutoff

    cfg = load_config()
    ecfg = cfg["etf"]
    cost = ecfg["cost_round_trip"]
    boundary = pd.Timestamp(cfg["split"]["boundary"])
    cut = confirmed_cutoff()

    nasdaq = load_symbol("^IXIC", "us", force)
    rows = []
    for code, name in ecfg["universe"].items():
        df = load_symbol(str(code), "kr", force)
        df = df[df.index <= cut]  # 장중 실행 시 당일 미완성 봉 제외
        us_mapped = map_us_to_kr(us_returns(nasdaq), df.index, "ixic")["ixic"]
        bh = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
        for strat in ecfg["strategies"]:
            entry, exit_, max_hold = build_flags(df, strat, us_mapped)
            trades = simulate(df, entry, exit_, max_hold, cost)
            if trades.empty:
                continue
            r = trades.set_index("entry_date")["net_ret"]
            st = combo_stats(r)
            st_first = combo_stats(r[r.index < boundary])
            st_second = combo_stats(r[r.index >= boundary])
            rows.append({
                "etf": name, "code": code, "strategy": strat,
                "start": df.index.min().date(), "bh_cum": bh,
                "n": st["n"], "avg_hold": trades["hold"].mean(),
                "mean": st["mean"], "win": st["win_rate"], "cum": st["cum_ret"],
                "mdd": st["mdd"], "t_stat": st["t_stat"],
                "first_mean": st_first["mean"], "second_mean": st_second["mean"],
                "sign_holds": bool(st_first["n"] >= 10 and st_second["n"] >= 10
                                   and st_first["mean"] > 0 and st_second["mean"] > 0),
                "rankable": st["n"] >= ecfg["min_trades"],
            })
    out = (pd.DataFrame(rows)
           .sort_values("t_stat", ascending=False).reset_index(drop=True))
    out.to_csv(RESULTS_DIR / "etf_screening.csv", index=False, encoding="utf-8-sig")
    _write_report(out)
    return out


def _write_report(df: pd.DataFrame) -> None:
    cfg = load_config()["etf"]
    top = df[df["rankable"] & df["sign_holds"] & (df["t_stat"] >= 2)]
    lines = [f"""# ETF 스윙 스크리닝 (Stage 1)

생성일: {pd.Timestamp.today().date()} | 유니버스 {len(cfg['universe'])}종 x
전략 {len(cfg['strategies'])}계열 = {len(df)}조합 | 비용 왕복 {cfg['cost_round_trip']:.2%}

**주의: 탐색 단계 결과다. 아래 후보는 Stage 2(사전 등록 + 섀도)를 통과해야 채택.**

## 통과 후보 (N >= {cfg['min_trades']}, 전/후반 모두 양수, t >= 2)

| ETF | 전략 | N | 평균보유 | 평균 | 승률 | 누적 | MDD | t | 전반 | 후반 |
|---|---|---|---|---|---|---|---|---|---|---|"""]
    for _, r in top.iterrows():
        lines.append(f"| {r['etf']} | {r['strategy']} | {r['n']} | {r['avg_hold']:.1f}일 "
                     f"| {r['mean']:+.3%} | {r['win']:.1%} | {r['cum']:+.1%} "
                     f"| {r['mdd']:.1%} | {r['t_stat']:.2f} "
                     f"| {r['first_mean']:+.3%} | {r['second_mean']:+.3%} |")
    if top.empty:
        lines.append("| (통과 후보 없음) | | | | | | | | | | |")
    lines.append("\n전체 표는 `etf_screening.csv` (t-stat 내림차순).")
    (RESULTS_DIR / "etf_screening.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
