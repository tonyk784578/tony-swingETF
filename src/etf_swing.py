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


def simulate_volbreak(df: pd.DataFrame, k: float, cost: float,
                      return_open: bool = False):
    """변동성 돌파 전용 엔진 — 장중 스탑 매수 + 익일 시가 청산.

    일반 simulate()는 '신호 다음날 시가 진입'이라 장중 트리거 체결을 표현할 수
    없어 전용 엔진을 둔다 (사전 등록된 체결 모형 — config etf.strategies.volbreak).
    - 트리거(D) = 시가(D) + k x (고가(D-1) - 저가(D-1))  ← 개장 전 확정 가능
    - 고가(D) >= 트리거면 max(시가, 트리거)에 체결 (갭 상승 개장이면 시가)
    - 청산 = 다음날 시가. 마지막 봉의 트리거는 다음 봉이 없으므로 미완성 처리.
    """
    trig = df["Open"] + k * (df["High"].shift(1) - df["Low"].shift(1))
    hit = (df["High"] >= trig).fillna(False)
    trades = []
    for i in range(len(df) - 1):
        if hit.iloc[i]:
            fill = max(df["Open"].iloc[i], trig.iloc[i])
            trades.append({
                "entry_date": df.index[i], "exit_date": df.index[i + 1], "hold": 1,
                "net_ret": df["Open"].iloc[i + 1] / fill - 1 - cost,
            })
    result = pd.DataFrame(trades)
    if not return_open:
        return result
    open_pos = None
    if hit.iloc[-1]:
        fill = max(df["Open"].iloc[-1], trig.iloc[-1])
        open_pos = {"entry_date": df.index[-1], "entry_price": fill, "hold": 0,
                    "unrealized": df["Close"].iloc[-1] / fill - 1}
    return result, open_pos


def volbreak_trigger(df: pd.DataFrame, k: float) -> float:
    """다음 거래일의 트리거 증분 — '시가 + 이 값' 이상이면 매수 (프리뷰 표시용)."""
    return float(k * (df["High"].iloc[-1] - df["Low"].iloc[-1]))


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
    if strategy == "pullback":
        # 상승장 눌림목: 장기 추세(MA200) 위에서의 단기 과매도만 매수
        trend_ok = close > close.rolling(p["trend_ma"]).mean()
        return (rsi(close, p["period"]) < p["entry_below"]) & trend_ok
    if strategy == "us_dip":
        return us_ret_mapped.reindex(df.index) <= p["threshold"]
    if strategy == "trend_ride":
        fast = close.rolling(p["trend_fast"]).mean()
        slow = close.rolling(p["trend_slow"]).mean()
        return (close > close.rolling(p["lookback"]).max().shift(1)) & (fast > slow)
    if strategy == "tom":
        # 월초 첫 거래일 = 직전 거래일과 (연,월)이 다름. us_dip처럼 그 자체가
        # 'D 시가 진입' 플래그다 (달력 정보는 D 개장 전에 확정 — shift 불필요)
        per = pd.Series(df.index.to_period("M"), index=df.index)
        flag = per.ne(per.shift(1))
        flag.iloc[0] = False   # 데이터 첫 행은 월초 여부를 판별할 수 없음
        return flag
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
    if strategy in ("rsi2", "pullback"):
        return raw.shift(1), close > close.rolling(p["exit_ma"]).mean(), cfg["max_hold"]
    if strategy == "trend_ride":
        # 상승장 추세 라이더: 전략별 max_hold(60)가 기본 10일 캡을 대체 —
        # 추세 이탈(종가<MA20) 전까지 보유를 연장하는 것이 가설의 핵심
        return raw.shift(1), close < close.rolling(p["exit_ma"]).mean(), p["max_hold"]
    # us_dip / tom: raw 가 곧 진입 플래그, 고정 보유일 청산
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
        if cand["strategy"] == "volbreak":
            # 장중 체결 모형 — 플래그 방식 밖. 소비자는 simulate_volbreak로 분기
            yield cand, df, None, None, None, None
            continue
        us_mapped = map_us_to_kr(us_returns(nasdaq), df.index, "ixic")["ixic"]
        if adopted_exits:
            entry, exit_, max_hold, trailing = candidate_flags(cand, df, us_mapped)
        else:
            entry, exit_, max_hold = build_flags(df, cand["strategy"], us_mapped)
            trailing = None
        yield cand, df, entry, exit_, max_hold, trailing


def _screen_rows(universe: dict, strategies: list[str], force: bool) -> pd.DataFrame:
    """유니버스 x 전략 스크리닝 공용 루프 — 본 스크리닝과 확장 스크리닝이 공유."""
    from .data_loader import confirmed_cutoff

    cfg = load_config()
    ecfg = cfg["etf"]
    cost = ecfg["cost_round_trip"]
    boundary = pd.Timestamp(cfg["split"]["boundary"])
    cut = confirmed_cutoff()

    nasdaq = load_symbol("^IXIC", "us", force)
    rows = []
    for code, name in universe.items():
        df = load_symbol(str(code), "kr", force)
        df = df[df.index <= cut]  # 장중 실행 시 당일 미완성 봉 제외
        us_mapped = map_us_to_kr(us_returns(nasdaq), df.index, "ixic")["ixic"]
        bh = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
        for strat in strategies:
            if strat == "volbreak":   # 장중 체결 모형 — 전용 엔진 (플래그 방식 밖)
                trades = simulate_volbreak(df, ecfg["strategies"]["volbreak"]["k"], cost)
            else:
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
    return (pd.DataFrame(rows)
            .sort_values("t_stat", ascending=False).reset_index(drop=True))


def run_screening(force: bool = False) -> pd.DataFrame:
    ecfg = load_config()["etf"]
    out = _screen_rows(ecfg["universe"], list(ecfg["strategies"]), force)
    out.to_csv(RESULTS_DIR / "etf_screening.csv", index=False, encoding="utf-8-sig")
    _write_report(out)
    return out


def run_ext_screening(force: bool = False) -> pd.DataFrame | None:
    """확장 유니버스 x 생존 계열 스크리닝 (2026-08-07 사전 등록 — 게이트 동일)."""
    ecfg = load_config()["etf"]
    if not ecfg.get("universe_ext"):
        return None
    out = _screen_rows(ecfg["universe_ext"], list(ecfg["ext_strategies"]), force)
    out.to_csv(RESULTS_DIR / "etf_ext_screening.csv", index=False, encoding="utf-8-sig")
    _write_ext_report(out)
    return out


def _screen_report(df: pd.DataFrame, header: str, gate_note: str, stem: str) -> None:
    """스크리닝 결과 md 공용 작성기 — 본/확장 스크리닝이 표 형식을 공유한다."""
    cfg = load_config()["etf"]
    top = df[df["rankable"] & df["sign_holds"] & (df["t_stat"] >= 2)]
    lines = [header,
             f"\n## 통과 후보 (N >= {cfg['min_trades']}, 전/후반 모두 양수, t >= 2{gate_note})",
             "",
             "| ETF | 전략 | N | 평균보유 | 평균 | 승률 | 누적 | MDD | t | 전반 | 후반 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        lines.append(f"| {r['etf']} | {r['strategy']} | {r['n']} | {r['avg_hold']:.1f}일 "
                     f"| {r['mean']:+.3%} | {r['win']:.1%} | {r['cum']:+.1%} "
                     f"| {r['mdd']:.1%} | {r['t_stat']:.2f} "
                     f"| {r['first_mean']:+.3%} | {r['second_mean']:+.3%} |")
    if top.empty:
        lines.append("| (통과 후보 없음) | | | | | | | | | | |")
    lines.append(f"\n전체 표는 `{stem}.csv` (t-stat 내림차순).")
    (RESULTS_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# 전략별 계열 상태 (판정 이력의 요약 — 상세 근거는 CLAUDE.md·results/*.md)
_STRATEGY_STATUS = {
    "breakout":   "Stage 2 섀도 중 (후보 3)",
    "macross":    "Stage 2 섀도 중 (후보 1)",
    "trend_ride": "Stage 2 섀도 중 (후보 3)",
    "rsi2":       "종결 — 무필터 평균회귀 기각 (0/12)",
    "pullback":   "종결 — 필터형 평균회귀도 기각 (0/12)",
    "us_dip":     "종결 — 기각",
    "tom":        "1/12 통과 (KODEX_Gold) — Stage 2 섀도. 주의: 주식 가설인데 금만 통과",
    "volbreak":   "6/12 통과 — Stage 2 섀도 (후보별 Bonferroni 판정, 후반 감쇠 주의)",
}


def write_strategy_compare() -> pd.DataFrame:
    """전략별 최종 성과 비교표 — 스크리닝 CSV(본+확장)와 로테이션 에피소드를
    전략 단위로 묶어 한 표로. 어떤 계열이 살아남았는지 한눈에 보는 용도."""
    frames = []
    for stem in ("etf_screening", "etf_ext_screening"):
        path = RESULTS_DIR / f"{stem}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    allc = pd.concat(frames, ignore_index=True)

    rows = []
    for strat, g in allc.groupby("strategy"):
        gate = g[(g["n"] >= load_config()["etf"]["min_trades"]) & g["sign_holds"]
                 & (g["t_stat"] >= 2)]
        best = g.loc[g["t_stat"].idxmax()]
        rows.append({
            "전략": strat, "시험조합": len(g), "게이트통과": len(gate),
            "통과율": len(gate) / len(g),
            "평균수익(전조합)": g["mean"].mean(),
            "최고후보": f"{best['etf']} (평균 {best['mean']:+.2%}, t={best['t_stat']:.2f})",
            "상태": _STRATEGY_STATUS.get(strat, "-"),
        })
    # 로테이션은 조합 스크리닝이 아니라 단일 실험 — 에피소드로 같은 통계 산출
    for stem, label, status in [
            ("rotation_episodes", "rotation(12종)", "기각 — 유니버스 고상관"),
            ("rotation2_episodes", "rotation2(확장)", "통과 — Stage 2 섀도 중")]:
        path = RESULTS_DIR / f"{stem}.csv"
        if not path.exists():
            continue
        ep = pd.read_csv(path)
        st = combo_stats(ep["net_ret"])
        rows.append({"전략": label, "시험조합": 1,
                     "게이트통과": 1 if "통과" in status else 0,
                     "통과율": 1.0 if "통과" in status else 0.0,
                     "평균수익(전조합)": st["mean"],
                     "최고후보": f"에피소드 {st['n']}건 (평균 {st['mean']:+.2%}, "
                                f"t={st['t_stat']:.2f})",
                     "상태": status})

    out = pd.DataFrame(rows).sort_values(["게이트통과", "평균수익(전조합)"],
                                         ascending=False).reset_index(drop=True)
    lines = [f"""# 전략별 최종 성과 비교

생성일: {pd.Timestamp.today().date()} | 근거: etf_screening.csv + etf_ext_screening.csv
+ rotation 에피소드 | 게이트 = N>=30, 전/후반 모두 양수, t>=2

읽는 법: '게이트통과'가 0이면 그 전략 계열은 이 유니버스에서 검증 실패.
평균수익은 시험한 전 조합의 단순 평균이라 낮게 보이는 게 정상 — 실제 채택은
통과 후보만. **과거 성적표이며, 실전 투입은 포워드 판정(judge) 통과가 전제.**

| 전략 | 시험조합 | 통과 | 통과율 | 평균수익 | 최고 후보 | 상태 |
|---|---|---|---|---|---|---|"""]
    for _, r in out.iterrows():
        lines.append(f"| {r['전략']} | {r['시험조합']} | {r['게이트통과']} "
                     f"| {r['통과율']:.0%} | {r['평균수익(전조합)']:+.2%} "
                     f"| {r['최고후보']} | {r['상태']} |")
    (RESULTS_DIR / "strategy_compare.md").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")
    return out


def _write_report(df: pd.DataFrame) -> None:
    cfg = load_config()["etf"]
    _screen_report(df, f"""# ETF 스윙 스크리닝 (Stage 1)

생성일: {pd.Timestamp.today().date()} | 유니버스 {len(cfg['universe'])}종 x
전략 {len(cfg['strategies'])}계열 = {len(df)}조합 | 비용 왕복 {cfg['cost_round_trip']:.2%}

**주의: 탐색 단계 결과다. 아래 후보는 Stage 2(사전 등록 + 섀도)를 통과해야 채택.**""",
                   "", "etf_screening")


def _write_ext_report(df: pd.DataFrame) -> None:
    cfg = load_config()["etf"]
    _screen_report(df, f"""# 확장 유니버스 스크리닝 (2026-08-07 사전 등록)

생성일: {pd.Timestamp.today().date()} | 신자산 {len(cfg['universe_ext'])}종 x
생존 계열 {len(cfg['ext_strategies'])}전략 = {len(df)}조합 | 종결 계열은 재시험 안 함""",
                   " — 전부 Stage 2 등록", "etf_ext_screening")
