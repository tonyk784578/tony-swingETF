"""imom — 장중 모멘텀: 첫 30분 수익률 부호 → 15:00 매수·종가 매도.

2026-09-09 사전 등록 (PREREG_imom.md — 실행 전 커밋). Gao, Han, Li & Zhou(2018 JFE)
"market intraday momentum"의 표준형을 지수 ETF 2종(KODEX200/KODEX_Lev)에 적용.

규칙 (config `imom` — 전부 고정):
- r1 = P(09:30)/Close(D-1) - 1.  P(09:30) = 09:25 5분봉 종가 (라벨은 봉 시작 시각).
- r1 > 0 → 15:00 매수 (14:55 봉 종가) → 같은 날 종가 청산. r1 <= 0 → 무거래.
- 분봉(원가격)은 `align_minute_to_daily`로 일봉(분배금 조정가) 기준에 정렬한 뒤
  일봉 종가와 섞는다 — 09-09 데이터 기준 정정의 첫 적용.

Stage 1 = 분봉이 있는 첫날(pilot_start)~freeze, 후보별 게이트. 통과 후보만 Stage 2
섀도: freeze 이후 완결 트레이드를 paper/imom_ledger.csv 에 멱등 append.
룩어헤드: r1은 09:30, 진입가는 15:00, 청산가는 15:30 시점 정보 — 각 시점에
알 수 있는 값만 쓴다 (tests/test_imom.py 가 고정).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import combo_stats
from .config import RESULTS_DIR, ROOT_DIR, load_config
from .data_loader import confirmed_cutoff, load_symbol
from .minute_data import align_minute_to_daily, load_volbreak_minute

LEDGER_COLS = ["date", "code", "name", "r1", "entry_price", "exit_price",
               "gross_ret", "cost", "net_ret"]


def _bar_close(minute: pd.DataFrame, hhmm: str) -> pd.Series:
    idx = pd.DatetimeIndex(minute.index)
    sel = idx.strftime("%H:%M") == hhmm
    s = minute.loc[sel, "Close"].copy()
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s[~s.index.duplicated(keep="last")]


def day_table(daily: pd.DataFrame, minute: pd.DataFrame,
              signal_bar: str = "09:25", entry_bar: str = "14:55") -> pd.DataFrame:
    """날짜별 (prev_close, p930, p1500, close, r1, r13) — 필요한 봉이 다 있는 날만.

    `minute`은 이미 일봉 기준으로 정렬돼 있어야 한다. prev_close 는 일봉 전체
    시계열의 shift(1) — 분봉이 없는 날이 끼어도 진짜 직전 거래일 종가를 쓴다.
    """
    if minute.empty or daily.empty:
        return pd.DataFrame(columns=["prev_close", "p930", "p1500", "close", "r1", "r13"])
    p930 = _bar_close(minute, signal_bar)
    p1500 = _bar_close(minute, entry_bar)
    df = pd.DataFrame({"p930": p930, "p1500": p1500}).dropna()
    df["close"] = daily["Close"].reindex(df.index)
    df["prev_close"] = daily["Close"].shift(1).reindex(df.index)
    df = df.dropna()
    df["r1"] = df["p930"] / df["prev_close"] - 1
    df["r13"] = df["close"] / df["p1500"] - 1     # 마지막 30분 (무조건 — 대조군 재료)
    return df[["prev_close", "p930", "p1500", "close", "r1", "r13"]]


def simulate_imom(table: pd.DataFrame, cost: float) -> pd.DataFrame:
    """r1 > 0 인 날의 트레이드 — 15:00 매수·종가 매도, net = r13 - cost."""
    sig = table[table["r1"] > 0]
    out = pd.DataFrame({
        "date": sig.index, "r1": sig["r1"].to_numpy(),
        "entry_price": sig["p1500"].to_numpy(), "exit_price": sig["close"].to_numpy(),
        "gross_ret": sig["r13"].to_numpy(), "cost": cost,
        "net_ret": sig["r13"].to_numpy() - cost,
    })
    return out.reset_index(drop=True)


def gate_verdict(trades: pd.DataFrame, min_n: int, t_min: float) -> dict:
    st = combo_stats(trades["net_ret"]) if len(trades) else combo_stats(pd.Series(dtype=float))
    half = len(trades) // 2
    first = trades["net_ret"].iloc[:half].mean() if half else np.nan
    second = trades["net_ret"].iloc[half:].mean() if len(trades) > half else np.nan
    passed = (st["n"] >= min_n and st["mean"] > 0 and st["t_stat"] >= t_min
              and first > 0 and second > 0)
    return {**st, "first_half": first, "second_half": second, "passed": bool(passed)}


def _ledger_path():
    p = ROOT_DIR / load_config()["imom"]["ledger"]
    p.parent.mkdir(exist_ok=True)
    return p


def load_imom_ledger(path=None) -> pd.DataFrame:
    path = path or _ledger_path()
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLS)
    return pd.read_csv(path, dtype={"code": str}, parse_dates=["date"])


def append_forward(ledger: pd.DataFrame, trades: pd.DataFrame, code: str, name: str,
                   freeze: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """freeze 이후 날짜의 완결 트레이드를 (code, date) 키로 멱등 append."""
    fwd = trades[trades["date"] > freeze]
    done = set(zip(ledger["code"].astype(str), pd.to_datetime(ledger["date"]), strict=True)) \
        if len(ledger) else set()
    rows = [{"date": t["date"], "code": code, "name": name, "r1": round(t["r1"], 6),
             "entry_price": round(float(t["entry_price"]), 4),
             "exit_price": round(float(t["exit_price"]), 4),
             "gross_ret": round(float(t["gross_ret"]), 6), "cost": t["cost"],
             "net_ret": round(float(t["net_ret"]), 6)}
            for _, t in fwd.iterrows() if (code, t["date"]) not in done]
    if rows:
        new = pd.DataFrame(rows)
        ledger = new if ledger.empty else pd.concat([ledger, new], ignore_index=True)
    return ledger, len(rows)


def run_imom() -> None:
    cfg = load_config()
    icfg = cfg["imom"]
    cost = cfg["etf"]["cost_round_trip"]
    cut = confirmed_cutoff()
    start, freeze = pd.Timestamp(icfg["pilot_start"]), pd.Timestamp(icfg["freeze"])
    gate = icfg["gate"]
    jm = icfg["judgment"]

    print("=== imom 장중 모멘텀 (2026-09-09 사전 등록 — Stage 1 표본 "
          f"{start.date()}~{freeze.date()}, 이후 포워드) ===")
    lines = ["# imom — 장중 모멘텀 (첫 30분 → 마지막 30분) 스크리닝", "",
             f"생성일: {pd.Timestamp.today().date()} | 등록: PREREG_imom.md (2026-09-09) | "
             f"Stage 1 표본 {start.date()}~{freeze.date()} | 비용 왕복 {cost:.1%}", "",
             "규칙: r1 = P(09:30)/전일 종가 - 1 > 0 이면 15:00 매수(14:55 봉 종가) → "
             "종가 청산. 게이트: N>=30, 평균>0, t>=2, 전/후반 양수.", "",
             "| ETF | N(신호일) | 표본일 | 평균 | 승률 | t | 전반 | 후반 | MDD | 게이트 | "
             "대조 (a) 무조건 | 대조 (b) r1<=0 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ledger = load_imom_ledger()
    passed = []
    n_new = 0
    for code, name in icfg["universe"].items():
        code = str(code)
        minute = load_volbreak_minute(code)
        daily = load_symbol(code, "kr")
        daily = daily[daily.index <= cut]
        if minute.empty:
            print(f"  {name:12} 분봉 없음")
            continue
        table = day_table(daily, align_minute_to_daily(minute, daily),
                          icfg["signal_bar"], icfg["entry_bar"])
        table = table[table.index <= cut]
        pilot = table[(table.index >= start) & (table.index <= freeze)]
        trades = simulate_imom(pilot, cost)
        v = gate_verdict(trades, gate["min_n"], gate["t"])
        ctrl_all = combo_stats(pilot["r13"] - cost)
        ctrl_neg = combo_stats(pilot.loc[pilot["r1"] <= 0, "r13"] - cost)
        print(f"  {name:12} N={v['n']:3d}/{len(pilot)}일  평균 {v['mean']:+.3%}  "
              f"승률 {v['win_rate']:.0%}  t={v['t_stat']:+.2f}  전반 {v['first_half']:+.3%} "
              f"후반 {v['second_half']:+.3%}  MDD {v['mdd']:+.1%}  → "
              f"{'통과' if v['passed'] else '미달'}  | 대조 무조건 {ctrl_all['mean']:+.3%}"
              f"(t={ctrl_all['t_stat']:+.2f})  r1<=0 {ctrl_neg['mean']:+.3%}")
        lines.append(f"| {name} | {v['n']} | {len(pilot)} | {v['mean']:+.3%} | "
                     f"{v['win_rate']:.0%} | {v['t_stat']:+.2f} | {v['first_half']:+.3%} | "
                     f"{v['second_half']:+.3%} | {v['mdd']:+.1%} | "
                     f"{'**통과**' if v['passed'] else '미달'} | "
                     f"{ctrl_all['mean']:+.3%} (t={ctrl_all['t_stat']:+.2f}, n={ctrl_all['n']}) | "
                     f"{ctrl_neg['mean']:+.3%} (n={ctrl_neg['n']}) |")
        if v["passed"]:
            passed.append(name)
            all_trades = simulate_imom(table, cost)
            ledger, k = append_forward(ledger, all_trades, code, name, freeze)
            n_new += k

    t_crit = jm["t_single"] if len(passed) <= 1 else jm["t_two"]
    lines += ["", f"- Stage 1 통과: {len(passed)}건 ({', '.join(passed) or '없음'}). "
              "대조 (a)=무조건 15:00 매수 마지막 30분 평균, (b)=r1<=0 인 날 — 신호 가치는 "
              "(a)와의 차이 (등록 §2).",
              "- 표본 전부 2026년 폭락 레짐(가설에 유리할 수 있는 고변동 구간) — 등록 §3.",
              f"- Stage 2 판정: 후보별 완결 {jm['min_n']}건, 평균>0 AND 단측 t>={t_crit} "
              f"({'1후보' if len(passed) <= 1 else '2후보 Bonferroni'}). 그 전 실거래 금지."]
    if passed:
        ledger.to_csv(_ledger_path(), index=False, encoding="utf-8-sig")
        print(f"  [포워드] 장부 {n_new}건 신규 append (총 {len(ledger)}건) — {_ledger_path()}")
        for name in passed:
            sub = ledger[ledger["name"] == name]
            st = combo_stats(sub["net_ret"]) if len(sub) else None
            done = len(sub)
            extra = (f"  평균 {st['mean']:+.3%} t={st['t_stat']:+.2f}" if st and done > 1 else "")
            print(f"  [판정 준비] imom {name}: {done}/{jm['min_n']}건 (단측 t>={t_crit}){extra}")
            lines.append(f"- 포워드 {name}: {done}/{jm['min_n']}건{extra}")
    else:
        print("  Stage 1 통과 후보 없음 — 프로그램 종결 (변형 재시험 금지, 등록 §4)")
    out = RESULTS_DIR / "imom_screening.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out}")
