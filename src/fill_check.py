"""volbreak 체결 현실성 검증 — 분봉으로 '트리거 도달 시 실제 체결가'를 측정.

등록된 주의②("스탑 매수 슬리피지는 모형보다 불리")를 데이터로 답하는 장치.
측정 전용 — 신호·판정 불변. 판정 시 해석 자료 (분배금 진단과 같은 지위).

방법: 트리거 도달일마다 5분봉에서 최초 도달 봉을 찾고,
- 모형 체결가 = max(시가, 트리거)  (백테스트 가정)
- 보수 체결가 = 도달 봉의 종가     (주문 반영이 5분 늦는 최악 근사)
두 체결가로 각각 수익률을 재계산해 차이(슬리피지 상한 추정)를 보고한다.
분봉은 60일 한도라 매일 축적 중 (minute 단계) — 표본은 시간이 갈수록 늘어난다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, load_config
from .data_loader import confirmed_cutoff, load_symbol
from .minute_data import load_volbreak_minute


def _day_fills(daily: pd.DataFrame, minute: pd.DataFrame, k: float
               ) -> pd.DataFrame:
    """트리거 도달일별 (모형 체결가, 보수 체결가, 익일 시가) 표."""
    trig = daily["Open"] + k * (daily["High"].shift(1) - daily["Low"].shift(1))
    hit = (daily["High"] >= trig).fillna(False)
    m_days = set(minute.index.normalize()) if len(minute) else set()

    rows = []
    for i in range(len(daily) - 1):
        d = daily.index[i]
        if not hit.iloc[i] or d not in m_days:
            continue
        t = float(trig.iloc[i])
        model_fill = max(float(daily["Open"].iloc[i]), t)
        bars = minute[minute.index.normalize() == d]
        touch = bars[bars["High"] >= t]
        if touch.empty:
            # 일봉은 도달인데 분봉엔 없음 — 데이터 결손 (정직하게 스킵 집계)
            rows.append({"date": d, "trigger": t, "model_fill": model_fill,
                         "cons_fill": np.nan, "next_open": np.nan, "gap_open": False})
            continue
        first = touch.iloc[0]
        gap_open = bool(first["Open"] >= t)   # 봉 시가부터 위 — 갭/급등 통과
        cons_fill = float(first["Close"])      # 5분 지연 최악 근사
        rows.append({"date": d, "trigger": t, "model_fill": model_fill,
                     "cons_fill": cons_fill,
                     "next_open": float(daily["Open"].iloc[i + 1]),
                     "gap_open": gap_open})
    return pd.DataFrame(rows)


def run_fill_check() -> None:
    cfg = load_config()
    k = cfg["etf"]["strategies"]["volbreak"]["k"]
    cost = cfg["etf"]["cost_round_trip"]
    cut = confirmed_cutoff()
    vb = [c for c in cfg["etf_paper"]["candidates"] if c["strategy"] == "volbreak"]

    lines = ["# volbreak 체결 현실성 검증 (분봉 실측)", "",
             f"생성일: {pd.Timestamp.today().date()} | 모형 체결 = max(시가, 트리거) "
             "vs 보수 체결 = 최초 도달 5분봉 종가 (주문 5분 지연 최악 근사)", "",
             "측정 전용 — 판정 시 해석 자료. 분봉 축적에 따라 표본이 늘어난다.", "",
             "| ETF | 표본일 | 갭통과% | 평균수익(모형) | 평균수익(보수) | 슬리피지/건 |",
             "|---|---|---|---|---|---|"]
    print("=== volbreak 체결 검증 (분봉 표본 구간) ===")
    for cand in vb:
        code = str(cand["code"])
        minute = load_volbreak_minute(code)
        if minute.empty:
            print(f"  {cand['name']:18s} 분봉 없음 — `minute` 단계 축적 대기")
            continue
        daily = load_symbol(code, "kr")
        daily = daily[daily.index <= cut]
        f = _day_fills(daily, minute, k)
        f = f.dropna(subset=["cons_fill", "next_open"])
        if f.empty:
            print(f"  {cand['name']:18s} 도달일 표본 없음 (분봉 구간 내)")
            continue
        r_model = f["next_open"] / f["model_fill"] - 1 - cost
        r_cons = f["next_open"] / f["cons_fill"] - 1 - cost
        slip = (f["cons_fill"] / f["model_fill"] - 1)
        gap_pct = f["gap_open"].mean()
        print(f"  {cand['name']:18s} {len(f):3d}일 | 갭통과 {gap_pct:.0%} | "
              f"모형 {r_model.mean():+.3%} → 보수 {r_cons.mean():+.3%} "
              f"(슬리피지 {slip.mean():+.3%}/건)")
        lines.append(f"| {cand['name']} | {len(f)} | {gap_pct:.0%} "
                     f"| {r_model.mean():+.3%} | {r_cons.mean():+.3%} "
                     f"| {slip.mean():+.3%} |")

    lines += ["", "- 갭통과 = 최초 도달 봉이 시가부터 트리거 위 (스탑이 즉시 체결됐을 날)",
              "- 보수 체결은 상한 추정 — 실제(조건부 주문 자동 발동)는 모형과 보수 사이.",
              "- 슬리피지가 트레이드당 엣지(인샘플 +0.10~0.36%) 대비 큰 후보는",
              "  판정 통과해도 실전 투입 시 재검토 대상."]
    out = RESULTS_DIR / "volbreak_fill_check.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out}")
