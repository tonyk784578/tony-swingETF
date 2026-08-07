"""사전 등록 손절 룰 검증 (config.yaml stoploss — 동결).

주력 조건 발동일을 분봉으로 리플레이: 시가 진입 후 장중 저가가
시가*(1+level) 이하로 내려오면 손절가 체결로 가정, 아니면 종가 청산.
- 갭스루(봉 시가가 이미 손절가 아래)면 그 봉 시가 체결로 보수적 처리.
- 발동일 중 분봉이 있는 날이 min_days 미만이면 '표본 부족' — 판정 보류.
"""

from __future__ import annotations

import pandas as pd

from .backtest import condition_mask
from .config import RESULTS_DIR, load_config
from .minute_data import coverage, load_minute


def _replay_day(bars: pd.DataFrame, day_open: float, day_close: float,
                level: float, cost: float) -> float:
    """하루치 분봉 리플레이 → 순수익. level 예: -0.02."""
    stop_price = day_open * (1 + level)
    for _, b in bars.iterrows():
        if b["Open"] <= stop_price:          # 갭스루: 봉 시가로 체결
            return b["Open"] / day_open - 1 - cost
        if b["Low"] <= stop_price:           # 봉 내 터치: 손절가 체결 가정
            return level - cost
    return day_close / day_open - 1 - cost   # 미발동: 종가 청산


def run_stoploss_study(masters: dict[str, pd.DataFrame]) -> dict:
    cfg = load_config()
    scfg = cfg["stoploss"]
    cost = cfg["cost"]["round_trip"]
    stock = cfg["main_stock"]
    master = masters[stock]

    cond = cfg["paper"]["conditions"][0]["condition"]
    trig_days = master.index[condition_mask(master, cond)]

    minute = load_minute()
    have = coverage(minute)
    days = trig_days.intersection(have)
    verdict_ready = len(days) >= scfg["min_days"]

    rows = []
    for d in days:
        bars = minute[minute.index.normalize() == d].sort_index()
        row = master.loc[d]
        base = row["day_ret"] - cost
        entry = {"date": d.date(), "no_stop": base}
        for level in scfg["levels"]:
            entry[f"stop{level:.0%}"] = _replay_day(bars, row["open"], row["close"],
                                                    level, cost)
        rows.append(entry)
    detail = pd.DataFrame(rows)

    lines = [f"""# 손절 룰 검증 (사전 등록 2026-08-06)

조건: `{cond}` | 손절 레벨: {', '.join(f'{lv:.0%}' for lv in scfg['levels'])} (동결)
분봉 확보 발동일: **{len(days)}건** / 판정 기준 {scfg['min_days']}건
상태: **{'판정 가능' if verdict_ready else '표본 부족 — 참고치만, 판정 보류'}**
"""]
    if not detail.empty:
        detail.to_csv(RESULTS_DIR / "stoploss_detail.csv", index=False, encoding="utf-8-sig")
        summary = detail.drop(columns="date").agg(["count", "mean", "min"]).T
        summary["win"] = [(detail[c] > 0).mean() for c in summary.index]
        lines.append("| 전략 | N | 평균 | 최악 | 승률 |")
        lines.append("|---|---|---|---|---|")
        for name, r in summary.iterrows():
            lines.append(f"| {name} | {int(r['count'])} | {r['mean']:+.3%} "
                         f"| {r['min']:+.2%} | {r['win']:.0%} |")
        lines.append("\n일별 상세: `stoploss_detail.csv`")
    else:
        summary = pd.DataFrame()
        lines.append("분봉이 확보된 발동일이 아직 없음 — `minute` 수집이 쌓이면 재실행.")

    (RESULTS_DIR / "stoploss.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"n_days": len(days), "ready": verdict_ready, "summary": summary}
