"""일일 브리핑 — 프로젝트 루트 STATUS.md 생성 (운영 편의 계층, config ops).

아침 프리뷰·저녁 장부 끝에 자동 갱신되고 `brief` 명령으로 수동 갱신도 가능.
신호·판정·레짐 수치는 전부 기존 공용 함수(candidate_states/rotation2_state/
readiness_rows/regime_rows)에서 온다 — 브리핑 전용 재구현 금지.

주문 계획은 판정 전 참고용이다: 포트폴리오 규칙(슬롯 1/3 x 총노출, 점수 중립
0.5 가정, 레버리지는 현금 절반)을 가상 자본에 환산한 것. STATUS.md 는 상태
파일(gitignore)이며 기록은 ledger 와 git 이 담당한다.
"""

from __future__ import annotations

import pandas as pd

from .config import ROOT_DIR, load_config
from .etf_paper import _state_text, candidate_states, load_etf_ledger, rotation2_state
from .health import ALERT_PATH, _notify_desktop, readiness_rows, regime_rows

STATUS_PATH = ROOT_DIR / "STATUS.md"


def order_plan(states: list[dict], capital: float, exposure: float,
               leverage: dict) -> list[dict]:
    """발동 신호 → 주문 계획. 비중 = 노출 x 슬롯(1/3) (승률 점수 중립 0.5 가정),
    레버리지 ETF는 현금을 1/lev 만 써서 동일 노출을 만든다."""
    plans = []
    for st in states:
        if not st["enter_today"] or st["open_pos"]:
            continue
        lev = leverage.get(str(st["cand"]["code"]), 1)
        amount = capital * exposure / 3 / lev
        qty = int(amount // st["last_close"]) if st["last_close"] > 0 else 0
        plans.append({"name": st["cand"]["name"], "strategy": st["cand"]["strategy"],
                      "price_ref": st["last_close"], "weight": exposure / 3 / lev,
                      "amount": amount, "qty": qty})
    return plans


def write_status(mode: str, force: bool = False, notify: bool = False) -> None:
    """STATUS.md 갱신. mode: morning | evening | manual."""
    cfg = load_config()
    ops = cfg.get("ops", {})
    capital = ops.get("virtual_capital", 10_000_000)
    exposure = ops.get("exposure", 0.5)
    lev_map = cfg.get("portfolio", {}).get("leverage", {})

    states = candidate_states(force)
    rot = rotation2_state(force)
    plans = order_plan(states, capital, exposure, lev_map)
    ready = readiness_rows()
    regime = regime_rows()
    now = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M")
    mode_label = {"morning": "아침 프리뷰", "evening": "저녁 장부",
                  "manual": "수동"}.get(mode, mode)

    lines = ["# TonySwingETF 상태 브리핑",
             "",
             f"갱신: {now} KST ({mode_label}) · "
             + ("**ALERT.md 있음 — 먼저 확인!**" if ALERT_PATH.exists() else "경고 없음"),
             ""]

    lines.append("## 오늘 신호 · 주문 계획 (참고용 — 판정 전 실거래 금지)")
    lines.append(f"\n가상 자본 {capital:,.0f}원 · 총노출 {exposure:.2f} "
                 "(config ops — 운영 편의 값)\n")
    if plans:
        lines.append("| ETF | 전략 | 기준가(전일종가) | 비중 | 금액 | 수량 |")
        lines.append("|---|---|---|---|---|---|")
        for p in plans:
            lines.append(f"| {p['name']} | {p['strategy']} | {p['price_ref']:,.0f} "
                         f"| {p['weight']:.1%} | {p['amount']:,.0f} | {p['qty']} |")
    else:
        lines.append("- 오늘 시가 진입 신호 없음")
    if rot and rot["rebalance_today"]:
        lines.append("- **로테이션 월초 리밸런스**: "
                     + ("; ".join(rot["moves"]) or "변경 없음"))

    lines.append("\n## 포지션 (Stage 2 섀도)")
    lines.append("\n| 후보 | 전략 | 상태 |")
    lines.append("|---|---|---|")
    for st in states:
        lines.append(f"| {st['cand']['name']} | {st['cand']['strategy']} "
                     f"| {_state_text(st)} |")
    if rot:
        held = (", ".join(f"{p['name']}({p['unrealized']:+.1%})"
                          for p in rot["open_pos"]) or "전 슬롯 현금")
        lines.append(f"| (로테이션) | rotation2 | 보유 {held} |")

    lines.append("\n## 판정 진행 (도달 시 데스크톱 알림)")
    lines.append("")
    for r in ready:
        if r.get("error"):
            lines.append(f"- {r['label']}")
        else:
            mark = " **← 도달**" if r["ready"] else ""
            lines.append(f"- {r['label']}: **{r['n']}/{r['need']}**{mark}")

    lines.append("\n## 시장 레짐 (60일 고점 대비)")
    lines.append("")
    for r in regime:
        lines.append(f"- {r['name']}: {r['dd']:+.1%}"
                     + (" ← 폭락 구간" if r["crash"] else ""))

    led = load_etf_ledger()
    lines.append("\n## 최근 완결 트레이드 (ETF 섀도 장부)")
    lines.append("")
    if led.empty:
        lines.append("- 아직 없음 (freeze 이후 완결 트레이드부터 기록)")
    else:
        for _, t in led.tail(5).iloc[::-1].iterrows():
            lines.append(f"- {t['entry_date'].date()}→{t['exit_date'].date()} "
                         f"{t['name']} {t['strategy']} {t['net_ret']:+.2%}")

    lines.append("\n---\n갱신 주체: `paper`(저녁)·`paper --preview`(아침)·`brief`(수동). "
                 "이 파일은 상태이며 기록은 `paper/*.csv` + git 이 담당.")
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if notify and cfg.get("ops", {}).get("notify_signals", True):
        msgs = [f"{p['name']} {p['strategy']} 진입 신호" for p in plans]
        if rot and rot["rebalance_today"] and rot["moves"]:
            msgs.append("로테이션 리밸런스: " + "; ".join(rot["moves"]))
        if msgs:
            _notify_desktop("TonySwingETF 오늘 신호", " · ".join(msgs))
