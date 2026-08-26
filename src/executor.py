"""Phase D 실행기 — 섀도 신호를 KIS 모의계좌 주문으로 번역 (dry-run 기본).

목적 (2026-08-26 계좌 이관으로 개시): 판정까지 기다리는 동안 **모의계좌로
집행 경로를 검증**한다 — 모형 체결 vs 실제(모의) 체결의 비교가 최종
fillcheck 가 된다. 신호·판정 기준 불변, 이 모듈은 소비자일 뿐이다.

범위 (v1 — 정직한 한계 명시):
- **시가 진입 계열(trend 등)만 제출 가능** — 아침 신호를 시장가 매수로 번역
  (수량은 brief.order_plan 과 동일 환산). 청산 신호도 같은 방식(시장가 매도).
- volbreak: KIS OpenAPI 는 조건부(스탑) 주문을 지원하지 않는다 — 계획 표시만.
  실전 구현은 장중 감시 루프 또는 조건주문 지원 채널 필요 (Phase D 후반 과제).
- overnight: 종가 동시호가(15:20~) 시점 실행 필요 — 현행 cron 시간대 밖이라
  계획 표시만. 편입 후보(Semicon)의 라이브화는 15:20 cron 추가가 선행 조건.
- 기본 dry-run: 주문을 내지 않고 계획을 출력 + paper/exec_plan.csv 에 기록
  (타임스탬프 증빙 — preview_signals 와 같은 취지). --live-mock 일 때만 제출.
- 잔여 보유 보호: SwingETF 이관 잔여 종목은 건드리지 않는다.
  --liquidate-legacy (--live-mock 필요)로만 전량 시장가 청산.
- 같은 날 중복 제출 방지: exec_plan.csv 에 오늘 제출 기록이 있으면 스킵.

실행: python -m src.main trade [--live-mock] [--liquidate-legacy]
"""

from __future__ import annotations

import sys

import pandas as pd

from .config import ROOT_DIR, load_config

EXEC_LOG = ROOT_DIR / "paper" / "exec_plan.csv"
_COLS = ["date", "generated_at", "mode", "action", "code", "name", "strategy",
         "qty", "note", "order_no"]


def _load_log() -> pd.DataFrame:
    if EXEC_LOG.exists():
        return pd.read_csv(EXEC_LOG, dtype={"code": str})
    return pd.DataFrame(columns=_COLS)


def _append_log(rows: list[dict]) -> None:
    if not rows:
        return
    log = pd.concat([_load_log(), pd.DataFrame(rows)], ignore_index=True)
    EXEC_LOG.parent.mkdir(exist_ok=True)
    log.to_csv(EXEC_LOG, index=False, encoding="utf-8-sig")


def build_plan() -> list[dict]:
    """오늘의 실행 계획 — 신호 계산은 기존 공용 경로만 사용 (재구현 금지)."""
    from .brief import order_plan
    from .etf_paper import candidate_states

    cfg = load_config()
    ops = cfg.get("ops", {})
    states = candidate_states(force=False)
    plans = order_plan(states, ops.get("virtual_capital", 10_000_000),
                       ops.get("exposure", 0.5),
                       cfg.get("portfolio", {}).get("leverage", {}))
    out = []
    for p in plans:
        out.append({"action": "buy_open", "code": p["code"], "name": p["name"],
                    "strategy": p["strategy"], "qty": p["qty"],
                    "note": f"시가 진입 신호 (비중 {p['weight']:.1%})"})
    for st in states:   # 제출 불가 계열은 계획으로만 기록 (한계의 정직한 표시)
        if st["open_pos"]:
            continue
        if "trigger_inc" in st:
            out.append({"action": "plan_only", "code": str(st["cand"]["code"]),
                        "name": st["cand"]["name"], "strategy": st["cand"]["strategy"],
                        "qty": 0, "note": f"스탑매수 시가+{st['trigger_inc']:,.0f}원 "
                                          "(KIS 조건부 주문 미지원 — 표시만)"})
        elif st.get("close_entry"):
            out.append({"action": "plan_only", "code": str(st["cand"]["code"]),
                        "name": st["cand"]["name"], "strategy": st["cand"]["strategy"],
                        "qty": 0, "note": "종가 조건부 매수 (15:20 실행 창 밖 — 표시만)"})
    return out


def run_trade(live_mock: bool = False, liquidate_legacy: bool = False,
              auto: bool = False) -> None:
    if auto:   # 아침 cron 경로 — 모드는 config ops.trade_mode (운영 편의 값)
        live_mock = load_config().get("ops", {}).get("trade_mode") == "live_mock"
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "live_mock" if live_mock else "dry_run"

    if liquidate_legacy:
        if not live_mock:
            raise SystemExit("--liquidate-legacy 는 --live-mock 과 함께만 (실제 제출 필요)")
        _liquidate_legacy(today, now)
        return

    plan = build_plan()
    print(f"=== Phase D 실행기 ({mode}) — 오늘 계획 {len(plan)}건 ===")
    log = _load_log()
    submitted_today = set()
    if len(log):
        sub = log[(log["date"] == today) & (log["mode"] == "live_mock")]
        submitted_today = set(zip(sub["name"], sub["strategy"], strict=True))

    rows = []
    kis = None
    for p in plan:
        line = f"  [{p['name']}] {p['strategy']}: {p['action']} x{p['qty']} — {p['note']}"
        order_no = ""
        if p["action"] == "buy_open" and live_mock:
            if (p["name"], p["strategy"]) in submitted_today:
                line += " (오늘 이미 제출 — 스킵)"
            else:
                import time

                from .kis import KIS
                kis = kis or KIS()
                try:
                    time.sleep(0.6)   # 모의 레이트리밋 초당 2건 방어
                    res = kis.order_cash(p["code"], int(p["qty"]), "buy")
                    order_no = res.get("ODNO", "")
                    line += f" → 제출 odno={order_no}"
                except Exception as e:   # noqa: BLE001 — 주문 실패는 기록하고 계속
                    line += f" → [WARN] 제출 실패: {e}"
        print(line)
        rows.append({"date": today, "generated_at": now, "mode": mode, **p,
                     "order_no": order_no})
    if not plan:
        print("  오늘 실행할 신호 없음")
    _append_log(rows)
    print(f"log: {EXEC_LOG}")
    if not live_mock:
        print("(dry-run — 주문 미제출. 제출은 --live-mock. 실계좌 경로는 코드에 없음)")


def _liquidate_legacy(today: str, now: str) -> None:
    """SwingETF 이관 잔여 보유 청산 — config ops.legacy_codes 목록만 매도.

    잔고 전체를 팔면 실행기가 나중에 연 포지션까지 청산하는 사고가 되므로,
    이관 시점에 동결한 목록으로 대상을 한정한다.
    """
    import time

    from .kis import KIS

    legacy = set(load_config().get("ops", {}).get("legacy_codes", []))
    if not legacy:
        print("legacy_codes 비어 있음 — 청산 대상 없음 (이미 완료)")
        return
    kis = KIS()
    targets = [h for h in kis.balance()["holdings"] if h["code"] in legacy]
    print(f"=== 잔여 보유 청산 — 대상 {len(targets)}종목 (동결 목록 {len(legacy)}건 한정) ===")
    rows = []
    for h in targets:
        time.sleep(0.6)   # 모의 레이트리밋 초당 2건(EGW00201) 방어
        try:
            res = kis.order_cash(h["code"], h["qty"], "sell")
            odno = res.get("ODNO", "")
            print(f"  매도 {h['name']} x{h['qty']} → odno={odno}")
        except Exception as e:   # noqa: BLE001
            odno = ""
            print(f"  [WARN] {h['name']} 매도 실패: {e}", file=sys.stderr)
        rows.append({"date": today, "generated_at": now, "mode": "live_mock",
                     "action": "liquidate_legacy", "code": h["code"], "name": h["name"],
                     "strategy": "-", "qty": h["qty"], "note": "SwingETF 잔여 청산",
                     "order_no": odno})
    _append_log(rows)
    print("완료 — 체결 확인은 다음 `trade` 실행 시 잔고로")
