"""Phase D 실행기 — 섀도 신호를 KIS 모의계좌 주문으로 번역 (dry-run 기본).

목적 (2026-08-26 계좌 이관으로 개시): 판정까지 기다리는 동안 **모의계좌로
집행 경로를 검증**한다 — 모형 체결 vs 실제(모의) 체결의 비교가 최종
fillcheck 가 된다. 신호·판정 기준 불변, 이 모듈은 소비자일 뿐이다.

범위 (v2 — 2026-09-02 15:20 실행 창 추가):
- **아침 창** (`trade --auto`, 개장 전): 시가 진입 계열(trend/tom) 시장가 매수
  (수량은 brief.order_plan 과 동일 환산) + **1일 회전 계열(volbreak/overnight)
  전일 종가 매수분의 시가 매도** (개장 전 시장가 제출 = 시가 동시호가 참여).
- **15:20 창** (`trade --auto --close-window`, 마감 동시호가 직전): 현재가
  API로 잠정 당일 봉(시가/고가/저가/현재가)을 만들어 **동결 엔진에 그대로
  넣고** 확정 리플레이와의 차이로 진입/청산을 판정한다 — 신호 재구현 금지
  원칙의 집행 계층 적용. volbreak 트리거 도달·overnight 양봉이면 종가 매수,
  swing 청산 신호(MA 이탈·트레일링·보유 한도)면 종가 매도. 시장가 주문이
  마감 동시호가(15:20~15:30)에 들어가 종가 체결을 근사한다.
  종가 경계 오차(15:19 현재가 vs 실제 종가)는 등록된 대로 실측으로 평가.
- 같은 밤 동일 ETF 노출 합산 캡 (config ops.same_code_slot_cap): Semicon
  volbreak+overnight 같은 중복 발동 시 두 번째 슬리브 매수를 생략 —
  섀도 장부(판정)는 양쪽 다 기록하며, 이 캡은 집행 계층에만 있다.
- 휴장일 방어: 시세 API가 직전 거래일 값을 돌려줘도 후속 주문이 '장 운영시간
  아님'으로 거절되어 자기교정된다. 확정 봉이 이미 오늘이면(16시 이후 실행)
  잠정 봉을 만들지 않아 늦은 실행도 무해하다.
- 기본 dry-run: 주문을 내지 않고 계획을 출력 + paper/exec_plan.csv 에 기록
  (타임스탬프 증빙 — preview_signals 와 같은 취지). --live-mock 일 때만 제출.
  (15:20 창은 dry-run 이어도 시세 조회는 한다 — 조회 없이는 계획 자체가 불가)
- 잔여 보유 보호: SwingETF 이관 잔여 종목은 건드리지 않는다.
  --liquidate-legacy (--live-mock 필요)로만 전량 시장가 청산.
- 같은 날 중복 제출 방지: exec_plan.csv 에 오늘 제출 기록이 있으면 스킵.

실행: python -m src.main trade [--live-mock] [--liquidate-legacy] [--close-window]
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


def _slot_qty(code: str, price: float) -> tuple[int, float]:
    """슬롯 사이징 (brief.order_plan 과 동일 환산): (수량, 비중)."""
    cfg = load_config()
    ops = cfg.get("ops", {})
    lev = cfg.get("portfolio", {}).get("leverage", {}).get(str(code), 1)
    amount = ops.get("virtual_capital", 10_000_000) * ops.get("exposure", 0.5) / 3 / lev
    qty = int(amount // price) if price > 0 else 0
    return qty, ops.get("exposure", 0.5) / 3 / lev


def mock_positions(log: pd.DataFrame | None = None) -> dict[tuple[str, str], dict]:
    """exec_plan 라이프사이클로 본 실행기 보유 — {(code, strategy): {qty, name, date}}.

    live_mock 제출(주문번호 수신) 매수 이후 매도가 없는 (code, strategy) 쌍.
    모의 서버는 주문 상태 조회가 안 되므로(2026-08-26 실측) 잔고 차분과 이
    라이프사이클이 유일한 추적 수단이다. 실행기 밖 보유(legacy 등)는 안 잡힌다
    — 계획 외 보유를 건드리지 않는 보호가 그대로 유지된다.
    """
    log = _load_log() if log is None else log
    pos: dict[tuple[str, str], dict] = {}
    if not len(log):
        return pos
    lm = log[log["mode"] == "live_mock"].copy()
    lm["order_no"] = lm["order_no"].fillna("").astype(str)
    for _, r in lm.iterrows():
        action = str(r["action"])
        key = (str(r["code"]), str(r["strategy"]))
        if action.startswith("buy") and r["order_no"] not in ("", "nan"):
            pos[key] = {"qty": int(float(r["qty"])), "name": r["name"],
                        "date": r["date"]}
        elif action.startswith("sell") and r["order_no"] not in ("", "nan"):
            pos.pop(key, None)
        elif action == "liquidate_legacy":
            pos.pop((str(r["code"]), "-"), None)
    return pos


def _close_action(strategy: str, before: dict | None, trades_after: pd.DataFrame,
                  after: dict | None, today: pd.Timestamp) -> str | None:
    """15:20 결정 규칙 — 확정 리플레이(before) vs 잠정 봉 리플레이(after) 차분.

    반환: 'buy'(종가 매수) / 'sell'(종가 청산) / None.
    - 1일 회전(volbreak/overnight): 잠정 봉에서 오늘 진입이 생기면 매수.
      청산(익일 시가)은 아침 창 담당이라 여기서 매도는 없다.
    - swing(trend/tom): 보유 중인데 잠정 봉에서 그 포지션이 오늘 청산되면 매도.
      진입은 '신호 다음날 시가'라 종가 창에서 생길 수 없다 (엔진 구조가 보증).
    """
    if strategy in ("volbreak", "overnight"):
        if after is not None and after["entry_date"] == today:
            return "buy"
        return None
    if before is not None and len(trades_after):
        last = trades_after.iloc[-1]
        if (last["exit_date"] == today
                and last["entry_date"] == before["entry_date"]):
            return "sell"
    return None


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
                                          "(15:20 창에서 트리거 확인 후 제출)"})
        elif st.get("close_entry"):
            out.append({"action": "plan_only", "code": str(st["cand"]["code"]),
                        "name": st["cand"]["name"], "strategy": st["cand"]["strategy"],
                        "qty": 0, "note": "종가 조건부 매수 (15:20 창에서 양봉 확인 후 제출)"})
    return out


def build_morning_sells() -> list[dict]:
    """1일 회전 계열(volbreak/overnight)의 전일 종가 매수분 → 오늘 시가 매도.

    개장 전 시장가 매도 제출 = 시가 동시호가 참여 — 엔진의 '익일 시가 청산'
    가정과 정합. swing 청산(종가)은 15:20 창 담당이라 여기 없다.
    """
    out = []
    for (code, strategy), p in mock_positions().items():
        if strategy in ("volbreak", "overnight"):
            out.append({"action": "sell_open", "code": code, "name": p["name"],
                        "strategy": strategy, "qty": p["qty"],
                        "note": f"1일 회전 청산 — 시가 매도 (진입 {p['date']})"})
    return out


def run_trade(live_mock: bool = False, liquidate_legacy: bool = False,
              auto: bool = False, close_window: bool = False) -> None:
    if close_window:   # 15:20 창 — 별도 경로 (창 밖 가드는 run_close_window 안)
        run_close_window(live_mock=live_mock, auto=auto)
        return
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

    # 매도(전일 1일 회전 청산)를 매수보다 먼저 — 예수금 확보 + 둘 다 시가 참여
    plan = build_morning_sells() + build_plan()
    print(f"=== Phase D 실행기 ({mode}) — 오늘 계획 {len(plan)}건 ===")
    _submit_plan(plan, today, now, mode, live_mock)


def _submit_plan(plan: list[dict], today: str, now: str, mode: str,
                 live_mock: bool) -> None:
    """계획 목록을 출력·제출·기록 — 아침/15:20 창 공용 경로."""
    log = _load_log()
    submitted_today = set()
    if len(log):
        sub = log[(log["date"] == today) & (log["mode"] == "live_mock")]
        submitted_today = set(zip(sub["name"], sub["strategy"], sub["action"],
                                  strict=True))

    rows = []
    kis = None
    for p in plan:
        line = f"  [{p['name']}] {p['strategy']}: {p['action']} x{p['qty']} — {p['note']}"
        order_no = ""
        submittable = p["action"] in ("buy_open", "sell_open", "buy_close",
                                      "sell_close") and p["qty"] > 0
        if submittable and live_mock:
            if (p["name"], p["strategy"], p["action"]) in submitted_today:
                line += " (오늘 이미 제출 — 스킵)"
            else:
                import time

                from .kis import KIS
                kis = kis or KIS()
                side = "buy" if p["action"].startswith("buy") else "sell"
                try:
                    time.sleep(0.6)   # 모의 레이트리밋 초당 2건 방어
                    res = kis.order_cash(p["code"], int(p["qty"]), side)
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


def build_close_plan(today: pd.Timestamp | None = None) -> list[dict]:
    """15:20 실행 창 계획 — 잠정 당일 봉을 동결 엔진에 넣어 진입/청산 차분.

    시세 조회가 필요하므로 dry-run 이어도 KIS 를 쓴다 (조회 전용 — 주문 없음).
    종목별 독립 실패 처리 (minute 수집 관례): 시세 실패 종목은 건너뛰고 기록.
    """
    import time

    from .data_loader import confirmed_cutoff
    from .etf_swing import (
        candidate_flags,
        iter_candidates,
        simulate,
        simulate_overnight,
        simulate_volbreak,
    )
    from .kis import KIS

    cfg = load_config()
    ecfg = cfg["etf"]
    cost = ecfg["cost_round_trip"]
    today = pd.Timestamp.today().normalize() if today is None else today
    kis = KIS()

    # 오늘 밤 보유 예정 코드 (합산 캡 판정용): 현재 실행기 보유 + 이 창의 매수
    cap = int(cfg.get("ops", {}).get("same_code_slot_cap", 1))
    night_codes: dict[str, int] = {}
    for (code, _s), _p in mock_positions().items():
        night_codes[code] = night_codes.get(code, 0) + 1

    quotes: dict[str, dict] = {}
    out = []
    for cand, df, entry, exit_, max_hold, trailing in iter_candidates(
            cutoff=confirmed_cutoff()):
        code, strategy = str(cand["code"]), cand["strategy"]
        if df.index[-1] >= today:
            continue   # 오늘 봉이 이미 확정(16시 이후 실행) — 잠정 봉 불필요·무해
        if code not in quotes:
            # 모의 레이트리밋: 명목 초당 2건이지만 0.6s 간격에도 EGW00201 실측
            # (2026-09-02) — 1.1s 간격 + 리밋 1회 재시도. 9종 x 1.1s ≈ 10s 로
            # 15:20 창 안에 충분하다
            for attempt in (1, 2):
                try:
                    time.sleep(1.1)
                    quotes[code] = kis.quote(code)
                    break
                except Exception as e:   # noqa: BLE001 — 종목별 독립 실패 처리
                    if "EGW00201" in str(e) and attempt == 1:
                        continue
                    quotes[code] = {}
                    print(f"  [WARN] {cand['name']} 시세 실패 — 건너뜀: {e}",
                          file=sys.stderr)
        q = quotes[code]
        if not q:
            continue
        bar = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"],
                            "Close": q["price"], "Volume": float("nan")},
                           index=[today])
        df_ext = pd.concat([df, bar])

        if strategy == "volbreak":
            k = ecfg["strategies"]["volbreak"]["k"]
            _, before = simulate_volbreak(df, k, cost, return_open=True)
            trades_after, after = simulate_volbreak(df_ext, k, cost, return_open=True)
        elif strategy == "overnight":
            a = ecfg["strategies"]["overnight"]["entry_above"]
            _, before = simulate_overnight(df, a, cost, return_open=True)
            trades_after, after = simulate_overnight(df_ext, a, cost, return_open=True)
        else:
            from .align import map_us_to_kr, us_returns
            from .data_loader import load_symbol

            _, before = simulate(df, entry, exit_, max_hold, cost,
                                 return_open=True, trailing=trailing)
            us_ext = map_us_to_kr(us_returns(load_symbol("^IXIC", "us")),
                                  df_ext.index, "ixic")["ixic"]
            e2, x2, mh2, tr2 = candidate_flags(cand, df_ext, us_ext)
            trades_after, after = simulate(df_ext, e2, x2, mh2, cost,
                                          return_open=True, trailing=tr2)

        action = _close_action(strategy, before, trades_after, after, today)
        if action == "buy":
            if night_codes.get(code, 0) >= cap:
                out.append({"action": "plan_only", "code": code,
                            "name": cand["name"], "strategy": strategy, "qty": 0,
                            "note": f"진입 신호 발동 — 동일 ETF 합산 캡({cap}슬롯)으로"
                                    " 매수 생략 (섀도 장부는 기록됨)"})
                continue
            qty, weight = _slot_qty(code, q["price"])
            night_codes[code] = night_codes.get(code, 0) + 1
            fill = after["entry_price"]
            out.append({"action": "buy_close", "code": code, "name": cand["name"],
                        "strategy": strategy, "qty": qty,
                        "note": f"종가 매수 (비중 {weight:.1%}, 모형 체결가 "
                                f"{fill:,.0f} — 실체결과의 차이가 슬리피지 실측)"})
        elif action == "sell":
            pos = mock_positions().get((code, strategy))
            if pos is None:
                out.append({"action": "plan_only", "code": code,
                            "name": cand["name"], "strategy": strategy, "qty": 0,
                            "note": "청산 신호 — 실행기 보유 없음 (섀도만 청산)"})
            else:
                out.append({"action": "sell_close", "code": code,
                            "name": cand["name"], "strategy": strategy,
                            "qty": pos["qty"],
                            "note": f"청산 신호 — 종가 매도 (진입 {pos['date']})"})
    return out


def run_close_window(live_mock: bool = False, auto: bool = False) -> None:
    """15:20 실행 창 — 마감 동시호가 직전 종가 근사 제출."""
    cfg = load_config()
    if auto:
        live_mock = cfg.get("ops", {}).get("trade_mode") == "live_mock"
        start, end = cfg.get("ops", {}).get("close_window", ["15:15", "15:30"])
        hhmm = pd.Timestamp.now().strftime("%H:%M")
        if not (start <= hhmm < end):
            print(f"실행 창({start}~{end}) 밖({hhmm}) — 스킵 (창을 놓친 날은 "
                  "모의 제출만 빠지고 섀도 장부는 영향 없음)")
            return
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "live_mock" if live_mock else "dry_run"
    plan = build_close_plan()
    print(f"=== Phase D 15:20 실행 창 ({mode}) — 계획 {len(plan)}건 ===")
    _submit_plan(plan, today, now, mode, live_mock)


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
