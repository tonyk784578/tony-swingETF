"""ETF 스윙 섀도 (Stage 2) — 사전 등록 후보의 포워드 검증 장부.

방식: 매 저녁 확정 봉(16시 이전 실행 시 당일 제외)까지로 전체 히스토리를
리플레이하고, freeze_date 이후 **진입**한 완결 트레이드만 장부에 append한다
(키: name+strategy+entry_date, 멱등). 미청산 포지션은 상태로만 표시.
리플레이 방식이라 증분 상태 관리 버그가 없고, 데이터 정정은 오염 감시가 잡는다.
"""

from __future__ import annotations

import pandas as pd

from .backtest import combo_stats
from .config import ROOT_DIR, load_config
from .data_loader import confirmed_cutoff
from .etf_swing import iter_candidates, raw_entry_signal, simulate

LEDGER_COLS = ["entry_date", "exit_date", "name", "strategy", "hold", "net_ret",
               "preview"]

# preview 컬럼 — 섀도와 실거래의 구조적 괴리 진단용 (2026-08-07 추가):
#   ok   = 진입일 아침 개장(09:00) 전 프리뷰가 신호를 표시 → 실거래 가능했던 트레이드
#   late = 프리뷰가 개장 후에야 실행됨 (WSL 늦은 부팅) → 시가 진입 불가능했던 날
#   miss = 프리뷰는 제때 돌았지만 신호 미표시 (데이터 지연 등) → 원인 조사 대상
#   none = 그날 아침 프리뷰 기록 없음 / rotation2 등 대상 외
# 판정(계열 t검정)은 전체 트레이드로 하되, ok 부분집합 성과를 병기해
# "섀도 성과가 실거래로 옮겨지는가"를 Phase D 전에 답할 수 있게 한다.


def _ledger_path():
    path = ROOT_DIR / load_config()["etf_paper"]["ledger"]
    path.parent.mkdir(exist_ok=True)
    return path


def _preview_log_path():
    path = ROOT_DIR / "paper" / "preview_signals.csv"
    path.parent.mkdir(exist_ok=True)
    return path


def load_etf_ledger() -> pd.DataFrame:
    path = _ledger_path()
    if path.exists():
        led = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
        if "preview" not in led.columns:   # 구버전 장부 호환
            led["preview"] = "none"
        return led
    return pd.DataFrame(columns=LEDGER_COLS)


def record_preview_signals(states: list[dict]) -> None:
    """아침 프리뷰가 본 진입 신호를 타임스탬프와 함께 기록 (실행가능성 증빙).

    같은 날 재실행돼도 첫 기록을 보존한다 — '언제 처음 알 수 있었나'가 증빙이므로
    나중 실행으로 덮어쓰면 안 된다.

    16시(확정 컷오프) 이후 실행은 기록하지 않는다: 당일 봉이 확정된 뒤의 신호는
    '내일 시가 진입'을 뜻하므로 date=오늘로 적으면 진입일 매칭이 어긋난다.
    내일 신호는 내일 아침 프리뷰가 (어제까지 데이터로 동일하게) 다시 계산한다.
    """
    now = pd.Timestamp.now()
    today = now.normalize()
    if today <= confirmed_cutoff():
        return
    path = _preview_log_path()
    prev = (pd.read_csv(path, parse_dates=["date"]) if path.exists()
            else pd.DataFrame(columns=["date", "generated_at", "name", "strategy",
                                       "enter_today"]))
    have = set(zip(prev["name"], prev["strategy"],
                   pd.to_datetime(prev["date"]).dt.normalize(), strict=True))
    rows = [{"date": today, "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
             "name": st["cand"]["name"], "strategy": st["cand"]["strategy"],
             "enter_today": bool(st.get("enter_today"))}
            for st in states
            if (st["cand"]["name"], st["cand"]["strategy"], today) not in have]
    if rows:
        pd.concat([prev, pd.DataFrame(rows)], ignore_index=True) \
            .to_csv(path, index=False, encoding="utf-8-sig")


def _preview_flag(entry_date: pd.Timestamp, name: str, strategy: str,
                  snap: pd.DataFrame, deadline: str) -> str:
    """완결 트레이드의 진입일 아침 프리뷰 기록 → ok/late/miss/none."""
    if snap.empty:
        return "none"
    sub = snap[(snap["name"] == name) & (snap["strategy"] == strategy)
               & (pd.to_datetime(snap["date"]).dt.normalize() == entry_date.normalize())]
    if sub.empty:
        return "none"
    row = sub.iloc[0]                      # 첫 기록 = 최초 인지 시점
    on_time = pd.Timestamp(row["generated_at"]) < pd.Timestamp(
        f"{entry_date.date()} {deadline}")
    if not on_time:
        return "late"
    return "ok" if row["enter_today"] else "miss"


def _candidate_frames(force: bool = False):
    """후보별 (설정, 확정봉 df, entry/exit 플래그, max_hold, trailing) 생성."""
    yield from iter_candidates(force, cutoff=confirmed_cutoff())


def update_etf_ledger(force: bool = False) -> tuple[pd.DataFrame, int, list[dict]]:
    cfg = load_config()
    cost = cfg["etf"]["cost_round_trip"]
    ledger = load_etf_ledger()
    snap_path = _preview_log_path()
    snap = (pd.read_csv(snap_path, parse_dates=["date"]) if snap_path.exists()
            else pd.DataFrame())
    deadline = cfg["etf_paper"]["preview_deadline"]

    new_rows = []
    status = []
    for cand, df, entry, exit_, max_hold, trailing in _candidate_frames(force):
        # 후보별 freeze — 나중에 등록된 후보(예: trend_ride 2026-08-07)는 자기
        # 등록일 이후만 아웃오브샘플. 기존 후보의 freeze는 건드리지 않는다.
        freeze = pd.Timestamp(cand.get("freeze", cfg["etf_paper"]["freeze_date"]))
        trades, open_pos = simulate(df, entry, exit_, max_hold, cost,
                                    return_open=True, trailing=trailing)
        insample = None
        if len(trades):
            r = trades.set_index("entry_date")["net_ret"]
            insample = combo_stats(r[r.index < freeze])
        # freeze 당일 진입은 제외(>) — 그날 신호는 후보 선정에 쓴 데이터에서 나온다
        fwd = trades[trades["entry_date"] > freeze] if len(trades) else trades
        done = set()
        if not ledger.empty:
            sub = ledger[(ledger["name"] == cand["name"])
                         & (ledger["strategy"] == cand["strategy"])]
            done = set(sub["entry_date"])
        for _, t in fwd.iterrows():
            if t["entry_date"] in done:
                continue
            new_rows.append({"entry_date": t["entry_date"], "exit_date": t["exit_date"],
                             "name": cand["name"], "strategy": cand["strategy"],
                             "hold": t["hold"], "net_ret": round(t["net_ret"], 6),
                             "preview": _preview_flag(t["entry_date"], cand["name"],
                                                      cand["strategy"], snap, deadline)})
        status.append({"name": cand["name"], "strategy": cand["strategy"],
                       "insample_mean": insample["mean"] if insample else float("nan"),
                       "open_pos": open_pos})

    status += _rotation2_rows(cfg, ledger, new_rows, force)

    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
        ledger = ledger.sort_values(["entry_date", "name"]).reset_index(drop=True)
        ledger.to_csv(_ledger_path(), index=False, encoding="utf-8-sig")
    return ledger, len(new_rows), status


def _rotation2_rows(cfg: dict, ledger: pd.DataFrame, new_rows: list,
                    force: bool) -> list[dict]:
    """확장 로테이션(Stage 2, 2026-08-07 통과·편입) 완결 에피소드 append + 상태."""
    r2 = cfg.get("etf_rotation2", {})
    if not r2.get("freeze"):
        return []
    from .rotation import rotation2_episodes

    ep, open_pos, _ = rotation2_episodes(force)
    freeze = pd.Timestamp(r2["freeze"])
    insample = combo_stats(ep[ep["entry_date"] < freeze]["net_ret"]) if len(ep) else None
    fwd = ep[ep["entry_date"] > freeze] if len(ep) else ep
    sub = ledger[ledger["strategy"] == "rotation2"] if not ledger.empty else ledger
    done = set(zip(sub["name"], sub["entry_date"], strict=True)) if len(sub) else set()
    for _, t in fwd.iterrows():
        if (t["name"], t["entry_date"]) in done:
            continue
        new_rows.append({"entry_date": t["entry_date"], "exit_date": t["exit_date"],
                         "name": t["name"], "strategy": "rotation2",
                         "hold": t["hold"], "net_ret": round(t["net_ret"], 6),
                         "preview": "none"})   # 월초 리밸런스는 전월 말에 예고 — 대상 외
    pos_text = (", ".join(f"{p['name']}({p['unrealized']:+.1%})" for p in open_pos)
                if open_pos else "전 슬롯 현금")
    return [{"name": "(포트폴리오)", "strategy": "rotation2",
             "insample_mean": insample["mean"] if insample else float("nan"),
             "open_pos": None, "position_text": f"보유 {pos_text}"}]


def etf_forward_summary(ledger: pd.DataFrame, status: list[dict]) -> pd.DataFrame:
    rows = []
    for st in status:
        if st["name"] == "(포트폴리오)":   # rotation2 — 에피소드가 ETF명으로 기록됨
            sub = ledger[ledger["strategy"] == st["strategy"]]
        else:
            sub = ledger[(ledger["name"] == st["name"])
                         & (ledger["strategy"] == st["strategy"])]
        fs = combo_stats(sub["net_ret"]) if len(sub) else combo_stats(pd.Series([], dtype=float))
        # 실거래 가능(preview=ok) 부분집합 — 섀도→실거래 전이 가능성 진단용 병기
        ok = sub[sub["preview"] == "ok"] if len(sub) else sub
        pos = st["open_pos"]
        rows.append({
            "name": st["name"], "strategy": st["strategy"],
            "fwd_n": fs["n"], "fwd_mean": fs["mean"], "fwd_cum": fs["cum_ret"],
            "ok_n": len(ok),
            "ok_mean": float(ok["net_ret"].mean()) if len(ok) else float("nan"),
            "insample_mean": st["insample_mean"],
            "position": st.get("position_text")
            or (f"보유 {pos['hold']}일째 ({pos['unrealized']:+.2%})"
                if pos else "무포지션"),
        })
    return pd.DataFrame(rows)


def candidate_states(force: bool = False) -> list[dict]:
    """후보별 아침 상태 (프리뷰·브리핑 공용) — 진입 판정은 백테스트와 동일한
    raw_entry_signal의 마지막 값. 별도 구현을 두면 전략 변경 시 조용히 어긋난다."""
    cfg = load_config()
    out = []
    for cand, df, entry, exit_, max_hold, trailing in _candidate_frames(force):
        if cand["strategy"] == "us_dip":
            # 오늘의 us_dip 플래그 = 가장 최근 확정 미국 봉 (아직 df 인덱스 밖)
            from .align import us_returns
            from .data_loader import load_symbol

            latest_us = us_returns(load_symbol("^IXIC", "us")).iloc[-1]
            thr = cfg["etf"]["strategies"]["us_dip"]["threshold"]
            enter_today = bool(latest_us <= thr)
        else:
            # 마지막 확정 봉의 raw 신호 = '다음 개장(오늘) 진입' — 백테스트와 동일 함수
            raw = raw_entry_signal(df, cand["strategy"], pd.Series(dtype=float))
            enter_today = bool(raw.iloc[-1])

        _, open_pos = simulate(df, entry, exit_, max_hold,
                               cfg["etf"]["cost_round_trip"],
                               return_open=True, trailing=trailing)
        out.append({"cand": cand, "enter_today": enter_today, "open_pos": open_pos,
                    "last_close": float(df["Close"].iloc[-1]),
                    "last_date": df.index[-1]})
    return out


def rotation2_state(force: bool = False) -> dict | None:
    """확장 로테이션 상태 (프리뷰·브리핑 공용): 보유 슬롯 + 월초 리밸런스 여부."""
    cfg = load_config()
    if not cfg.get("etf_rotation2", {}).get("freeze"):
        return None
    from .rotation import rotation2_episodes, rotation2_universe, select_targets

    _, open_pos, idx = rotation2_episodes(force)
    # 마지막 확정 봉이 그 달의 월말이면(=오늘이 다음 달 첫 거래일이면) 오늘 리밸런스
    rebalance_today = bool(idx[-1].month != pd.Timestamp.today().month)
    moves: list[str] = []
    if rebalance_today:
        from .data_loader import load_symbol
        r2 = cfg["etf_rotation2"]
        closes = pd.DataFrame({n: load_symbol(str(c), "kr").loc[:idx[-1], "Close"]
                               for c, n in rotation2_universe().items()})
        mom = (closes / closes.shift(r2["lookback"]) - 1).iloc[-1]
        target = select_targets(mom, r2["top_k"])
        cur = {p["name"] for p in open_pos}
        if target - cur:
            moves.append("IN " + ", ".join(sorted(target - cur)))
        if cur - target:
            moves.append("OUT " + ", ".join(sorted(cur - target)))
    return {"open_pos": open_pos, "rebalance_today": rebalance_today, "moves": moves}


def _state_text(st: dict) -> str:
    pos = st["open_pos"]
    if pos:
        return (f"보유 {pos['hold']}일째, 진입 {pos['entry_date'].date()} "
                f"@{pos['entry_price']:,.0f}, 평가 {pos['unrealized']:+.2%}")
    return "오늘 시가 진입 대기" if st["enter_today"] else "무포지션 · 신호 없음"


def preview_etf(force: bool = False) -> None:
    """아침용: 오늘 시가 진입 신호 여부 + 보유 포지션 상태."""
    print("\n=== ETF swing preview (Stage 2 candidates) ===")
    states = candidate_states(force)
    record_preview_signals(states)   # 실행가능성 증빙 — 최초 인지 시점 기록
    for st in states:
        print(f"  [{st['cand']['name']}] {st['cand']['strategy']}: {_state_text(st)}")
    rot = rotation2_state(force)
    if rot:
        held = (", ".join(f"{p['name']}({p['unrealized']:+.1%})"
                          for p in rot["open_pos"]) or "현금")
        line = f"  [로테이션] rotation2: 보유 {held}"
        if rot["rebalance_today"]:
            line += " · 오늘 시가 리밸런스: " + ("; ".join(rot["moves"]) or "변경 없음")
        print(line)
