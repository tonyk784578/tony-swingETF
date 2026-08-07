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

LEDGER_COLS = ["entry_date", "exit_date", "name", "strategy", "hold", "net_ret"]


def _ledger_path():
    path = ROOT_DIR / load_config()["etf_paper"]["ledger"]
    path.parent.mkdir(exist_ok=True)
    return path


def load_etf_ledger() -> pd.DataFrame:
    path = _ledger_path()
    if path.exists():
        return pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    return pd.DataFrame(columns=LEDGER_COLS)


def _candidate_frames(force: bool = False):
    """후보별 (설정, 확정봉 df, entry/exit 플래그, max_hold, trailing) 생성."""
    yield from iter_candidates(force, cutoff=confirmed_cutoff())


def update_etf_ledger(force: bool = False) -> tuple[pd.DataFrame, int, list[dict]]:
    cfg = load_config()
    cost = cfg["etf"]["cost_round_trip"]
    freeze = pd.Timestamp(cfg["etf_paper"]["freeze_date"])
    ledger = load_etf_ledger()

    new_rows = []
    status = []
    for cand, df, entry, exit_, max_hold, trailing in _candidate_frames(force):
        trades, open_pos = simulate(df, entry, exit_, max_hold, cost,
                                    return_open=True, trailing=trailing)
        insample = combo_stats(trades.set_index("entry_date")["net_ret"]
                               [lambda s: s.index < freeze]) if len(trades) else None
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
                             "hold": t["hold"], "net_ret": round(t["net_ret"], 6)})
        status.append({"name": cand["name"], "strategy": cand["strategy"],
                       "insample_mean": insample["mean"] if insample else float("nan"),
                       "open_pos": open_pos})

    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
        ledger = ledger.sort_values(["entry_date", "name"]).reset_index(drop=True)
        ledger.to_csv(_ledger_path(), index=False, encoding="utf-8-sig")
    return ledger, len(new_rows), status


def etf_forward_summary(ledger: pd.DataFrame, status: list[dict]) -> pd.DataFrame:
    rows = []
    for st in status:
        sub = ledger[(ledger["name"] == st["name"])
                     & (ledger["strategy"] == st["strategy"])]
        fs = combo_stats(sub["net_ret"]) if len(sub) else combo_stats(pd.Series([], dtype=float))
        pos = st["open_pos"]
        rows.append({
            "name": st["name"], "strategy": st["strategy"],
            "fwd_n": fs["n"], "fwd_mean": fs["mean"], "fwd_cum": fs["cum_ret"],
            "insample_mean": st["insample_mean"],
            "position": (f"보유 {pos['hold']}일째 ({pos['unrealized']:+.2%})"
                         if pos else "무포지션"),
        })
    return pd.DataFrame(rows)


def preview_etf(force: bool = False) -> None:
    """아침용: 오늘 시가 진입 신호 여부 + 보유 포지션 상태.

    진입 판정은 백테스트와 동일한 raw_entry_signal의 마지막 값을 쓴다 —
    별도 구현을 두면 전략 정의 변경 시 프리뷰가 조용히 어긋난다.
    """
    print("\n=== ETF swing preview (Stage 2 candidates) ===")
    for cand, df, entry, exit_, max_hold, trailing in _candidate_frames(force):
        if cand["strategy"] == "us_dip":
            # 오늘의 us_dip 플래그 = 가장 최근 확정 미국 봉 (아직 df 인덱스 밖)
            from .align import us_returns
            from .data_loader import load_symbol

            latest_us = us_returns(load_symbol("^IXIC", "us")).iloc[-1]
            thr = load_config()["etf"]["strategies"]["us_dip"]["threshold"]
            enter_today = bool(latest_us <= thr)
        else:
            # 마지막 확정 봉의 raw 신호 = '다음 개장(오늘) 진입' — 백테스트와 동일 함수
            raw = raw_entry_signal(df, cand["strategy"], pd.Series(dtype=float))
            enter_today = bool(raw.iloc[-1])

        _, open_pos = simulate(df, entry, exit_, max_hold,
                               load_config()["etf"]["cost_round_trip"],
                               return_open=True, trailing=trailing)
        if open_pos:
            state = (f"보유 {open_pos['hold']}일째, 진입 {open_pos['entry_date'].date()} "
                     f"@{open_pos['entry_price']:,.0f}, 평가 {open_pos['unrealized']:+.2%}")
        elif enter_today:
            state = "오늘 시가 진입 대기"
        else:
            state = "무포지션 · 신호 없음"
        print(f"  [{cand['name']}] {cand['strategy']}: {state}")
