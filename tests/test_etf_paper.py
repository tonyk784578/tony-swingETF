"""ETF 섀도 장부 — 실행가능성(preview) 플래그 분류 테스트."""

import pandas as pd

from src.etf_paper import _preview_flag

DEADLINE = "09:00"


def _snap(rows):
    return pd.DataFrame(rows, columns=["date", "generated_at", "name", "strategy",
                                       "enter_today"])


def test_ok_when_signal_seen_before_open():
    snap = _snap([{"date": pd.Timestamp("2026-08-10"),
                   "generated_at": "2026-08-10 08:40:00",
                   "name": "KODEX200", "strategy": "breakout", "enter_today": True}])
    assert _preview_flag(pd.Timestamp("2026-08-10"), "KODEX200", "breakout",
                         snap, DEADLINE) == "ok"


def test_late_when_preview_ran_after_open():
    # WSL 늦은 부팅으로 09:35에야 프리뷰 실행 — 시가 진입 불가능
    snap = _snap([{"date": pd.Timestamp("2026-08-10"),
                   "generated_at": "2026-08-10 09:35:00",
                   "name": "KODEX200", "strategy": "breakout", "enter_today": True}])
    assert _preview_flag(pd.Timestamp("2026-08-10"), "KODEX200", "breakout",
                         snap, DEADLINE) == "late"


def test_miss_when_on_time_but_no_signal():
    # 프리뷰는 제때 돌았는데 신호 미표시 (데이터 지연 등) — 조사 대상
    snap = _snap([{"date": pd.Timestamp("2026-08-10"),
                   "generated_at": "2026-08-10 08:40:00",
                   "name": "KODEX200", "strategy": "breakout", "enter_today": False}])
    assert _preview_flag(pd.Timestamp("2026-08-10"), "KODEX200", "breakout",
                         snap, DEADLINE) == "miss"


def test_none_when_no_record_for_that_day():
    snap = _snap([{"date": pd.Timestamp("2026-08-10"),
                   "generated_at": "2026-08-10 08:40:00",
                   "name": "KODEX200", "strategy": "breakout", "enter_today": True}])
    assert _preview_flag(pd.Timestamp("2026-08-11"), "KODEX200", "breakout",
                         snap, DEADLINE) == "none"
    assert _preview_flag(pd.Timestamp("2026-08-10"), "KODEX_Lev", "breakout",
                         snap, DEADLINE) == "none"
    assert _preview_flag(pd.Timestamp("2026-08-10"), "KODEX200", "trend_ride",
                         snap, DEADLINE) == "none"


def test_none_on_empty_snapshot():
    assert _preview_flag(pd.Timestamp("2026-08-10"), "KODEX200", "breakout",
                         pd.DataFrame(), DEADLINE) == "none"


def test_first_record_wins():
    # 같은 날 두 번 실행 — 최초 인지 시점(08:40)이 기준이어야 함
    snap = _snap([
        {"date": pd.Timestamp("2026-08-10"), "generated_at": "2026-08-10 08:40:00",
         "name": "KODEX200", "strategy": "breakout", "enter_today": True},
        {"date": pd.Timestamp("2026-08-10"), "generated_at": "2026-08-10 09:40:00",
         "name": "KODEX200", "strategy": "breakout", "enter_today": True},
    ])
    assert _preview_flag(pd.Timestamp("2026-08-10"), "KODEX200", "breakout",
                         snap, DEADLINE) == "ok"
