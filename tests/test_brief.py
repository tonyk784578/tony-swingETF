"""브리핑 주문 계획 테스트 — 비중·수량 환산과 제외 규칙."""

import pytest

from src.brief import order_plan


def _state(name, code, enter=True, open_pos=None, close=10000.0):
    return {"cand": {"name": name, "code": code, "strategy": "breakout"},
            "enter_today": enter, "open_pos": open_pos, "last_close": close,
            "last_date": None}


def test_order_plan_basic_sizing():
    plans = order_plan([_state("KODEX200", "069500")], capital=10_000_000,
                       exposure=0.6, leverage={})
    assert len(plans) == 1
    p = plans[0]
    assert p["amount"] == pytest.approx(10_000_000 * 0.6 / 3)   # 슬롯 1/3
    assert p["qty"] == int(p["amount"] // 10000)


def test_order_plan_leverage_halves_cash():
    plans = order_plan([_state("KODEX_Lev", "122630")], capital=9_000_000,
                       exposure=1.0, leverage={"122630": 2})
    assert plans[0]["amount"] == pytest.approx(9_000_000 / 3 / 2)  # 현금 절반, 노출 동일


def test_order_plan_skips_holding_and_no_signal():
    states = [
        _state("A", "1", enter=False),                       # 신호 없음
        _state("B", "2", enter=True, open_pos={"hold": 3}),  # 이미 보유 — 단일 포지션
        _state("C", "3", enter=True),
    ]
    plans = order_plan(states, capital=1_000_000, exposure=0.5, leverage={})
    assert [p["name"] for p in plans] == ["C"]
