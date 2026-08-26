"""일별 정산 테스트 — 가상 계좌 산수(잔액/평가/총자산)가 정의대로 계산되는지."""

import pandas as pd
import pytest

from src import daily_settle
from src.config import load_config


def _fake_prices(code, market, force=False):
    idx = pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-24"])
    return pd.DataFrame({"Close": [100.0, 101.0, 99.0]}, index=idx)


def test_build_daily_cash_and_equity(monkeypatch):
    monkeypatch.setattr("src.data_loader.load_symbol", _fake_prices)
    monkeypatch.setattr("src.data_loader.confirmed_cutoff",
                        lambda: pd.Timestamp("2026-08-24"))
    ledger = pd.DataFrame([{
        "entry_date": pd.Timestamp("2026-08-20"), "exit_date": pd.Timestamp("2026-08-21"),
        "code": "091160", "name": "KODEX_Semicon", "strategy": "overnight",
        "family": "overnight", "hold": 1, "entry_price": 100.0, "exit_price": 101.0,
        "gross_ret": 0.011, "cost": 0.001, "net_ret": 0.01, "preview": "ok"}])
    out = daily_settle.build_daily(ledger, status=[])

    ops = load_config()["ops"]
    capital = ops["virtual_capital"]
    lev = load_config()["portfolio"]["leverage"].get("091160", 1)
    n = capital * ops["exposure"] / 3 / lev

    d1, d2 = out.iloc[0], out.iloc[1]
    # 진입일: 현금 = 자본 - 명목, 평가 = 명목(진입가=종가) → 총자산 = 자본
    assert d1["n_entry"] == 1 and d1["cash"] == round(capital - n)
    assert d1["equity"] == pytest.approx(capital, abs=2)
    # 청산일: 명목 x (1+net_ret) 회수 → 총자산 = 자본 + 명목 x 1%
    assert d2["n_exit"] == 1 and d2["n_open"] == 0
    assert d2["equity"] == pytest.approx(capital + n * 0.01, abs=2)
    assert d2["win_krw"] == round(n * 0.01) and d2["loss_krw"] == 0
    # 이후 무포지션 — 총자산 불변
    assert out.iloc[2]["equity"] == d2["equity"]


def test_rotation2_and_priceless_rows_excluded(monkeypatch):
    monkeypatch.setattr("src.data_loader.load_symbol", _fake_prices)
    monkeypatch.setattr("src.data_loader.confirmed_cutoff",
                        lambda: pd.Timestamp("2026-08-24"))
    ledger = pd.DataFrame([{
        "entry_date": pd.Timestamp("2026-08-20"), "exit_date": pd.Timestamp("2026-08-21"),
        "code": "", "name": "KODEX200", "strategy": "rotation2", "family": "rotation2",
        "hold": 1, "entry_price": float("nan"), "exit_price": float("nan"),
        "gross_ret": float("nan"), "cost": float("nan"), "net_ret": 0.05,
        "preview": "none"}])
    assert daily_settle.build_daily(ledger, status=[]).empty
