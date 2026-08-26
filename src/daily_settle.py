"""일별 정산 — ETF 섀도 장부를 '가상 계좌' 관점의 일별 CSV로 (운영 편의 계층).

산출: paper/etf_daily.csv — 날짜별 진입/청산 건수, 총수익/총손실(원), 잔액(현금),
평가금액, 총자산, 일간/누적 수익률. 저녁 `paper` 가 자동 갱신, `daily` 로 수동
갱신·조회. 신호·판정과 무관한 파생 산출물이라 재생성 안전 (장부가 원본).

계좌 모형 가정 (config ops — STATUS 주문 계획과 동일 슬롯 규칙):
- 가상 자본 = ops.virtual_capital, 트레이드당 명목 = 자본 x exposure/3 / 레버리지
- 잔액: 진입일 종가 시점에 명목 차감, 청산일에 명목 x (1+net_ret) 회수
- 평가금액: 보유 중 명목 x (당일 종가/진입가) 합 — MTM 은 일봉 종가 근사
- **이론 계좌** — 증거금·주문가능금액 개념 없음. 동시 신호가 많은 날은 잔액이
  음수가 될 수 있다 (그 자체가 동시 노출 과다의 가시화 — sleeves 와 함께 볼 것)
- rotation2 는 별도 계좌 전략(체결가 미보유)이라 제외, 체결가 없는 행도 제외
- 미청산 포지션 포함 (마지막 확정일까지 MTM)
"""

from __future__ import annotations

import pandas as pd

from .config import ROOT_DIR, load_config

DAILY_PATH = ROOT_DIR / "paper" / "etf_daily.csv"


def _notional(capital: float, exposure: float, lev: int) -> float:
    return capital * exposure / 3 / lev


def _trade_rows(ledger: pd.DataFrame, status: list[dict], cfg: dict) -> list[dict]:
    """완결 + 미청산 트레이드를 (code, 진입/청산일, 진입가, net_ret) 로 정규화."""
    code_of = {(c["name"], c["strategy"]): str(c["code"])
               for c in cfg["etf_paper"]["candidates"]}
    rows = []
    if len(ledger):
        done = ledger[(ledger["strategy"] != "rotation2")
                      & ledger["entry_price"].notna()]
        for _, t in done.iterrows():
            rows.append({"code": str(t["code"]) or code_of.get(
                             (t["name"], t["strategy"]), ""),
                         "name": t["name"], "entry_date": t["entry_date"],
                         "exit_date": t["exit_date"],
                         "entry_price": float(t["entry_price"]),
                         "net_ret": float(t["net_ret"])})
    for st in status:
        pos = st.get("open_pos")
        if not pos or st["name"] == "(포트폴리오)":
            continue
        code = code_of.get((st["name"], st["strategy"]))
        if code:
            rows.append({"code": code, "name": st["name"],
                         "entry_date": pd.Timestamp(pos["entry_date"]),
                         "exit_date": None,
                         "entry_price": float(pos["entry_price"]),
                         "net_ret": float("nan")})
    return rows


def build_daily(ledger: pd.DataFrame, status: list[dict]) -> pd.DataFrame:
    from .data_loader import confirmed_cutoff, load_symbol

    cfg = load_config()
    ops = cfg.get("ops", {})
    capital = float(ops.get("virtual_capital", 10_000_000))
    exposure = float(ops.get("exposure", 0.5))
    lev_map = cfg.get("portfolio", {}).get("leverage", {})

    trades = _trade_rows(ledger, status, cfg)
    if not trades:
        return pd.DataFrame()
    for t in trades:
        t["notional"] = _notional(capital, exposure, int(lev_map.get(t["code"], 1)))

    cut = confirmed_cutoff()
    closes = {}
    for code in {t["code"] for t in trades}:
        px = load_symbol(code, "kr")["Close"]
        closes[code] = px[px.index <= cut]
    calendar = closes[max(closes, key=lambda c: len(closes[c]))].index
    start = min(t["entry_date"] for t in trades)
    calendar = calendar[(calendar >= start) & (calendar <= cut)]

    rows = []
    cash = capital
    for d in calendar:
        entries = [t for t in trades if t["entry_date"] == d]
        exits = [t for t in trades if t["exit_date"] == d]
        cash -= sum(t["notional"] for t in entries)
        pnl = [(t["notional"] * t["net_ret"]) for t in exits]
        cash += sum(t["notional"] * (1 + t["net_ret"]) for t in exits)

        pos_value, n_open = 0.0, 0
        for t in trades:
            if t["entry_date"] <= d and (t["exit_date"] is None or d < t["exit_date"]):
                px = closes[t["code"]].asof(d)
                if pd.notna(px):
                    pos_value += t["notional"] * float(px) / t["entry_price"]
                    n_open += 1
        equity = cash + pos_value
        rows.append({"date": d.date(), "n_entry": len(entries), "n_exit": len(exits),
                     "win_krw": round(sum(p for p in pnl if p > 0)),
                     "loss_krw": round(sum(p for p in pnl if p < 0)),
                     "realized_krw": round(sum(pnl)),
                     "cash": round(cash), "position_value": round(pos_value),
                     "n_open": n_open, "equity": round(equity),
                     "cum_ret": round(equity / capital - 1, 6)})
    out = pd.DataFrame(rows)
    out["daily_ret"] = (out["equity"] / out["equity"].shift(1) - 1).round(6).fillna(0.0)
    return out


def update_daily_settlement(ledger: pd.DataFrame, status: list[dict],
                            show_tail: int = 8) -> pd.DataFrame:
    daily = build_daily(ledger, status)
    if daily.empty:
        print("[일별 정산] 대상 트레이드 없음")
        return daily
    DAILY_PATH.parent.mkdir(exist_ok=True)
    daily.to_csv(DAILY_PATH, index=False, encoding="utf-8-sig")
    won = "{:,.0f}".format
    print(f"\n=== 일별 정산 (가상 계좌 — ops 슬롯 규칙, 최근 {show_tail}일) ===")
    print(daily.tail(show_tail).to_string(
        index=False, formatters={"win_krw": won, "loss_krw": won,
                                 "realized_krw": won, "cash": won,
                                 "position_value": won, "equity": won,
                                 "cum_ret": "{:+.2%}".format,
                                 "daily_ret": "{:+.2%}".format}))
    print(f"ledger: {DAILY_PATH}")
    return daily
