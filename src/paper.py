"""포워드 페이퍼 테스트 — 브로커 없는 시그널 장부(ledger).

config paper.conditions에 사전 등록된 (종목, 조건)만 추적한다.
freeze_date 이후 각 한국 거래일에 대해 트리거 여부와 가상 체결
(시가 매수 → 종가 매도, 왕복 비용 차감)을 ledger CSV에 append한다.
- 트리거 안 된 날도 기록 (커버리지 확인용) — 이미 기록된 날짜는 건너뛴다(멱등).
- freeze_date 이전 구간은 인샘플이므로 ledger에 넣지 않고 비교 기준으로만 쓴다.
- ledger는 삭제/재생성하지 않는다 (사전 등록 기록의 의미가 사라짐).

preview: 다음 거래일 아침(미국장 마감 후, 한국장 개장 전) 기준으로 각 조건이
발동 대기 상태인지 미리 보여준다. 갭 조건은 당일 시가가 나와야 확정되므로
'개장 시 확인'으로 표시한다.
"""

from __future__ import annotations

import pandas as pd

from .backtest import combo_stats, condition_mask
from .config import ROOT_DIR, load_config
from .signals import band

LEDGER_COLS = ["date", "stock", "condition", "triggered",
               "open", "close", "gap", "net_ret", "weight"]


def _ledger_path():
    path = ROOT_DIR / load_config()["paper"]["ledger"]
    path.parent.mkdir(exist_ok=True)
    return path


def load_ledger() -> pd.DataFrame:
    path = _ledger_path()
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame(columns=LEDGER_COLS)


def update_ledger(masters: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    cfg = load_config()
    pcfg = cfg["paper"]
    cost = cfg["cost"]["round_trip"]
    freeze = pd.Timestamp(pcfg["freeze_date"])
    ledger = load_ledger()

    from .data_loader import confirmed_cutoff

    cutoff = confirmed_cutoff()   # 16시 이전 실행 시 당일 미완성 봉 제외
    scfg = cfg["sizing"]
    new_rows = []
    for item in pcfg["conditions"]:
        stock, cond = item["stock"], item["condition"]
        master = masters[stock]
        mask = condition_mask(master, cond)
        done = set(ledger.loc[(ledger["stock"] == stock)
                              & (ledger["condition"] == cond), "date"])
        # freeze 당일은 제외(>) — 그날 신호는 조건 선정에 쓴 데이터에서 나온다
        sel = master[(master.index > freeze) & (master.index <= cutoff)]
        for d, row in sel.iterrows():
            if d in done:
                continue
            trig = bool(mask.loc[d])
            # 2단 사이징 포워드 추적 (config sizing — 판정은 포워드 데이터로)
            weight = (scfg["strong_weight"] if row["nvda_ret"] <= scfg["strong_threshold"]
                      else scfg["weak_weight"])
            new_rows.append({
                "date": d, "stock": stock, "condition": cond, "triggered": trig,
                "open": row["open"], "close": row["close"],
                "gap": round(row["gap"], 6),
                "net_ret": round(row["day_ret"] - cost, 6) if trig else None,
                "weight": weight,
            })

    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
        ledger = ledger.sort_values(["date", "stock", "condition"]).reset_index(drop=True)
        ledger.to_csv(_ledger_path(), index=False, encoding="utf-8-sig")
    return ledger, len(new_rows)


def forward_summary(masters: dict[str, pd.DataFrame], ledger: pd.DataFrame) -> pd.DataFrame:
    """조건별 포워드 성과 vs freeze 이전(인샘플) 백테스트 기대치."""
    cfg = load_config()
    pcfg = cfg["paper"]
    cost = cfg["cost"]["round_trip"]
    freeze = pd.Timestamp(pcfg["freeze_date"])

    rows = []
    for item in pcfg["conditions"]:
        stock, cond = item["stock"], item["condition"]
        pre = masters[stock][masters[stock].index < freeze]
        st_pre = combo_stats((pre["day_ret"] - cost)[condition_mask(pre, cond)])

        sub = ledger[(ledger["stock"] == stock) & (ledger["condition"] == cond)]
        fwd = sub[sub["triggered"] == True]  # noqa: E712 (CSV 왕복 후 bool 비교)
        st_fwd = combo_stats(fwd["net_ret"].dropna())
        w = fwd["weight"] if "weight" in fwd else pd.Series(1.0, index=fwd.index)
        sized_mean = ((w * fwd["net_ret"]).sum() / w.sum()
                      if len(fwd) and w.sum() else float("nan"))
        rows.append({
            "stock": stock, "condition": cond,
            "days_watched": len(sub), "fwd_n": st_fwd["n"],
            "fwd_mean": st_fwd["mean"], "fwd_sized_mean": sized_mean,
            "fwd_win": st_fwd["win_rate"], "fwd_cum": st_fwd["cum_ret"],
            "insample_mean": st_pre["mean"], "insample_win": st_pre["win_rate"],
        })
    return pd.DataFrame(rows)


def preview(data: dict[str, pd.DataFrame]) -> None:
    """다음 거래일 기준 조건 발동 여부 미리보기 (갭 조건은 개장 시 확정).

    장부(update_ledger)와 동일하게 confirmed_cutoff를 적용한다. 이게 없으면
    장중에 실행했을 때 미완성 당일 봉으로 조건을 평가해 실제와 다른 값이 찍힌다
    (아침 캐치업이 개장 후에 도는 날이 있으므로 실제로 발생하는 경로다).
    """
    from .data_loader import confirmed_cutoff

    scfg = load_config()["signals"]
    pcfg = load_config()["paper"]
    cutoff = confirmed_cutoff()

    latest: dict[str, tuple[float, pd.Timestamp]] = {}
    for name, key in [("SOX", "sox"), ("NVDA", "nvda"), ("KOSPI", "kospi")]:
        ret = data[name]["Close"].pct_change().dropna()
        ret = ret[ret.index <= cutoff]
        if ret.empty:
            print(f"  [{name}] 확정 봉 없음 — 프리뷰 생략")
            return
        latest[key] = (float(ret.iloc[-1]), ret.index[-1])

    print("=== paper preview — 최신 확정 데이터 기준 ===")
    for key, (ret, d) in latest.items():
        print(f"  {key.upper():5s} last bar {d.date()}  {ret:+.2%}")

    for item in pcfg["conditions"]:
        stock, cond = item["stock"], item["condition"]
        closes = data[stock]["Close"]
        closes = closes[closes.index <= cutoff]
        above = bool(closes.iloc[-1] > closes.rolling(20).mean().iloc[-1])
        parts = []
        armed = True
        pending_gap = False
        for name in cond.split(" & "):
            cat, label = name.split("_", 1)
            if cat == "gap":
                parts.append(f"{name}=개장시확인")
                pending_gap = True
            elif cat == "ma20":
                want = scfg["ma20"][label]
                hit = above == want
                parts.append(f"{name}={hit}")
                armed &= hit
            else:
                lo, hi = scfg[cat][label]
                val = latest[cat][0]
                hit = bool(band(pd.Series([val]), lo, hi).iloc[0])
                parts.append(f"{name}={hit}({val:+.2%})")
                armed &= hit
        if not armed:
            verdict = "대기 안 함"
        elif pending_gap:
            verdict = "조건부 발동 — 갭 조건만 개장 시 확인"
        else:
            verdict = "발동 대기 (시가 매수 대상)"
        print(f"  [{stock}] {cond}\n      {', '.join(parts)}  ->  {verdict}")
