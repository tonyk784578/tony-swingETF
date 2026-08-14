"""아침 시장 스냅샷 — '개장 전 세계 상태'의 시점(point-in-time) 기록.

왜: "개장 전 나스닥 선물이 오르면 돌파 성공률이 높은가" 같은 아침 정보 가설을
나중에 시험하려면 그날 아침 그 시각의 실제 값이 필요한데, 선물·환율의 과거
장중 시세는 소급으로 구할 수 없다 (무료 소스 기준). 매일 아침 프리뷰가 몇 개
지표를 CSV 한 줄로 기록해 두면, 1~2년 뒤 새 가설을 사전 등록할 때 정직한
시점 데이터가 준비되어 있다.

수집 전용 — 어떤 신호·판정에도 쓰지 않는다 (재료 축적). 티커는 config
`snapshot.tickers`. 같은 날 재실행은 첫 기록 보존 (preview_signals와 동일 원칙).

**개장 후 기록 주의**: 도입 당일(2026-08-10)에 22:35 기록분이 1행 섞였다.
그 행의 값은 '아침 상태'가 아니므로 아침 정보 가설에 쓰면 안 된다. 과거 행을
고쳐 쓰지 않는 대신(기록은 기록이다) 두 겹으로 막는다 —
(1) 기록 시점이 config `snapshot.premarket_deadline` 이후면 stderr 경고,
(2) 소비자는 `load_snapshots(premarket_only=True)` 로 읽어 자동 제외.
"""

from __future__ import annotations

import sys

import pandas as pd

from .config import ROOT_DIR, load_config


def _snapshot_path():
    path = ROOT_DIR / "paper" / "market_snapshots.csv"
    path.parent.mkdir(exist_ok=True)
    return path


def _deadline() -> str:
    return load_config().get("snapshot", {}).get("premarket_deadline", "09:00")


def is_premarket(taken_at) -> bool:
    """기록 시각이 개장 전인가 — '아침 정보'로 쓸 수 있는 행인지 판별."""
    ts = pd.to_datetime(taken_at, errors="coerce")
    if pd.isna(ts):
        return False
    h, m = (int(x) for x in _deadline().split(":"))
    return bool((ts.hour, ts.minute) < (h, m))


def load_snapshots(premarket_only: bool = False) -> pd.DataFrame:
    """스냅샷 CSV 로드. premarket_only=True 면 개장 후 기록분을 제외한다.

    아침 정보 가설에 쓸 때는 반드시 True — 개장 후 기록은 그날 장중 결과가
    이미 반영된 값이라 룩어헤드가 된다 (모듈 docstring의 08-10 사례).
    """
    path = _snapshot_path()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    if premarket_only and "taken_at" in df.columns:
        df = df[df["taken_at"].map(is_premarket)].reset_index(drop=True)
    return df


def _fetch_quote(symbol: str) -> float:
    """최신 체결가 (선물·환율은 거의 24시간 거래라 아침에도 살아있는 값)."""
    import yfinance as yf

    t = yf.Ticker(symbol)
    try:
        px = t.fast_info["last_price"]
        if px and px > 0:
            return float(px)
    except Exception:
        pass
    hist = t.history(period="1d", interval="1m")["Close"].dropna()
    return float(hist.iloc[-1])


def record_market_snapshot() -> None:
    """아침 스냅샷 1행 기록 — 티커별 독립 시도, 실패는 결측으로 남기고 진행."""
    cfg = load_config().get("snapshot", {})
    tickers = cfg.get("tickers", {})
    if not tickers:
        return
    now = pd.Timestamp.now()
    today = now.normalize()

    path = _snapshot_path()
    prev = pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame()
    if len(prev) and (pd.to_datetime(prev["date"]).dt.normalize() == today).any():
        return   # 오늘 기록 있음 — 최초 시점 보존

    row: dict = {"date": today, "taken_at": now.strftime("%Y-%m-%d %H:%M:%S")}
    for symbol, name in tickers.items():
        try:
            row[name] = _fetch_quote(symbol)
        except Exception as e:   # 티커별 독립 — 한 소스 장애가 행 전체를 막으면 안 됨
            print(f"[warn] snapshot {symbol} failed: {e}", file=sys.stderr)
            row[name] = float("nan")
    pd.concat([prev, pd.DataFrame([row])], ignore_index=True) \
        .to_csv(path, index=False, encoding="utf-8-sig")
    vals = ", ".join(f"{n}={row[n]:,.2f}" for n in tickers.values()
                     if pd.notna(row[n]))
    late = "" if is_premarket(row["taken_at"]) else "  [개장 후 기록 — 아침 정보 아님]"
    print(f"  market snapshot: {vals}{late}")
    if late:
        print(f"[warn] snapshot 기록 시각 {row['taken_at']} 이 개장({_deadline()}) 이후 — "
              "이 행은 아침 정보 가설에 쓸 수 없다 (load_snapshots(premarket_only=True) 가 제외)",
              file=sys.stderr)
