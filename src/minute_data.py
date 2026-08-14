"""분봉 수집 — 손절 룰 검증용 (yfinance 5분봉).

yfinance는 5분봉을 최근 약 60일까지만 제공하므로, 매일 받아서 parquet에
append(중복 제거)하며 장기 이력을 직접 쌓는다. 인덱스는 KST tz-naive.

**알려진 커버리지 한계 (2026-08-14 실측)**: 하루치가 09:00~14:55 (72봉)뿐이고
**마감 30분(15:00~15:30, 정규장 종료 15:20 + 종가 동시호가)이 빠져 있다.**
소스의 제약이라 우회 불가. 영향 측정 결과 volbreak 트리거 도달일 138건 중
분봉에서 도달 봉을 못 찾은 건 1건(1%) — 돌파는 대개 장 초반에 터지기 때문이라
실질 영향은 작지만, 이 비율은 `fillcheck`가 매번 리포트에 찍는다(침묵 금지).
`intraday_window()`가 실제 커버 구간을 반환하므로 한계가 변하면 드러난다.
"""

from __future__ import annotations

import sys

import pandas as pd

from .config import DATA_DIR, load_config


def _cache_path():
    return DATA_DIR / load_config()["minute"]["cache"]


def load_minute() -> pd.DataFrame:
    path = _cache_path()
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _fetch_minute(symbol: str, interval: str) -> pd.DataFrame:
    """yfinance 분봉 다운로드 + KST 정규화 (60일 한도)."""
    import yfinance as yf

    df = yf.download(symbol, period="60d", interval=interval,
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Seoul").tz_localize(None)
    df.index = idx
    df.index.name = "dt"
    return df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]


def _merge_into(path, new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_parquet(path)
    return merged


def volbreak_minute_path(code: str):
    return DATA_DIR / f"minute_{code}_5m.parquet"


def load_volbreak_minute(code: str) -> pd.DataFrame:
    path = volbreak_minute_path(code)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def collect_minute() -> pd.DataFrame:
    """최근 구간 분봉을 받아 기존 캐시에 병합. 실패 시 기존 캐시 유지.

    주력종목(손절 검증용) + Stage 2 후보 ETF(체결·연구 재료)를 함께 수집한다.
    종목별로 독립 시도 — 하나가 실패해도 나머지는 진행 (침묵 유실 방지).
    """
    mcfg = load_config()["minute"]
    old = load_minute()
    try:
        df = _fetch_minute(mcfg["symbol"], mcfg["interval"])
        result = _merge_into(_cache_path(), df, old)
    except Exception as e:
        print(f"[warn] minute download failed ({e}); keeping cache", file=sys.stderr)
        result = old

    for code in mcfg.get("etf_codes", []):
        try:
            df = _fetch_minute(f"{code}.KS", mcfg["interval"])
            _merge_into(volbreak_minute_path(code), df, load_volbreak_minute(code))
        except Exception as e:   # 종목별 독립 — 실패는 경고만
            print(f"[warn] minute download failed for {code} ({e}); keeping cache",
                  file=sys.stderr)
    return result


def coverage(minute: pd.DataFrame) -> pd.DatetimeIndex:
    """분봉이 존재하는 거래일 목록."""
    if minute.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(set(minute.index.normalize())))


def intraday_window(minute: pd.DataFrame) -> dict:
    """실제로 커버되는 장중 구간 — {first, last, bars_median, days}.

    소스가 마감 30분을 안 주는 것(모듈 docstring)을 숫자로 드러내는 용도.
    한계가 변하면(예: 공급선 교체로 15:30까지 들어오면) 여기서 바로 보인다.
    """
    if minute.empty:
        return {"first": None, "last": None, "bars_median": 0, "days": 0}
    idx = pd.DatetimeIndex(minute.index)
    per_day = pd.Series(1, index=idx).groupby(idx.normalize()).sum()
    return {"first": min(idx.time), "last": max(idx.time),
            "bars_median": int(per_day.median()), "days": int(len(per_day))}
