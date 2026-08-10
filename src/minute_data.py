"""분봉 수집 — 손절 룰 검증용 (yfinance 5분봉).

yfinance는 5분봉을 최근 약 60일까지만 제공하므로, 매일 받아서 parquet에
append(중복 제거)하며 장기 이력을 직접 쌓는다. 인덱스는 KST tz-naive.
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

    주력종목(손절 검증용) + volbreak ETF 6종(체결 검증용)을 함께 수집한다.
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

    for code in mcfg.get("volbreak_codes", []):
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
