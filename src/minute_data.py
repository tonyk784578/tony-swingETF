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


def collect_minute() -> pd.DataFrame:
    """최근 구간 분봉을 받아 기존 캐시에 병합. 실패 시 기존 캐시 유지."""
    import yfinance as yf

    mcfg = load_config()["minute"]
    old = load_minute()
    try:
        df = yf.download(mcfg["symbol"], period="60d", interval=mcfg["interval"],
                         auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        idx = pd.to_datetime(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("Asia/Seoul").tz_localize(None)
        df.index = idx
        df.index.name = "dt"
        df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    except Exception as e:
        print(f"[warn] minute download failed ({e}); keeping cache", file=sys.stderr)
        return old

    merged = pd.concat([old, df])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_parquet(_cache_path())
    return merged


def coverage(minute: pd.DataFrame) -> pd.DatetimeIndex:
    """분봉이 존재하는 거래일 목록."""
    if minute.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(set(minute.index.normalize())))
