"""데이터 다운로드 + parquet 캐시.

- 국내 종목/지수: FinanceDataReader
- 미국 지수/종목: yfinance — 인덱스를 미국 현지 날짜(tz-naive)로 정규화해 저장.
  미국 날짜 D의 봉은 한국시간 D+1 새벽에 마감되므로, 저장 단계에서는 현지 날짜만
  보존하고 사용 가능 시점 판단은 align.py가 담당한다.
- 다운로드 실패 시 기존 캐시가 있으면 경고 후 캐시 사용 (오프라인 재실행 가능).
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from .config import DATA_DIR, load_config

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9^=._-]+$")


def _safe_name(symbol: str) -> str:
    if not _SYMBOL_RE.match(symbol):
        raise ValueError(f"unsafe symbol name: {symbol!r}")
    return symbol.replace("^", "").replace("=", "_")


def confirmed_cutoff() -> pd.Timestamp:
    """확정 일봉 컷오프 — 16시(KST) 이전에는 당일 봉이 미완성이므로 전일까지.

    장부(paper/etf_paper)와 리서치(screening/portfolio/refine)가 모두 이 하나를
    쓴다. 시간 민감 로직은 한 곳에만 존재해야 한다.
    """
    now = pd.Timestamp.now(tz="Asia/Seoul")
    cut = now.normalize().tz_localize(None)
    if now.hour < 16:
        cut -= pd.Timedelta(days=1)
    return cut


def _cache_path(symbol: str):
    return DATA_DIR / f"{_safe_name(symbol)}.parquet"


def _cache_fresh(symbol: str, max_age_days: int) -> bool:
    path = _cache_path(symbol)
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:
        return False
    if df.empty:
        return False
    age = (pd.Timestamp.today().normalize() - df.index.max()).days
    return age <= max_age_days


def _fetch_kr(symbol: str, start: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    df = fdr.DataReader(symbol, start)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    return df


def _fetch_kr_fallback(symbol: str, start: str, cache: pd.DataFrame | None
                       ) -> pd.DataFrame:
    """KR 예비 공급선 — FDR 장애 시 yfinance({code}.KS)의 **최신 봉만** 이어붙임.

    3년 무인 관측에서 단일 소스는 최대 구조 위험이다: FDR이 몇 주 멈추면
    경고는 떠도 아침 신호가 유실된다.

    전체 대체가 아니라 append 인 이유(실측): yfinance KR 가격은 과거 구간이
    분배금 조정 기준이라 FDR 미조정가와 최대 25%+ 어긋난다 — 통째로 바꾸면
    과거 신호가 전부 뒤틀린다. 최신 봉(마지막 분배 이후)은 두 소스가 일치하므로
    캐시 끝 이후만 붙인다. 캐시가 아예 없으면 yfinance 전체를 쓰되 큰 경고를
    남긴다(기준이 다른 데이터임을 인지해야 함). 종목 지수(KS11)는 야후 티커
    체계가 달라 미지원 — 호출부가 캐시로 degrade 한다.
    """
    if not symbol.isdigit():
        raise ValueError(f"yfinance KR fallback unsupported for {symbol!r}")
    df = _fetch_us(f"{symbol}.KS", start)   # 다운로드·정규화 로직 재사용
    if df.empty:
        raise RuntimeError(f"{symbol}.KS: yfinance returned empty")
    if cache is None or cache.empty:
        print(f"[warn] {symbol}: no cache — using FULL yfinance history "
              "(주의: 분배금 조정 기준이 FDR과 다름)", file=sys.stderr)
        return df
    new = df[df.index > cache.index.max()]
    print(f"[warn] {symbol}: yfinance fallback appended {len(new)} recent bar(s) "
          "to FDR cache", file=sys.stderr)
    return pd.concat([cache, new])


def _fetch_us(symbol: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, start=start, interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    # yfinance 일봉 인덱스는 거래소 현지 날짜 (tz 정보가 있으면 제거)
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    return df


def _check_revisions(symbol: str, old: pd.DataFrame, new: pd.DataFrame) -> None:
    """과거 데이터가 소리 없이 바뀌었는지 감시 (액면분할·데이터 정정 등).

    겹치는 과거 구간(양쪽 모두 존재, 최신 3일 제외)에서 종가가 0.1% 넘게
    다르면 경고 + data/revisions.log 에 기록한다. 갭/수익률 계산 왜곡의
    주범이므로, 이 경고가 뜨면 백테스트·장부를 재검토해야 한다.
    """
    common = old.index.intersection(new.index)
    common = common[common < common.max() - pd.Timedelta(days=3)]
    if len(common) == 0:
        return
    # Open도 감시 — 시가만 정정돼도 gap/day_ret이 왜곡된다
    rel = pd.concat([(new.loc[common, c] / old.loc[common, c] - 1).abs()
                     for c in ["Open", "Close"]], axis=1).max(axis=1)
    changed = rel[rel > 0.001]
    if changed.empty:
        return
    first, last = changed.index.min().date(), changed.index.max().date()
    msg = (f"[REVISION] {symbol}: {len(changed)} past rows changed "
           f"(max {changed.max():.2%}, range {first}~{last}) — "
           f"백테스트/장부 재검토 필요 (액면분할·데이터 정정 의심)")
    print(msg, file=sys.stderr)
    with open(DATA_DIR / "revisions.log", "a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now():%F %T} {msg}\n")


def load_symbol(symbol: str, kind: str, force: bool = False) -> pd.DataFrame:
    """단일 심볼 로드. kind: 'kr' | 'us'. 캐시 우선, 실패 시 캐시로 degrade."""
    cfg = load_config()["data"]
    path = _cache_path(symbol)

    if not force and _cache_fresh(symbol, cfg["cache_max_age_days"]):
        return pd.read_parquet(path)

    try:
        if kind == "kr":
            try:
                df = _fetch_kr(symbol, cfg["start"])
            except Exception as e:
                print(f"[warn] {symbol} FDR failed ({e}); trying yfinance fallback",
                      file=sys.stderr)
                cache = pd.read_parquet(path) if path.exists() else None
                df = _fetch_kr_fallback(symbol, cfg["start"], cache)
        else:
            df = _fetch_us(symbol, cfg["start"])
        if path.exists():
            try:
                _check_revisions(symbol, pd.read_parquet(path), df)
            except Exception as e:
                print(f"[warn] {symbol} revision check failed: {e}", file=sys.stderr)
        # OHLC에 0이나 결측이 있는 봉(거래정지 등)은 제거
        ohlc = df[["Open", "High", "Low", "Close"]]
        df = df[(ohlc > 0).all(axis=1) & ohlc.notna().all(axis=1)]
        if df.empty:
            raise RuntimeError(f"{symbol}: empty data returned")
        df.to_parquet(path)
        return df
    except Exception as e:
        if path.exists():
            print(f"[warn] {symbol} download failed ({e}); using stale cache", file=sys.stderr)
            return pd.read_parquet(path)
        raise


def load_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """전체 심볼 로드. 반환 키는 config의 표시 이름 (예: 'SamsungElec', 'SOX')."""
    cfg = load_config()["data"]
    out: dict[str, pd.DataFrame] = {}
    for symbol, name in cfg["kr_stocks"].items():
        out[name] = load_symbol(str(symbol), "kr", force)
    for symbol, name in cfg["kr_index"].items():
        out[name] = load_symbol(str(symbol), "kr", force)
    for symbol, name in cfg["us_symbols"].items():
        out[name] = load_symbol(str(symbol), "us", force)
    return out


def summarize(data: dict[str, pd.DataFrame]) -> str:
    lines = []
    for name, df in data.items():
        lines.append(
            f"{name:16s} rows={len(df):5d}  {df.index.min().date()} ~ {df.index.max().date()}"
        )
    return "\n".join(lines)
