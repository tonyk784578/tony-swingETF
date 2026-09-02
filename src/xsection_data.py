"""횡단면 데이터 파이프라인 — KR 전 종목 일봉 (상폐 포함, 생존편향 제거).

PREREG_xsection.md §2 데이터 규정의 구현. 캐시는 data/xs/ 로 라이브 캐시와
완전 분리 (holdout 관례). 유니버스 = 현 상장 + start 이후 상폐 주권.

- 보통주만: 코드 끝자리 '0' + 이름 패턴 제외 (config xsection)
- 상폐 종목은 상폐일까지의 봉이 소스에 남아 있음 (2026-09-03 실측 3/3)
- 연구 캐시라 정정 감시는 두지 않는다 (라이브 판정 장부와 무관 — 한계는
  PREREG 에 명시). 재백필은 --force.

실행: python -m src.main xdata [--force]
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from .config import DATA_DIR, load_config

XS_DIR = DATA_DIR / "xs"


def _xcfg() -> dict:
    return load_config()["xsection"]


def xs_universe() -> pd.DataFrame:
    """유니버스 목록: [code, name, market, delisted, end]. 성과 미참조 규칙만."""
    import FinanceDataReader as fdr

    cfg = _xcfg()
    start = pd.Timestamp(cfg["start"])
    markets = set(cfg["markets"])
    patterns = cfg["exclude_name_patterns"]

    cur = fdr.StockListing("KRX")
    cur = cur[cur["Market"].isin(markets)]
    rows = [{"code": str(r["Code"]), "name": r["Name"], "market": r["Market"],
             "delisted": False, "end": None} for _, r in cur.iterrows()]

    dl = fdr.StockListing("KRX-DELISTING")
    dl["DelistingDate"] = pd.to_datetime(dl["DelistingDate"])
    dl = dl[(dl["SecuGroup"] == "주권") & (dl["DelistingDate"] >= start)
            & (dl["Market"].isin(markets))]
    rows += [{"code": str(r["Symbol"]), "name": r["Name"], "market": r["Market"],
              "delisted": True, "end": r["DelistingDate"]} for _, r in dl.iterrows()]

    uni = pd.DataFrame(rows).drop_duplicates("code", keep="first")
    uni = uni[uni["code"].str.len().eq(6) & uni["code"].str.endswith("0")]  # 보통주
    mask = ~uni["name"].str.contains("|".join(patterns), na=False)
    return uni[mask].reset_index(drop=True)


def _path(code: str):
    return XS_DIR / f"{code}.parquet"


def load_xs(code: str) -> pd.DataFrame:
    p = _path(code)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def backfill(force: bool = False) -> dict:
    """전 유니버스 일봉 백필 — 종목별 독립 실패 처리 (minute 수집 관례).

    이미 받은 종목은 건너뛴다(멱등). 상장 중 종목의 증분 갱신은 섀도 개시
    단계에서 붙인다 — Stage 1 백테스트는 1회 스냅샷이면 충분.
    """
    import FinanceDataReader as fdr

    cfg = _xcfg()
    start = cfg["start"]
    XS_DIR.mkdir(parents=True, exist_ok=True)
    uni = xs_universe()
    done = skipped = failed = empty = 0
    t0 = time.time()
    for i, r in uni.iterrows():
        p = _path(r["code"])
        if p.exists() and not force:
            skipped += 1
            continue
        try:
            end = r["end"].strftime("%Y-%m-%d") if r["delisted"] else None
            df = fdr.DataReader(r["code"], start, end)
            if df is None or df.empty:
                empty += 1
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
            df.to_parquet(p)
            done += 1
        except Exception as e:   # noqa: BLE001 — 종목별 독립 실패
            failed += 1
            print(f"[warn] xs {r['code']} {r['name']}: {type(e).__name__} "
                  f"{str(e)[:80]}", file=sys.stderr)
        time.sleep(0.05)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(uni)} ({time.time() - t0:.0f}s, "
                  f"신규 {done} 스킵 {skipped} 실패 {failed})", flush=True)
    stats = {"universe": len(uni), "downloaded": done, "skipped": skipped,
             "failed": failed, "empty": empty,
             "delisted_in_universe": int(uni["delisted"].sum())}
    print(f"xdata 완료: {stats}")
    return stats


def coverage_report() -> pd.DataFrame:
    """백필 커버리지 감사 (PREREG §9) — 유니버스 대비 캐시 보유율·기간."""
    uni = xs_universe()
    rows = []
    for _, r in uni.iterrows():
        df = load_xs(r["code"])
        rows.append({"code": r["code"], "name": r["name"], "delisted": r["delisted"],
                     "rows": len(df),
                     "first": df.index.min() if len(df) else None,
                     "last": df.index.max() if len(df) else None})
    cov = pd.DataFrame(rows)
    have = cov[cov["rows"] > 0]
    print(f"커버리지: {len(have)}/{len(cov)}종목 "
          f"({len(have) / len(cov):.1%}) | 상폐 포함 "
          f"{int(cov[cov['delisted']]['rows'].gt(0).sum())}/"
          f"{int(cov['delisted'].sum())}")
    return cov
