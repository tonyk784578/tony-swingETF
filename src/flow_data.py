"""외국인 수급(투자자별 매매동향) 수집 — flow 전략 데이터 공급선.

공급선 (PREREG_flow.md §4, 2026-08-26 실측 근거):
- KRX 공식(pykrx)은 익명 API 차단("LOGOUT" 응답)으로 불가 — 로그인 계정 확보 시
  교체 가능 (같은 KRX 원천 수치라 신호 정의 불변).
- 채택: 네이버 금융 투자자별 매매동향(일별) — KRX/코스콤 원천 확정치(억원),
  2009년 이후 KOSPI/KOSDAQ 일별 제공 실측 확인.

시점 방어 (룩어헤드 방지):
- 공급 페이지는 당일 장중에 잠정 행을 노출한다 → 저장·로드 모두
  `confirmed_cutoff()`(16시) 이전의 당일 행을 버린다.
- 잠정→확정 정정: 매 갱신 시 최근 5영업일 창을 재수집해 덮어쓴다.
  그보다 오래된 캐시는 불변 (가격 캐시 정정 감시 관례와 정합).

내성: 수급은 소급 수집 가능 — 공급이 끊겨도 리플레이 장부는 무손실이고
프리뷰 신호만 늦는다. 수집 실패는 [warn] + 캐시 degrade (오프라인 재실행 가능).
"""

from __future__ import annotations

import io
import sys
import time

import pandas as pd
import requests

from .config import DATA_DIR, load_config

_URL = "https://finance.naver.com/sise/investorDealTrendDay.naver"
_SOSOK = {"KOSPI": "01", "KOSDAQ": "02"}
_HEADERS = {"User-Agent": "Mozilla/5.0"}
FLOW_START = "2009-01-02"   # 홀드아웃 구간(2009~)까지 한 캐시로 — 소스가 같아 분리 불필요
_REFETCH_BDAYS = 5          # 잠정→확정 흡수용 재수집 창


def flow_path(market: str):
    return DATA_DIR / f"flow_{market}.parquet"


def parse_flow_page(html: str) -> pd.DataFrame:
    """매매동향 페이지 HTML → 일별 순매매(억원) 프레임 (date 인덱스, 과거→최신)."""
    tables = pd.read_html(io.StringIO(html))
    t = next(t for t in tables if "날짜" in t.columns.get_level_values(0))
    t = t.loc[:, ~t.columns.get_level_values(0).duplicated()]   # 기관 세부열 제거
    t.columns = t.columns.get_level_values(0)
    t = t.dropna(subset=["날짜"])
    out = pd.DataFrame({
        "individual": t["개인"].astype(float),
        "foreign": t["외국인"].astype(float),
        "institution": t["기관계"].astype(float),
    })
    out.index = pd.to_datetime(t["날짜"], format="%y.%m.%d")
    out.index.name = "date"
    return out.sort_index()


def _fetch_range(market: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """bizdate 역방향 페이지네이션 — 페이지 마지막(가장 오래된) 날짜 앞으로 이동."""
    frames = []
    cursor = end
    while cursor >= start:
        r = requests.get(_URL, params={"bizdate": cursor.strftime("%Y%m%d"),
                                       "sosok": _SOSOK[market]},
                         headers=_HEADERS, timeout=15)
        r.raise_for_status()
        page = parse_flow_page(r.text)
        if page.empty:
            break
        frames.append(page)
        oldest = page.index.min()
        if oldest >= cursor:   # 진전 없음 — 공급 끝
            break
        cursor = oldest - pd.Timedelta(days=1)
        time.sleep(0.15)       # 공개 페이지 예의 — 부하 최소화
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[(out.index >= start) & (out.index <= end)]


def load_flow(market: str) -> pd.DataFrame:
    """캐시된 일별 수급 (읽기 전용 — 갱신은 update_flow_caches). 확정 봉만."""
    from .data_loader import confirmed_cutoff

    path = flow_path(market)
    if not path.exists():
        return pd.DataFrame(columns=["individual", "foreign", "institution"])
    df = pd.read_parquet(path)
    return df[df.index <= confirmed_cutoff()]   # 이중 방어 — 잠정 당일 행 차단


def update_flow_caches() -> None:
    """flow 전략이 참조하는 전 시장 수급 캐시 갱신 (download 단계에서 호출).

    캐시 없으면 2009년부터 전체 백필, 있으면 최근 창만 재수집(잠정→확정 흡수).
    실패는 [warn] + degrade — 다운로드 실패가 파이프라인을 멈추지 않는다.
    """
    from .data_loader import confirmed_cutoff

    fcfg = load_config()["etf"]["strategies"].get("flow")
    if not fcfg:
        return
    cut = confirmed_cutoff()
    for market in sorted(set(fcfg["markets"].values())):
        path = flow_path(market)
        cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        try:
            if cached.empty:
                print(f"[flow] {market} 전체 백필 ({FLOW_START}~) — 최초 1회, 수 분 소요",
                      file=sys.stderr)
                fetched = _fetch_range(market, pd.Timestamp(FLOW_START), cut)
            else:
                refetch_from = cached.index.max() - pd.tseries.offsets.BDay(_REFETCH_BDAYS)
                fetched = _fetch_range(market, refetch_from, cut)
            if fetched.empty:
                raise RuntimeError("빈 응답 — 페이지 구조 변경 가능성")
            keep = cached[cached.index < fetched.index.min()] if not cached.empty \
                else cached
            merged = pd.concat([keep, fetched]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged[merged.index <= cut]   # 장중 잠정 행 차단
            merged.to_parquet(path)
            print(f"[flow] {market}: {len(merged)}일 "
                  f"({merged.index.min().date()}~{merged.index.max().date()})",
                  file=sys.stderr)
        except Exception as e:   # noqa: BLE001 — 공급 실패는 degrade가 사양
            print(f"[warn] flow {market} 수집 실패 — 캐시 degrade "
                  f"({len(cached)}일 보유): {e}", file=sys.stderr)
