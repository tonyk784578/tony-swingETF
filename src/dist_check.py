"""분배금 편향 진단 — 가격 시계열만 쓰는 백테스트의 채권 ETF 왜곡 정량화.

배경: 데이터 파이프라인은 미조정 가격을 쓴다. 주식형 ETF는 분배금이 작아
방향만 보수적(전략에 불리)이지만, KOSEF_KTB10Y 는 연말 분배가 회당 1.2~4.0%로
일변동(~0.3%) 대비 거대하다 — 분배락 하락이 (1) 종가<MA20 청산을 오발시키고
(2) 신고가 진입을 지연시키며 (3) 분배락을 걸쳐 보유한 트레이드의 수익을
분배금만큼 과소계상한다.

이 모듈은 **동결된 신호를 바꾸지 않는다** — 왜곡의 방향과 크기를 측정해 기록할
뿐이다 (판정 시 해석 자료). 수정주가로 신호를 다시 만들면 사후 변경이 된다.

분배금 이력: yfinance (KRX 배당락 공시 반영). data/ 에 캐시 — 오프라인 재실행 가능.
"""

from __future__ import annotations

import sys

import pandas as pd

from .config import DATA_DIR, RESULTS_DIR, load_config
from .data_loader import confirmed_cutoff
from .etf_swing import iter_candidates, simulate


def _load_dividends(code: str) -> pd.Series:
    """분배금 이력 (배당락일 index, 원). 캐시 우선, 실패 시 캐시로 degrade."""
    path = DATA_DIR / f"dividends_{code}.parquet"
    try:
        import yfinance as yf

        div = yf.Ticker(f"{code}.KS").dividends
        if len(div):
            div.index = div.index.tz_localize(None).normalize()
            div.to_frame("dividend").to_parquet(path)
    except Exception as e:  # noqa: BLE001 — 네트워크 실패는 캐시로 degrade
        print(f"  (분배금 다운로드 실패 — 캐시 사용: {e})", file=sys.stderr)
    if path.exists():
        s = pd.read_parquet(path)["dividend"]
        s.index = pd.DatetimeIndex(s.index)
        return s
    return pd.Series(dtype=float)


def run_dist_check() -> None:
    cfg = load_config()
    dcfg = cfg["etf"]["dist_check"]
    window = dcfg["exit_window_days"]

    lines = ["# 분배금 편향 진단 (가격 미조정 백테스트의 왜곡 정량화)", "",
             f"생성일: {pd.Timestamp.today().date()} | 설정: config `etf.dist_check` | "
             "동결 신호 불변 — 판정 시 해석 자료.", ""]

    for cand, df, entry, exit_, mh, tr in iter_candidates(False, cutoff=confirmed_cutoff()):
        code = str(cand["code"])
        if code not in dcfg["codes"]:
            continue
        name = f"{cand['name']} {cand['strategy']}"
        div = _load_dividends(code)
        print(f"=== {name} ===")
        if div.empty:
            print("  분배금 이력 없음/조회 실패 — 진단 불가")
            lines += [f"## {name}", "", "분배금 이력 조회 실패 — 진단 불가", ""]
            continue

        close = df["Close"]
        # 분배수익률: 배당락 전일 종가 대비
        yields = {}
        for d, amt in div.items():
            prev = close[close.index < d]
            if len(prev):
                yields[d] = amt / prev.iloc[-1]
        ydf = pd.Series(yields)
        years = (close.index[-1] - close.index[0]).days / 365.25
        ann_yield = ydf.sum() / years

        trades = simulate(df, entry, exit_, mh, cfg["etf"]["cost_round_trip"], trailing=tr)
        # (1) 분배락 직후 window 영업일 내 청산 — MA 이탈 오발 의심
        bd = pd.tseries.offsets.BusinessDay(window)
        suspect = trades[trades["exit_date"].apply(
            lambda x, _bd=bd, _idx=ydf.index: any(d <= x <= d + _bd for d in _idx))]
        rest = trades.drop(suspect.index)
        # (2) 분배락을 걸쳐 보유 — 실거래는 분배금 수령, 백테스트는 과소계상
        held_missed = []
        for _, t in trades.iterrows():
            for d, y in ydf.items():
                if t["entry_date"] < d <= t["exit_date"]:
                    held_missed.append(y)

        print(f"  분배 이력 {len(ydf)}건 | 회당 수익률 {ydf.min():.2%}~{ydf.max():.2%} "
              f"| 연평균 분배수익률 {ann_yield:.2%}")
        print(f"  (1) 분배락 후 {window}영업일 내 청산: {len(suspect)}/{len(trades)}건"
              + (f" — 평균 {suspect['net_ret'].mean():+.2%} vs 나머지 "
                 f"{rest['net_ret'].mean():+.2%}" if len(suspect) else ""))
        print(f"  (2) 분배락 걸친 보유 {len(held_missed)}건 — 과소계상 합계 "
              f"{sum(held_missed):+.2%} (트레이드당 평균 보정 "
              f"{sum(held_missed)/len(trades):+.3%})")
        adj_mean = trades["net_ret"].mean() + sum(held_missed) / len(trades)
        print(f"  ▶ 백테스트 평균 {trades['net_ret'].mean():+.3%} → 분배금 반영 시 "
              f"약 {adj_mean:+.3%} (방향: 과소평가 — 전략에 보수적)")

        lines += [
            f"## {name}", "",
            f"- 분배 이력 {len(ydf)}건, 회당 {ydf.min():.2%}~{ydf.max():.2%}, "
            f"연평균 분배수익률 {ann_yield:.2%} (기간 {years:.1f}년)",
            f"- **오발 청산 의심**: 분배락 후 {window}영업일 내 청산 "
            f"{len(suspect)}/{len(trades)}건"
            + (f" (평균 {suspect['net_ret'].mean():+.2%} vs 나머지 "
               f"{rest['net_ret'].mean():+.2%})" if len(suspect) else ""),
            f"- **수익 과소계상**: 분배락 걸친 보유 {len(held_missed)}건, 합계 "
            f"{sum(held_missed):+.2%} → 트레이드당 평균 "
            f"{sum(held_missed)/len(trades):+.3%} 보정",
            f"- 백테스트 평균 {trades['net_ret'].mean():+.3%} → 반영 시 약 {adj_mean:+.3%}."
            " 미조정 가격의 편향은 **전략에 불리한 방향**(보수적) — 판정 통과 시"
            " 실거래 성과가 장부보다 좋을 수 있는 요인으로 기록.",
            "- 진입 지연(분배락으로 낮아진 가격이 신고가 형성을 늦춤)은 정량화하지"
            " 않았다 — 이 역시 트레이드 수를 줄이는 보수적 방향.", ""]

    out = RESULTS_DIR / "kosef_dist_check.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport: {out}")
