"""KR-US 날짜 정렬 — 룩어헤드 방지의 핵심.

규칙: 한국 거래일 D의 매매 판단에 사용 가능한 정보
  (1) 미국 현지 날짜 < D 인 마지막 미국 일봉의 수익률
      (미국 날짜 D-1 봉은 한국시간 D일 새벽에 마감 → 사용 가능)
  (2) 한국 거래일 D-1까지의 국내 데이터 (KOSPI 수익률, 20일 이평 등)
  (3) 한국 거래일 D의 시가 (갭 계산용 — 진입 판단에만 사용)

미국 수익률은 각 심볼 자체 거래 캘린더에서 pct_change로 계산한 뒤 merge_asof로
매핑한다. union 인덱스에 ffill하는 방식은 휴장일에 "유령 0% 수익률"을 만들므로
사용하지 않는다. 미국 휴장 시에는 마지막 확정 봉의 수익률이 재사용된다.
"""

from __future__ import annotations

import pandas as pd


def us_returns(df: pd.DataFrame) -> pd.Series:
    """미국 심볼 자체 캘린더 기준 일간 종가 수익률 (인덱스 = 미국 현지 날짜)."""
    return df["Close"].pct_change().dropna()


def map_us_to_kr(us_ret: pd.Series, kr_dates: pd.DatetimeIndex, col: str) -> pd.DataFrame:
    """한국 거래일 D마다 '미국 날짜 < D'인 마지막 수익률을 매핑.

    반환: kr_dates 인덱스, [col, f"{col}_us_date"] 컬럼.
    """
    us = us_ret.rename(col).rename_axis("us_date").reset_index()
    us["us_date"] = us["us_date"].astype("datetime64[ns]")
    kr = pd.DataFrame({"date": kr_dates.astype("datetime64[ns]")}).sort_values("date")
    # merge_asof는 key <= 매칭이므로, 미국 날짜 < 한국 날짜(엄격)를 보장하기 위해
    # 한국 날짜에서 1일을 빼서 매칭한다 (D-1 이하 = D 미만).
    merged = pd.merge_asof(
        kr.assign(_key=kr["date"] - pd.Timedelta(days=1)),
        us.assign(_key=us["us_date"]),
        on="_key",
        direction="backward",
    )
    merged = merged.set_index("date")[[col, "us_date"]]
    return merged.rename(columns={"us_date": f"{col}_us_date"})


def build_master(data: dict[str, pd.DataFrame], stock_name: str) -> pd.DataFrame:
    """종목별 마스터 프레임 (인덱스 = 한국 거래일 D).

    컬럼:
      open, close                 : 종목 D 시가/종가
      day_ret                     : D 종가/시가 - 1 (결과 계산 전용 — 판단 사용 금지)
      gap                         : D 시가 / D-1 종가 - 1 (진입 판단에만 사용)
      prev_close, ma20_prev       : D-1 종가, D-1 기준 20일 이평
      above_ma20                  : D-1 종가 > D-1 이평 (D 정보 미포함)
      kospi_prev_ret              : KOSPI D-1 종가 수익률
      sox_ret / ixic_ret / nvda_ret + *_us_date : 정렬된 미국 수익률과 그 봉의 미국 날짜
    """
    stock = data[stock_name]
    m = pd.DataFrame(index=stock.index)
    m["open"] = stock["Open"]
    m["close"] = stock["Close"]
    m["day_ret"] = stock["Close"] / stock["Open"] - 1
    m["prev_close"] = stock["Close"].shift(1)
    m["gap"] = stock["Open"] / m["prev_close"] - 1
    ma20 = stock["Close"].rolling(20).mean()
    m["ma20_prev"] = ma20.shift(1)
    m["above_ma20"] = m["prev_close"] > m["ma20_prev"]

    # KOSPI D-1 수익률도 '날짜 < D인 마지막 봉' 방식으로 매핑 — shift+reindex는
    # KOSPI 데이터가 종목보다 하루 늦게 갱신될 때 최신일을 NaN으로 탈락시킨다.
    kospi_ret = data["KOSPI"]["Close"].pct_change().dropna()
    m["kospi_prev_ret"] = map_us_to_kr(kospi_ret, m.index, "kospi_prev_ret")["kospi_prev_ret"]

    for us_name, col in [("SOX", "sox_ret"), ("NASDAQ", "ixic_ret"), ("NVDA", "nvda_ret")]:
        mapped = map_us_to_kr(us_returns(data[us_name]), m.index, col)
        m = m.join(mapped)

    # 워밍업 구간(이평/전일값 결측) 및 미국 데이터 없는 초기 행 제거
    m = m.dropna(subset=["prev_close", "ma20_prev", "kospi_prev_ret", "sox_ret"])
    return m


def verify_alignment(master: pd.DataFrame, stock_name: str) -> bool:
    """정렬 전수 검증 + 샘플 날짜 출력. 실패 시 False."""
    ok = True
    for col in ["sox_ret", "ixic_ret", "nvda_ret"]:
        us_dates = master[f"{col}_us_date"]
        violation = us_dates >= master.index.to_series()
        if violation.any():
            print(f"[FAIL] {stock_name} {col}: {violation.sum()} rows with US date >= KR date")
            ok = False

    # 미국 봉이 비정상적으로 오래된 행 (연휴 감안 7일 초과) 리포트
    lag = (master.index.to_series() - master["sox_ret_us_date"]).dt.days
    stale = lag[lag > 7]
    if not stale.empty:
        print(f"[info] {stock_name}: {len(stale)} rows where mapped US bar is >7 days old "
              f"(long holidays), max lag {stale.max()} days on {stale.idxmax().date()}")

    print(f"\n=== {stock_name}: alignment samples (KR date -> mapped US SOX bar date) ===")
    samples = []
    for target in ["2023-10-04",   # 추석+연휴 직후
                   "2023-11-24",   # 미국 Thanksgiving(11/23) 다음 KR 거래일
                   "2024-07-05",   # 미국 독립기념일(7/4) 다음 KR 거래일
                   "2024-01-02",   # 연초
                   ]:
        t = pd.Timestamp(target)
        pos = master.index.searchsorted(t)
        if pos < len(master):
            samples.append(master.index[pos])
    # 일반 평일 + 월요일(금요일 봉 매핑 확인) — 최근 10일에 월요일이 없을 수 있다
    recent = master.index[-10:]
    mondays = recent[recent.dayofweek == 0]
    if len(mondays):
        samples.append(mondays[-1])
    samples.append(master.index[-1])                   # 마지막 거래일

    for d in samples:
        row = master.loc[d]
        print(f"  KR {d.date()} ({d.day_name()[:3]}): "
              f"SOX bar {row['sox_ret_us_date'].date()} ({row['sox_ret']:+.2%}), "
              f"NVDA bar {row['nvda_ret_us_date'].date()} ({row['nvda_ret']:+.2%}), "
              f"KOSPI prev {row['kospi_prev_ret']:+.2%}, gap {row['gap']:+.2%}")

    if ok:
        print(f"[OK] {stock_name}: all {len(master)} rows satisfy US date < KR date")
    return ok
