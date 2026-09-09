"""체결 검증 — 표본 탈락 집계와 분봉 커버리지 진단.

핵심 관심사: 일봉상 트리거 도달인데 분봉에 도달 봉이 없으면 그 날은 표본에서
빠진다. 조용히 빠지면 슬리피지 추정이 낙관 편향되므로, 탈락이 '집계는 되되
평균에는 안 들어가는' 구조인지 합성 데이터로 고정한다.
"""

import pandas as pd

from src.fill_check import _day_fills
from src.minute_data import align_minute_to_daily, intraday_window


def _daily(highs, opens=None, closes=None):
    n = len(highs)
    idx = pd.bdate_range("2026-06-01", periods=n)
    opens = opens or [100.0] * n
    return pd.DataFrame({"Open": opens, "High": highs,
                         "Low": [99.0] * n, "Close": closes or [100.0] * n}, index=idx)


def _bars(day, times, highs, opens=None, closes=None):
    idx = pd.to_datetime([f"{day} {t}" for t in times])
    n = len(times)
    return pd.DataFrame({"Open": opens or [100.0] * n, "High": highs,
                         "Low": [99.0] * n, "Close": closes or list(highs)}, index=idx)


def test_day_fills_marks_missing_bar_days_instead_of_dropping_them():
    """분봉에 도달 봉이 없는 날도 행으로 남아야 한다 (탈락 집계를 위해).

    날짜 4개인 이유: 첫날은 전일 범위가 없어 트리거 NaN, 마지막 날은 다음 시가가
    없어 제외 — 실제로 채점되는 건 가운데 2일이다.
    """
    daily = _daily([100.0, 105.0, 105.0, 100.0])
    # 06-02는 09:00 봉이 트리거(100.5) 위 → 체결가 확보.
    # 06-03은 트리거가 103.0인데 주어진 봉의 고가가 100뿐 — 도달이 마감 30분 안에
    # 일어나 소스가 그 봉을 주지 않은 상황 재현.
    minute = pd.concat([
        _bars("2026-06-02", ["09:00", "09:05"], [101.0, 101.0]),
        _bars("2026-06-03", ["09:00", "09:05"], [100.0, 100.0]),
    ])
    f = _day_fills(daily, minute, k=0.5)
    assert len(f) == 2                              # 두 도달일 모두 행으로 남음
    assert f["cons_fill"].notna().sum() == 1        # 체결가를 얻은 건 하루뿐
    assert f["cons_fill"].isna().sum() == 1         # 나머지는 탈락으로 표시


def test_day_fills_excludes_last_bar_as_incomplete():
    """마지막 봉은 다음날 시가(청산가)가 없어 미완성 — 표본에 넣으면 안 된다."""
    daily = _daily([100.0, 105.0])                  # 채점 가능한 날이 없음
    minute = _bars("2026-06-02", ["09:00"], [101.0])
    assert _day_fills(daily, minute, k=0.5).empty


def test_day_fills_skips_days_without_any_minute_data():
    """분봉 자체가 없는 날은 도달 여부와 무관하게 표본 밖 (탈락도 아님)."""
    daily = _daily([100.0, 105.0, 100.0])
    f = _day_fills(daily, pd.DataFrame(), k=0.5)
    assert f.empty


def test_day_fills_flags_gap_open_when_bar_opens_above_trigger():
    daily = _daily([100.0, 105.0, 100.0])
    minute = _bars("2026-06-02", ["09:00"], [102.0], opens=[101.0])   # 시가부터 위
    f = _day_fills(daily, minute, k=0.5)
    assert bool(f["gap_open"].iloc[0])


def test_conservative_fill_is_the_touch_bar_close():
    daily = _daily([100.0, 105.0, 100.0])
    minute = _bars("2026-06-02", ["09:00", "09:05"], [100.4, 101.0],
                   closes=[100.4, 100.8])
    f = _day_fills(daily, minute, k=0.5)             # 트리거 100.5 → 09:05 봉이 최초 도달
    assert f["cons_fill"].iloc[0] == 100.8
    assert f["model_fill"].iloc[0] == 100.5          # max(시가 100, 트리거 100.5)


# ------------------------------------------------------------- 커버리지 진단

def test_intraday_window_reports_actual_covered_span():
    """소스가 마감 30분을 안 주는 한계를 숫자로 드러내는 함수."""
    m = pd.concat([_bars("2026-06-02", ["09:00", "14:55"], [1.0, 1.0]),
                   _bars("2026-06-03", ["09:00", "14:55"], [1.0, 1.0])])
    w = intraday_window(m)
    assert str(w["first"]) == "09:00:00"
    assert str(w["last"]) == "14:55:00"              # 15:30 이 아님 = 알려진 한계
    assert w["bars_median"] == 2 and w["days"] == 2


def test_intraday_window_handles_empty_input():
    w = intraday_window(pd.DataFrame())
    assert w["days"] == 0 and w["first"] is None


# ------------------------------------------------------------- 가격 기준 정렬

def test_align_minute_rescales_by_open_ratio_per_day():
    """분봉(원가격)을 일봉(분배금 조정가) 배율로 맞춘다 — 날짜별 시가 비율.

    2026-09-09 실측: 분배락 이전 날짜의 FDR 일봉이 원가격보다 분배수익률만큼 낮아
    정렬 없이는 그 차이가 슬리피지로 잡혔다 (KODEX_Securities 0.85%p).
    """
    daily = _daily([100.0, 100.0], opens=[99.0, 100.0])          # 첫날은 조정가(1% 낮음)
    minute = pd.concat([
        _bars("2026-06-01", ["09:00", "09:05"], [101.0, 102.0], opens=[100.0, 100.0]),
        _bars("2026-06-02", ["09:00"], [101.0], opens=[100.0]),
    ])
    a = align_minute_to_daily(minute, daily)
    assert abs(a.loc["2026-06-01 09:05", "High"] - 102.0 * 0.99) < 1e-9
    assert a.loc["2026-06-02 09:00", "High"] == 101.0               # 배율 1 (일치)
    assert (minute["High"] == [101.0, 102.0, 101.0]).all()           # 원본 불변


def test_align_minute_leaves_day_alone_when_first_bar_is_not_open_or_ratio_wild():
    daily = _daily([100.0, 100.0], opens=[50.0, 100.0])           # 50% 차이 = 비정상
    minute = pd.concat([
        _bars("2026-06-01", ["09:00"], [101.0], opens=[100.0]),
        _bars("2026-06-02", ["09:10"], [101.0], opens=[100.0]),     # 첫 봉이 09:10
    ])
    a = align_minute_to_daily(minute, daily)
    assert (a["High"] == minute["High"]).all()
