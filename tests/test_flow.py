"""flow 전략 테스트 — 등록된 규칙(양전 크로스·청산 1일 지연·룩어헤드)이 정확한지."""

import pandas as pd

from src.etf_swing import build_flags, raw_entry_signal, simulate
from src.flow_data import parse_flow_page


def _df(n, start="2026-03-02"):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0},
                        index=idx)


def _flow(df, values):
    return pd.Series(values, index=df.index, dtype=float)


def test_entry_on_positive_cross_next_open():
    # 5일 롤링합이 음수(-5)에서 +10 유입으로 양전 크로스 → 다음날 시가 진입.
    # 크로스 이전에 유효한(워밍업 지난) 음수 롤링합이 있어야 한다
    df = _df(9)
    flow = _flow(df, [-1, -1, -1, -1, -1, 10, 1, 1, 1])
    raw = raw_entry_signal(df, "flow", flow)
    assert raw.fillna(False).tolist().index(True) == 5       # 크로스는 D(6일차)
    entry, exit_, max_hold = build_flags(df, "flow", flow)
    assert entry.fillna(False).tolist().index(True) == 6     # 진입은 D+1


def test_no_entry_without_cross():
    # 롤링합이 처음부터 계속 양수 — 크로스가 없으므로 진입 없음
    df = _df(8)
    raw = raw_entry_signal(df, "flow", _flow(df, [1] * 8))
    assert not raw.fillna(False).any()


def test_exit_lagged_one_day():
    # 청산 플래그(D)는 롤링합(D-1)<0 — D 종가 시점엔 D의 확정 수급을 모른다
    df = _df(12)
    flow = _flow(df, [-1, -1, -1, -1, -1, 10, -20, -1, -1, -1, -1, -1])
    entry, exit_, max_hold = build_flags(df, "flow", flow)
    # 7일차(-20)에 롤링합 음전 → 청산 플래그는 8일차(지연 1일)
    neg_day = 6
    assert not exit_.fillna(False).iloc[neg_day]
    assert exit_.fillna(False).iloc[neg_day + 1]
    trades = simulate(df, entry, exit_, max_hold, cost=0.0)
    assert len(trades) == 1
    assert trades.iloc[0]["entry_date"] == df.index[6]
    assert trades.iloc[0]["exit_date"] == df.index[7]


def test_no_lookahead_future_flow_change():
    # 미래 수급을 바꿔도 이미 완결된 앞쪽 트레이드는 불변
    df = _df(14)
    base = [-1, -1, -1, -1, -1, 10, -20, -1, -1, -1, -1, -1, -1, -1]
    alt = base[:9] + [99, 99, 99, 99, 99]
    e1, x1, mh = build_flags(df, "flow", _flow(df, base))
    e2, x2, _ = build_flags(df, "flow", _flow(df, alt))
    t1 = simulate(df, e1, x1, mh, cost=0.0)
    t2 = simulate(df, e2, x2, mh, cost=0.0)
    pd.testing.assert_frame_equal(t1.iloc[:1], t2.iloc[:1])


def test_max_hold_cap_applies():
    # 롤링합이 계속 양수면 하우스 캡(10일)에서 강제 청산
    df = _df(18)
    flow = _flow(df, [-1, -1, -1, -1, -1, 10] + [1] * 12)
    entry, exit_, max_hold = build_flags(df, "flow", flow)
    trades = simulate(df, entry, exit_, max_hold, cost=0.0)
    assert len(trades) >= 1
    assert trades.iloc[0]["hold"] == max_hold


def test_parse_flow_page_multiheader():
    # 네이버 표 구조(멀티헤더 + 기관 세부열) 파싱 — 구조 변경 시 이 테스트가 깨진다
    html = """<table>
    <tr><th rowspan=2>날짜</th><th rowspan=2>개인</th><th rowspan=2>외국인</th>
        <th rowspan=2>기관계</th><th colspan=2>기관</th><th rowspan=2>기타법인</th></tr>
    <tr><th>금융투자</th><th>연기금등</th></tr>
    <tr><td>26.08.25</td><td>100</td><td>-200</td><td>50</td><td>30</td><td>20</td><td>50</td></tr>
    <tr><td>26.08.26</td><td>10</td><td>300</td><td>-310</td><td>-300</td><td>-10</td><td>0</td></tr>
    </table>"""
    out = parse_flow_page(html)
    assert list(out.columns) == ["individual", "foreign", "institution"]
    assert out.index[0] == pd.Timestamp("2026-08-25")
    assert out["foreign"].tolist() == [-200.0, 300.0]
