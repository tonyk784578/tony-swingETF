"""횡단면 ML 랭킹 — Stage 1 백테스트 (PREREG_xsection.md 의 구현, 1회 실행 판정).

시점 규율 (등록 §5):
- 신호일 S = 리밸런스 실행일 E 직전 거래일(마스터 캘린더 = KS200).
  피처는 S 종가까지의 그 종목 자체 거래일 시계열로 계산.
- 진입 E 시가, 학습 라벨 = E 시가 → E+5 마스터 거래일 시가 (날짜 내 랭크).
- 포트폴리오 보유는 다음 리밸런스일 E'까지 (라벨 지평과 구분 — 등록 §6).
- 상폐/정지로 청산가가 없으면 마지막 거래 종가로 처분 가정 (등록 §2 한계).
- 워크포워드 연도 Y 학습 표본은 라벨 종료일 X 가 Y 첫 예측 E 이전인 주만
  (라벨 창 침범 차단).

실행: python -m src.main xsection  (Stage 1 — 결과가 나온 뒤의 재실행은
리포트 재생성일 뿐, 설정 변경 후 재실행은 등록 위반)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, load_config
from .xsection_data import XS_DIR, load_xs, xs_universe


def _xcfg() -> dict:
    return load_config()["xsection"]


def master_calendar(force: bool = False) -> pd.DataFrame:
    """마스터 캘린더 + 헤지 지수 = KS200 일봉 (캐시 data/xs/_KS200.parquet)."""
    p = XS_DIR / "_KS200.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    import FinanceDataReader as fdr

    df = fdr.DataReader(_xcfg()["hedge"]["index"], _xcfg()["start"])
    df = df[["Open", "Close"]].dropna()
    XS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return df


def rebalance_days(cal: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """주 첫 거래일 목록 (ISO 연-주 단위). 첫 주는 신호일이 없어 제외."""
    s = pd.Series(cal, index=cal)
    iso = cal.isocalendar()
    days = sorted(s.groupby([iso.year, iso.week]).min().tolist())
    return [d for d in days if cal.get_loc(d) >= 1]


def symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """등록 §4 피처 10종 — 종목 자체 거래일 기준, S 종가까지만."""
    c, o, h, v = df["Close"], df["Open"], df["High"], df["Volume"]
    amt = (c * v).replace(0, np.nan)
    ret = c.pct_change()
    f = pd.DataFrame(index=df.index)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["mom_6_1"] = c.shift(21) / c.shift(126) - 1
    f["rev_21"] = c / c.shift(21) - 1
    f["vol_60"] = ret.rolling(60).std()
    f["high_52w"] = c / h.rolling(252).max()
    f["amihud_21"] = (ret.abs() / amt).rolling(21).mean()
    f["max_ret_21"] = ret.rolling(21).max()
    f["overnight_21"] = (o / c.shift(1) - 1).rolling(21).mean()
    f["amt_trend"] = amt.rolling(21).mean() / amt.rolling(126).mean()
    f["log_amt_63"] = np.log(amt.rolling(63).mean())
    # 자격 필터 재료 (등록 §3 — 전부 S 이전 데이터)
    f["_med_amt_20"] = amt.rolling(20).median()
    f["_close"] = c
    return f


def build_panel(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame,
                                               pd.DataFrame, pd.DataFrame]:
    """주간 패널 + 가격 와이드 프레임.

    반환: (panel, opens_w, exitpx_w, ks200)
    - panel: 행 = (신호일 S, code), 열 = 피처 + entry_day(E) + y_rank(학습 라벨)
    - opens_w: 마스터 캘린더 x code 시가 (진입가)
    - exitpx_w: 시가에 '마지막 거래 종가 ffill' 폴백을 합성한 청산가 프레임
    """
    cfg = _xcfg()
    liq = cfg["liquidity"]
    ks = master_calendar()
    cal = ks.index
    rebals = rebalance_days(cal)
    sig_of = {}   # E -> S
    for e in rebals:
        sig_of[e] = cal[cal.get_loc(e) - 1]
    sig_days = set(sig_of.values())

    uni = xs_universe()
    opens, closes = {}, {}
    feat_rows = []
    n_loaded = 0
    for _, r in uni.iterrows():
        df = load_xs(r["code"])
        if len(df) < 60:
            continue
        n_loaded += 1
        opens[r["code"]] = df["Open"]
        closes[r["code"]] = df["Close"]
        f = symbol_features(df)
        # 마스터 20일 창 내 실제 거래일 수 (정지 배제) — S 이전 데이터만
        aligned = df["Close"].reindex(cal)
        active20 = aligned.notna().rolling(20).sum()
        f = f[f.index.isin(sig_days)].copy()
        if f.empty:
            continue
        f["_active20"] = active20.reindex(f.index)
        f["code"] = r["code"]
        feat_rows.append(f)
        if verbose and n_loaded % 500 == 0:
            print(f"  ... 피처 {n_loaded}종목", flush=True)

    opens_w = pd.DataFrame(opens).reindex(cal).astype("float32")
    closes_w = pd.DataFrame(closes).reindex(cal).astype("float32")
    # 청산가: 당일 시가, 없으면 직전까지의 마지막 거래 종가 (상폐/정지 처분 가정)
    exitpx_w = opens_w.where(opens_w.notna(), closes_w.ffill().shift(1))

    panel = pd.concat(feat_rows)
    panel.index.name = "sig_day"
    panel = panel.reset_index()

    # 자격 필터 (등록 §3)
    elig = ((panel["_med_amt_20"] >= liq["min_median_amount_20d"])
            & (panel["_close"] >= liq["min_price"])
            & (panel["_active20"] >= liq["min_active_days_20d"]))
    panel = panel[elig].drop(columns=["_med_amt_20", "_close", "_active20"])

    # 진입일/라벨 (등록 §5): E = S 다음 마스터 거래일, X = E+5 마스터 거래일
    s_to_e = {s: e for e, s in sig_of.items()}
    panel["entry_day"] = pd.to_datetime(panel["sig_day"].map(s_to_e))
    e_pos = panel["entry_day"].map(lambda d: cal.get_loc(d))
    x_pos = (e_pos + cfg["target_horizon"]).clip(upper=len(cal) - 1)
    panel["exit_label_day"] = pd.to_datetime([cal[i] for i in x_pos])

    entry_px = opens_w.to_numpy()[e_pos.to_numpy(),
                                  opens_w.columns.get_indexer(panel["code"])]
    exit_px = exitpx_w.to_numpy()[x_pos.to_numpy(),
                                  exitpx_w.columns.get_indexer(panel["code"])]
    panel["entry_px"] = entry_px
    with np.errstate(divide="ignore", invalid="ignore"):   # 0/NaN 진입가는 아래서 제거
        panel["y_raw"] = exit_px / entry_px - 1
    panel = panel[np.isfinite(panel["entry_px"]) & (panel["entry_px"] > 0)
                  & np.isfinite(panel["y_raw"])]
    # 라벨 = 날짜 내 횡단면 랭크 (시장 방향 제거 — 등록 §5)
    panel["y_rank"] = panel.groupby("sig_day")["y_raw"].rank(pct=True)
    if verbose:
        wk = panel.groupby("sig_day").size()
        print(f"패널: {len(panel):,}행 / {len(wk)}주 / 주당 자격 종목 "
              f"중앙값 {wk.median():.0f} (로드 {n_loaded}종목)")
    return panel, opens_w, exitpx_w, ks


FEATURES = ["mom_12_1", "mom_6_1", "rev_21", "vol_60", "high_52w",
            "amihud_21", "max_ret_21", "overnight_21", "amt_trend", "log_amt_63"]


def walkforward_scores(panel: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """연 단위 확장 워크포워드 (ml 동결 구조·하이퍼파라미터 재사용, 등록 §5)."""
    from lightgbm import LGBMRegressor

    cfg = load_config()
    params = cfg["ml"]["model"]
    oos_start = cfg["xsection"]["walkforward"]["oos_start"]
    years = sorted({d.year for d in panel["entry_day"] if d.year >= oos_start})
    out = []
    for y in years:
        fold = panel[panel["entry_day"].dt.year == y]
        first_e = fold["entry_day"].min()
        train = panel[panel["exit_label_day"] < first_e]   # 라벨 창 침범 차단
        if train.empty or fold.empty:
            continue
        model = LGBMRegressor(**params, verbose=-1)
        model.fit(train[FEATURES], train["y_rank"])
        fold = fold.copy()
        fold["score"] = model.predict(fold[FEATURES])
        out.append(fold)
        if verbose:
            print(f"  워크포워드 {y}: 학습 {len(train):,}행 → 예측 {len(fold):,}행",
                  flush=True)
    return pd.concat(out, ignore_index=True)


def simulate_portfolio(scored: pd.DataFrame, opens_w: pd.DataFrame,
                       exitpx_w: pd.DataFrame, ks: pd.DataFrame,
                       top_n: int, cost_rt: float, hedge_cost: float,
                       exclude_codes: set | None = None) -> pd.DataFrame:
    """주간 시장중립 포트폴리오 (등록 §6). 반환: 주간 시리즈 df.

    exclude_codes 는 생존편향 진단용(상폐 제외 재계산) — 판정에는 None.
    """
    cal = ks.index
    weeks = sorted(scored["entry_day"].unique())
    prev_hold: set = set()
    rows = []
    for i, e in enumerate(weeks):
        nxt = weeks[i + 1] if i + 1 < len(weeks) else None
        if nxt is None:
            break   # 마지막 주는 청산일이 없어 미완성 (하우스 관례)
        g = scored[scored["entry_day"] == e]
        if exclude_codes:
            g = g[~g["code"].isin(exclude_codes)]
        top = g.nlargest(top_n, "score")
        codes = list(top["code"])
        ep = opens_w.loc[e, codes].to_numpy(dtype=float)
        xp = exitpx_w.loc[nxt, codes].to_numpy(dtype=float)
        ok = np.isfinite(ep) & np.isfinite(xp) & (ep > 0)
        if ok.sum() == 0:
            continue
        long_ret = float(np.nanmean(xp[ok] / ep[ok] - 1))
        e_loc, n_loc = cal.get_loc(e), cal.get_loc(nxt)
        hedge_ret = float(ks["Open"].iloc[n_loc] / ks["Open"].iloc[e_loc] - 1)
        n_new = len(set(codes) - prev_hold)
        cost = cost_rt * n_new / max(len(codes), 1) + hedge_cost
        rows.append({"entry_day": e, "exit_day": nxt, "n": int(ok.sum()),
                     "long_ret": long_ret, "hedge_ret": hedge_ret,
                     "turnover_new": n_new,
                     "net_ret": long_ret - hedge_ret - cost})
        prev_hold = set(codes)
    return pd.DataFrame(rows)


def _stats(weekly: pd.Series) -> dict:
    n = len(weekly)
    mean, sd = weekly.mean(), weekly.std(ddof=1)
    t = mean / sd * np.sqrt(n) if sd > 0 else float("nan")
    sharpe = mean / sd * np.sqrt(52) if sd > 0 else float("nan")
    eq = (1 + weekly).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    return {"n": n, "mean": float(mean), "t": float(t), "sharpe": float(sharpe),
            "win": float((weekly > 0).mean()), "cum": float(eq.iloc[-1] - 1),
            "mdd": mdd}


def run_xsection() -> dict:
    cfg = _xcfg()
    j = cfg["judgment"]["stage1"]
    print("=== xsection Stage 1 (사전 등록 2026-09-03 — 1회 실행 판정) ===")
    panel, opens_w, exitpx_w, ks = build_panel()
    scored = walkforward_scores(panel)
    port = simulate_portfolio(scored, opens_w, exitpx_w, ks, cfg["top_n"],
                              cfg["cost_round_trip"], cfg["hedge"]["cost_weekly"])
    weekly = port.set_index("entry_day")["net_ret"]
    st = _stats(weekly)
    yearly = weekly.groupby(weekly.index.year).agg(["mean", "sum", "count"])
    pos_years = int((yearly["mean"] > 0).sum())

    passed = bool(st["t"] >= j["t_min"] and st["sharpe"] >= j["sharpe_min"]
                  and pos_years >= j["positive_years_min"])

    # 부속 진단 (판정 아님, 등록 §7)
    delisted = set(xs_universe().query("delisted")["code"])
    port_nd = simulate_portfolio(scored, opens_w, exitpx_w, ks, cfg["top_n"],
                                 cfg["cost_round_trip"],
                                 cfg["hedge"]["cost_weekly"],
                                 exclude_codes=delisted)
    st_nd = _stats(port_nd.set_index("entry_day")["net_ret"])
    port_2x = simulate_portfolio(scored, opens_w, exitpx_w, ks, cfg["top_n"],
                                 cfg["cost_round_trip"] * 2,
                                 cfg["hedge"]["cost_weekly"] * 2)
    st_2x = _stats(port_2x.set_index("entry_day")["net_ret"])
    # top vs bottom 스프레드 (총수익 기준 — 랭킹 판별력)
    spread = []
    for _e, g in scored.groupby("entry_day"):
        top = g.nlargest(cfg["top_n"], "score")["y_raw"].mean()
        bot = g.nsmallest(cfg["top_n"], "score")["y_raw"].mean()
        spread.append(top - bot)
    spread_mean = float(np.nanmean(spread))

    port.to_csv(RESULTS_DIR / "xsection_weekly.csv", index=False,
                encoding="utf-8-sig")
    res = {"stats": st, "yearly": yearly, "pos_years": pos_years,
           "passed": passed, "no_delisted": st_nd, "cost2x": st_2x,
           "spread": spread_mean, "weeks": len(weekly)}
    _write_report(res, cfg)
    verdict = "통과 — Stage 2 섀도 개시 가능" if passed else "기각 — 프로그램 종결"
    print(f"\nStage 1 판정: {verdict}")
    print(f"  OOS 주간 {st['n']}주 | 평균 {st['mean']:+.3%} | t={st['t']:.2f} "
          f"(기준 {j['t_min']}) | Sharpe {st['sharpe']:.2f} (기준 {j['sharpe_min']}) "
          f"| 양수 연도 {pos_years}/7 (기준 {j['positive_years_min']})")
    return res


def _write_report(res: dict, cfg: dict) -> None:
    st = res["stats"]
    j = cfg["judgment"]["stage1"]
    lines = [f"""# 횡단면 ML 랭킹 — Stage 1 결과 (사전 등록 2026-09-03)

생성일: {pd.Timestamp.today().date()} | 등록 전문: PREREG_xsection.md |
설정: config `xsection` (실행 전 커밋 고정)

## 판정 (등록: t>={j['t_min']}, Sharpe>={j['sharpe_min']}, 양수연도>={j['positive_years_min']}/7)

**{'통과 — Stage 2 섀도 개시' if res['passed'] else '기각 — 프로그램 종결 (변형 재시험 금지)'}**

| 항목 | OOS(2020~) |
|---|---|
| 주간 관측 | {st['n']} |
| 평균/주 | {st['mean']:+.3%} |
| 단측 t | {st['t']:.2f} |
| 연환산 Sharpe | {st['sharpe']:.2f} |
| 승률 | {st['win']:.1%} |
| 누적 | {st['cum']:+.1%} |
| MDD | {st['mdd']:.1%} |
| 양수 연도 | {res['pos_years']}/7 |

## 연도별 (주간 평균 / 합산 / 주수)
"""]
    lines.append("| 연도 | 평균/주 | 합산 | 주수 |")
    lines.append("|---|---|---|---|")
    for y, r in res["yearly"].iterrows():
        lines.append(f"| {y} | {r['mean']:+.3%} | {r['sum']:+.1%} "
                     f"| {int(r['count'])} |")
    nd, c2 = res["no_delisted"], res["cost2x"]
    lines.append(f"""
## 부속 진단 (판정 아님)

- **생존편향 정량화**: 상폐 제외 시 평균 {nd['mean']:+.3%}/주, t={nd['t']:.2f},
  Sharpe {nd['sharpe']:.2f} — 본판(상폐 포함)과의 차이가 편향의 크기.
- **비용 2배 스트레스**: 평균 {c2['mean']:+.3%}/주, t={c2['t']:.2f},
  Sharpe {c2['sharpe']:.2f}.
- **랭킹 판별력**: top{cfg['top_n']} - bottom{cfg['top_n']} 총수익 스프레드
  평균 {res['spread']:+.3%}/주 (비용 전).

주간 시리즈: results/xsection_weekly.csv. 판정 규칙·검정력 한계는 등록 전문 참조.
""")
    out = RESULTS_DIR / "xsection.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out}", file=sys.stderr)
