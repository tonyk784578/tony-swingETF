"""듀얼 모멘텀 로테이션 백테스트 (config etf_rotation — 2026-08-07 사전 등록).

규칙: 매월 마지막 거래일 종가 기준 252일 수익률 상위 top_k(절대 모멘텀 > 0
충족분만)를 다음 거래일 시가에 동일비중(슬롯 1/3) 리밸런스. 나가는 포지션은
같은 시가에 청산하며 왕복 비용은 청산 시 일괄 차감. 자격 미달 슬롯은 현금.

룩어헤드: 신호는 월말 D 종가까지의 정보, 체결은 D+1 시가 — 스윙 엔진과 동일한
'신호 다음날 시가' 원칙. 판정은 완결 포지션 에피소드에 Stage 1 기준 적용.
"""

from __future__ import annotations

import pandas as pd

from .backtest import combo_stats
from .config import RESULTS_DIR, load_config
from .data_loader import confirmed_cutoff, load_symbol


def _wide_frames(universe: dict, force: bool = False
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """유니버스의 (opens, closes) 와이드 프레임 — 상장 전 구간은 NaN."""
    cut = confirmed_cutoff()
    opens, closes = {}, {}
    for code, name in universe.items():
        df = load_symbol(str(code), "kr", force)
        df = df[df.index <= cut]
        opens[name], closes[name] = df["Open"], df["Close"]
    return pd.DataFrame(opens), pd.DataFrame(closes)


def month_end_days(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """각 (연, 월)의 마지막 거래일."""
    s = pd.Series(index, index=index)
    return sorted(s.groupby([index.year, index.month]).max().tolist())


def select_targets(mom_row: pd.Series, top_k: int) -> set[str]:
    """절대 모멘텀(>0) 충족분 중 상위 top_k. NaN(상장 1년 미만)은 자격 없음."""
    eligible = mom_row.dropna()
    eligible = eligible[eligible > 0]
    return set(eligible.nlargest(top_k).index)


def simulate_rotation(opens: pd.DataFrame, closes: pd.DataFrame, lookback: int,
                      top_k: int, cost: float
                      ) -> tuple[pd.DataFrame, list[dict], pd.Series]:
    """반환: (완결 에피소드 df, 미청산 포지션 목록, 일별 MTM 수익 시리즈)."""
    idx = closes.index
    mom = closes / closes.shift(lookback) - 1
    positions: dict[str, dict] = {}
    episodes = []

    for d in month_end_days(idx):
        loc = idx.get_loc(d)
        if loc + 1 >= len(idx):
            break                      # 신호는 있으나 체결일이 아직 없음
        e = idx[loc + 1]
        target = select_targets(mom.loc[d], top_k)
        for name in sorted(set(positions) - target):
            px = opens.at[e, name]
            if pd.isna(px):
                continue               # 체결 불가(거래정지 등) — 다음 리밸런스로 이월
            pos = positions.pop(name)
            episodes.append({
                "entry_date": pos["entry_date"], "exit_date": e, "name": name,
                "hold": idx.get_loc(e) - idx.get_loc(pos["entry_date"]),
                "net_ret": px / pos["entry_price"] - 1 - cost,
            })
        for name in sorted(target - set(positions)):
            if len(positions) >= top_k:
                break
            px = opens.at[e, name]
            if pd.isna(px):
                continue
            positions[name] = {"entry_date": e, "entry_price": px}

    ep = pd.DataFrame(episodes)
    last_close = closes.ffill().iloc[-1]   # 캐시 지연 종목은 마지막 유효 종가로 평가
    open_pos = [{"name": n, "entry_date": p["entry_date"],
                 "entry_price": p["entry_price"],
                 "unrealized": last_close[n] / p["entry_price"] - 1}
                for n, p in positions.items()]

    # 일별 MTM (슬롯 비중 1/3 고정): 진입일 Close/Open, 이후 Close/Close,
    # 청산일은 시가 체결이므로 Open/전일Close - 비용.
    # 유니온 캘린더의 결측 봉(타 ETF 상장 전·거래정지)은 ffill — 그날 MTM 0%
    w = 1.0 / top_k
    daily = pd.Series(0.0, index=idx)
    closes_f = closes.ffill()
    legs = [(r["entry_date"], r["exit_date"], r["name"], True) for _, r in ep.iterrows()]
    legs += [(p["entry_date"], idx[-1], p["name"], False) for p in open_pos]
    for entry, exit_, name, closed in legs:
        i0, i1 = idx.get_loc(entry), idx.get_loc(exit_)
        c, o = closes_f[name], opens[name]
        daily.iloc[i0] += w * (c.iloc[i0] / o.iloc[i0] - 1)
        last_mid = i1 - 1 if closed else i1
        for i in range(i0 + 1, last_mid + 1):
            daily.iloc[i] += w * (c.iloc[i] / c.iloc[i - 1] - 1)
        if closed:
            daily.iloc[i1] += w * (o.iloc[i1] / c.iloc[i1 - 1] - 1 - cost)
    return ep, open_pos, daily


def _gate(ep: pd.DataFrame, boundary: pd.Timestamp) -> dict:
    """Stage 1과 동일 판정: N>=30, 전/후반(각 n>=10) 양수, t>=2."""
    cfg = load_config()["etf"]
    r = ep.set_index("entry_date")["net_ret"]
    full = combo_stats(r)
    first = combo_stats(r[r.index < boundary])
    second = combo_stats(r[r.index >= boundary])
    passed = bool(full["n"] >= cfg["min_trades"]
                  and first["n"] >= 10 and second["n"] >= 10
                  and first["mean"] > 0 and second["mean"] > 0
                  and full["t_stat"] >= 2)
    return {"full": full, "first": first, "second": second, "passed": passed}


def _run_experiment(universe: dict, lookback: int, top_k: int, out_prefix: str,
                    title: str, force: bool = False) -> dict:
    cfg = load_config()
    cost = cfg["etf"]["cost_round_trip"]
    boundary = pd.Timestamp(cfg["split"]["boundary"])

    opens, closes = _wide_frames(universe, force)
    ep, open_pos, daily = simulate_rotation(opens, closes, lookback, top_k, cost)
    verdict = _gate(ep, boundary)

    equity = (1 + daily).cumprod()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    stats = {
        "cagr": equity.iloc[-1] ** (1 / years) - 1,
        "mdd": (equity / equity.cummax() - 1).min(),
        "sharpe": daily.mean() / daily.std() * (252 ** 0.5) if daily.std() > 0 else float("nan"),
    }
    bh = closes["KODEX200"].dropna().pct_change().fillna(0.0)
    bh_eq = (1 + bh).cumprod()
    bench = {
        "cagr": bh_eq.iloc[-1] ** (1 / years) - 1,
        "mdd": (bh_eq / bh_eq.cummax() - 1).min(),
        "sharpe": bh.mean() / bh.std() * (252 ** 0.5),
    }
    # 현재 시그널 상태 (기술적 표시 — 다음 월말 리밸런스 대상 아님에 유의)
    mom_now = (closes / closes.shift(lookback) - 1).iloc[-1]
    now_target = sorted(select_targets(mom_now, top_k))

    ep.to_csv(RESULTS_DIR / f"{out_prefix}_episodes.csv", index=False,
              encoding="utf-8-sig")
    _write_report(ep, open_pos, verdict, stats, bench, now_target,
                  lookback, top_k, out_prefix, title)
    return {"episodes": ep, "open_pos": open_pos, "verdict": verdict,
            "stats": stats, "bench": bench, "now_target": now_target}


def run_rotation(force: bool = False) -> dict:
    """실험 1 (2026-08-07 등록·기각 — 재현용): 기존 12종 유니버스."""
    cfg = load_config()
    rcfg = cfg["etf_rotation"]
    return _run_experiment(cfg["etf"]["universe"], rcfg["lookback"], rcfg["top_k"],
                           "rotation", "듀얼 모멘텀 로테이션", force)


def rotation2_universe() -> dict:
    """확장 로테이션 유니버스: universe + universe_ext − 파생형(레버리지·인버스)."""
    cfg = load_config()
    excl = {str(c) for c in cfg["etf_rotation2"]["exclude"]}
    uni = {**cfg["etf"]["universe"], **cfg["etf"]["universe_ext"]}
    return {c: n for c, n in uni.items() if str(c) not in excl}


def run_rotation2(force: bool = False) -> dict:
    """실험 2 (2026-08-07 등록·통과): 확장 유니버스, 파생형 제외."""
    r2 = load_config()["etf_rotation2"]
    return _run_experiment(rotation2_universe(), r2["lookback"], r2["top_k"],
                           "rotation2", "확장 듀얼 모멘텀 로테이션 (자산군 다변화)", force)


def rotation2_episodes(force: bool = False
                       ) -> tuple[pd.DataFrame, list[dict], pd.DatetimeIndex]:
    """섀도용 리플레이 — 산출물 파일 없이 (에피소드, 미청산, 캘린더)만 반환."""
    cfg = load_config()
    r2 = cfg["etf_rotation2"]
    opens, closes = _wide_frames(rotation2_universe(), force)
    ep, open_pos, _ = simulate_rotation(opens, closes, r2["lookback"], r2["top_k"],
                                        cfg["etf"]["cost_round_trip"])
    return ep, open_pos, closes.index


def _write_report(ep: pd.DataFrame, open_pos: list, verdict: dict, stats: dict,
                  bench: dict, now_target: list, lookback: int, top_k: int,
                  out_prefix: str, title: str) -> None:
    v = verdict
    freq = ep.groupby("name")["net_ret"].agg(["size", "sum"]).sort_values("size",
                                                                          ascending=False)
    lines = [f"""# {title} (사전 등록 2026-08-07 — 1회 실행 판정)

생성일: {pd.Timestamp.today().date()} | 규칙: 월말 {lookback}일 모멘텀 상위
{top_k}종(절대 모멘텀>0) 익일 시가 동일비중 리밸런스 | 비용 왕복 0.1%

## 판정 (Stage 1 기준: N>=30, 전/후반 양수, t>=2)

| 구간 | N | 평균 | 승률 | t |
|---|---|---|---|---|"""]
    for label, st in [("전체", v["full"]), ("전반(~2020)", v["first"]),
                      ("후반(2021~)", v["second"])]:
        lines.append(f"| {label} | {st['n']} | {st['mean']:+.3%} "
                     f"| {st['win_rate']:.1%} | {st['t_stat']:.2f} |")
    lines.append(f"""
**판정: {'통과' if v['passed'] else '기각'}**

## 계좌 관점 (일별 MTM, 참고치)

| | 로테이션 | KODEX200 B&H |
|---|---|---|
| CAGR | {stats['cagr']:+.2%} | {bench['cagr']:+.2%} |
| MDD | {stats['mdd']:.1%} | {bench['mdd']:.1%} |
| Sharpe(연) | {stats['sharpe']:.2f} | {bench['sharpe']:.2f} |

## 보유 빈도 (에피소드 수 / 합산 순수익)

| ETF | 에피소드 | 합산 |
|---|---|---|""")
    for name, r in freq.iterrows():
        lines.append(f"| {name} | {int(r['size'])} | {r['sum']:+.1%} |")
    lines.append("\n미청산 포지션: "
                 + (", ".join(f"{p['name']}({p['unrealized']:+.1%})" for p in open_pos)
                    if open_pos else "없음"))
    now = ", ".join(now_target) or "전원 자격 미달(현금)"
    lines.append(f"현재 시그널(참고 — 다음 월말에 확정): {now}")
    (RESULTS_DIR / f"{out_prefix}.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
