"""ETF별 왕복 거래비용 실측 추정 — 판정용 동결 비용표.

왜 필요한가: 스크리닝·장부는 왕복 0.1% 일괄 가정인데, 트레이드당 엣지가 얇은
후보(KOSEF_KTB10Y +0.36%)는 비용 0.1%p 차이로 판정이 뒤집힌다. 비용을 성과가
아니라 시장 미시구조(호가단위·유동성·수수료)에서 유도하면 사후 튜닝이 아니다.

방법 (config `etf.costs`, 성과 미참조):
- 스프레드 = 호가단위 x 티어(일평균 거래대금 기준) / 중앙값 가격.
  한국 ETF는 LP 의무 호가로 스프레드가 묶여 있어, 고유동성(>=500억/일)은 1틱,
  중유동성(>=100억)은 2틱, 저유동성은 4틱을 보수적 추정으로 쓴다.
- 왕복 비용 = 스프레드 + 2 x 편도 수수료.
- **판정용 비용 = max(일괄 0.1%, 위 추정)** — 스크리닝 동결 가정보다 낮춰 잡는
  것을 금지한다 (비용을 낮추면 판정이 후해지므로, 실측이 더 높을 때만 반영).

Corwin-Schultz(2012) 고저가 추정도 병기하되 **판정에 쓰지 않는다**: 실측 검증에서
일 9,000억 거래·호가창 1틱짜리 KODEX200에 0.49%가 나오는 등 추정 순서가 유동성과
역상관 — 변동성(특히 2026 폭락 구간의 큰 일중 범위)을 스프레드로 오인한다.
일봉 CS는 이 유니버스에서 스프레드 추정기로 기각, 변동성 오염 상한 참고치로만 병기.

판정 연결: config `etf_paper.judgment` [4] — 포워드 판정 통계는 이 모듈이 산출한
`results/etf_costs.csv`(생성 시점 동결, git 추적)의 judgment_cost 로 재계산한다.
이 표는 판정용 비용 산정이지 게이트 재판정이 아니다 — 후보 목록·게이트 결과는
동결 그대로다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, load_config
from .data_loader import confirmed_cutoff
from .etf_swing import iter_candidates, simulate, simulate_volbreak


def corwin_schultz(df: pd.DataFrame, window_days: int) -> float:
    """CS 스프레드 추정 (참고치 전용 — 변동성 오염으로 판정 미사용)."""
    d = df.tail(window_days + 1)
    h, lo, o = d["High"].to_numpy(float), d["Low"].to_numpy(float), d["Open"].to_numpy(float)

    h1, l1 = h[:-1], lo[:-1]
    h2, l2, o2 = h[1:].copy(), lo[1:].copy(), o[1:]
    # 갭 보정: 익일 시가가 전일 범위 밖이면 익일 고저를 평행이동 (CS 2012 권고)
    shift = np.maximum(l1 - o2, 0.0) - np.maximum(o2 - h1, 0.0)
    h2 += shift
    l2 += shift

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
        gamma = np.log(np.maximum(h1, h2) / np.minimum(l1, l2)) ** 2
    denom = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = np.where(np.isfinite(s), np.maximum(s, 0.0), np.nan)
    return float(np.nanmean(s))


def _spread_ticks(adv_krw: float, tiers: list) -> int:
    """일평균 거래대금 → 스프레드 폭(틱 수). tiers는 [min_adv_억, ticks] 내림차순."""
    for min_adv_eok, ticks in tiers:
        if adv_krw >= min_adv_eok * 1e8:
            return int(ticks)
    return int(tiers[-1][1])


def estimate_costs() -> pd.DataFrame:
    cfg = load_config()
    ccfg = cfg["etf"]["costs"]
    flat = cfg["etf"]["cost_round_trip"]

    rows = []
    seen: set[str] = set()
    for cand, df, _, _, _, _ in iter_candidates(False, cutoff=confirmed_cutoff()):
        code = str(cand["code"])
        if code in seen:
            continue
        seen.add(code)
        win = df.tail(ccfg["window_days"])
        px = float(win["Close"].median())
        adv = float((win["Close"] * win["Volume"]).mean())
        ticks = _spread_ticks(adv, ccfg["adv_tiers"])
        spread = ticks * ccfg["tick_krw"] / px
        micro = spread + 2 * ccfg["commission_per_side"]
        rows.append({"code": code, "name": cand["name"], "adv_krw": adv,
                     "spread_ticks": ticks, "micro_cost": micro,
                     "cs_ref": corwin_schultz(df, ccfg["window_days"]),
                     "judgment_cost": max(flat, micro)})
    return pd.DataFrame(rows)


def run_costs() -> None:
    cfg = load_config()
    cost_flat = cfg["etf"]["cost_round_trip"]
    path = RESULTS_DIR / "etf_costs.csv"
    if path.exists():
        # 판정용 비용표는 생성 시점 동결 — 기존 행은 재측정하지 않는다.
        # (데이터가 늘어난 뒤 재생성하면 판정 직전 비용 선택이 가능해져 등록이 깨짐)
        # 단, 이후 편입된 신규 후보는 장부가 쌓이기 전 최초 1회만 측정해 append —
        # 신규 종목 추가는 사후 선택이 아니다.
        est = pd.read_csv(path, dtype={"code": str})
        fresh = estimate_costs()
        new = fresh[~fresh["code"].isin(set(est["code"]))]
        if len(new):
            est = pd.concat([est, new], ignore_index=True)
            est.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"동결 비용표에 신규 후보 {len(new)}건 추가: "
                  + ", ".join(new["name"]) + " (기존 행 불변)")
        else:
            print(f"기존 동결 비용표 사용: {path} (재생성 안 함)")
    else:
        est = estimate_costs()
        est.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"비용표 생성·동결: {path}")

    print(f"\n{'ETF':18s} {'거래대금(일)':>10s} {'틱':>3s} {'미시구조':>8s} "
          f"{'CS참고':>8s} {'판정비용':>8s}")
    for _, r in est.iterrows():
        print(f"{r['name']:18s} {r['adv_krw']/1e8:9.1f}억 {r['spread_ticks']:3d} "
              f"{r['micro_cost']*100:7.3f}% {r['cs_ref']*100:7.3f}% "
              f"{r['judgment_cost']*100:7.3f}%")

    # 후보별 인샘플 통계를 판정 비용으로 재계산 (정보 제공 — 게이트 재판정 아님)
    cost_map = dict(zip(est["code"], est["judgment_cost"], strict=True))
    lines = ["# ETF별 거래비용 — 판정용 동결 비용표", "",
             f"생성일: {pd.Timestamp.today().date()} | 방법: config `etf.costs` — "
             "스프레드 = 호가단위 x 유동성 티어(틱), 왕복 = 스프레드 + 수수료 x 2.",
             "판정비용 = max(일괄 0.1%, 미시구조 추정) — 동결 가정보다 낮추지 않는다.",
             "",
             "CS(Corwin-Schultz) 열은 참고치 전용: 일봉 CS는 이 유니버스에서 변동성을",
             "스프레드로 오인해(KODEX200 0.49% 등 유동성과 역상관) 추정기로 기각됐다.",
             "",
             "| ETF | 일평균 거래대금 | 스프레드(틱) | 미시구조 왕복 | CS 참고 | **판정비용** |",
             "|---|---|---|---|---|---|"]
    for _, r in est.iterrows():
        lines.append(f"| {r['name']} | {r['adv_krw']/1e8:.0f}억 | {r['spread_ticks']}틱 "
                     f"| {r['micro_cost']:.3%} | {r['cs_ref']:.3%} "
                     f"| **{r['judgment_cost']:.3%}** |")
    lines += ["", f"## 후보 인샘플 통계: 일괄 {cost_flat:.1%} vs 판정비용 (정보 제공)",
              "", "게이트·후보 목록은 동결 그대로 — 포워드 **판정 통계**에만 판정비용을 쓴다.",
              "", "| 후보 | 평균(일괄) | t(일괄) | 평균(판정) | t(판정) |", "|---|---|---|---|---|"]

    print(f"\n{'후보':30s} {'일괄 0.1%':>16s} {'판정비용':>16s}")
    k_vb = cfg["etf"]["strategies"]["volbreak"]["k"]
    for cand, df, entry, exit_, mh, tr in iter_candidates(False, cutoff=confirmed_cutoff()):
        key = f"{cand['name']} {cand['strategy']}"
        c_est = cost_map[str(cand["code"])]
        cells = []
        for c in (cost_flat, c_est):
            if cand["strategy"] == "volbreak":
                t = simulate_volbreak(df, k_vb, c)
            else:
                t = simulate(df, entry, exit_, mh, c, trailing=tr)
            r = t["net_ret"]
            tstat = r.mean() / r.std(ddof=1) * np.sqrt(len(r))
            cells.append((r.mean(), tstat))
        print(f"{key:30s} {cells[0][0]*100:+7.2f}%/t={cells[0][1]:4.2f} "
              f"{cells[1][0]*100:+7.2f}%/t={cells[1][1]:4.2f}")
        lines.append(f"| {key} | {cells[0][0]:+.2%} | {cells[0][1]:.2f} "
                     f"| {cells[1][0]:+.2%} | {cells[1][1]:.2f} |")

    (RESULTS_DIR / "etf_costs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport: {RESULTS_DIR / 'etf_costs.md'}")
