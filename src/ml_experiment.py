"""사전 등록 ML 실험 — 워크포워드 LightGBM vs 수동 룰 벤치마크.

설계(config.yaml ml 섹션에 동결):
- 피처: master 프레임의 룩어헤드-안전 컬럼만 (새 데이터 소스 없음)
- 모델: 소형 LGBMRegressor 1개 구성, 목표변수 = 당일 순수익(day_ret - cost)
- 워크포워드: oos_start 이전 전체로 학습 → 해당 연도 예측, 이후 매년 확장 재학습
- 진입 규칙: 예측 순수익 > 0 인 날 시가 매수 → 종가 매도
- 채택 기준(셋 다): OOS 거래수 >= min_trades,
  ML 누적 순수익 > 룰 누적 순수익, ML 평균 순수익 t-stat >= 2
그리드서치 금지 — 설정을 바꿔 재실행하면 실험은 무효(다중검정).
"""

from __future__ import annotations

import pandas as pd

from .backtest import combo_stats, condition_mask
from .config import RESULTS_DIR, load_config


def _features(master: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "sox_ret": master["sox_ret"],
        "ixic_ret": master["ixic_ret"],
        "nvda_ret": master["nvda_ret"],
        "kospi_prev_ret": master["kospi_prev_ret"],
        "gap": master["gap"],
        "ma20_dist": master["prev_close"] / master["ma20_prev"] - 1,
    }
    order = load_config()["ml"]["features"]
    return pd.DataFrame({k: cols[k] for k in order})


def walk_forward(master: pd.DataFrame) -> pd.DataFrame:
    """연 단위 확장 워크포워드. 반환: OOS 전 행 (pred, net_ret, traded)."""
    from lightgbm import LGBMRegressor

    cfg = load_config()
    mcfg = cfg["ml"]
    cost = cfg["cost"]["round_trip"]

    X = _features(master).dropna()
    y = (master["day_ret"] - cost).loc[X.index]
    years = range(mcfg["oos_start"], X.index.year.max() + 1)

    frames = []
    for yr in years:
        train = X.index.year < yr
        test = X.index.year == yr
        if train.sum() < 500 or test.sum() == 0:
            continue
        model = LGBMRegressor(**mcfg["model"], verbose=-1)
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        frames.append(pd.DataFrame({
            "pred": pred, "net_ret": y[test], "year": yr,
        }, index=X.index[test]))

    wf = pd.concat(frames)
    wf["traded"] = wf["pred"] > 0
    return wf


def feature_importance(master: pd.DataFrame) -> pd.Series:
    """전체 기간 학습 모델의 gain 기준 피처 중요도 (해석용 참고치)."""
    from lightgbm import LGBMRegressor

    cfg = load_config()
    X = _features(master).dropna()
    y = (master["day_ret"] - cfg["cost"]["round_trip"]).loc[X.index]
    model = LGBMRegressor(**cfg["ml"]["model"], importance_type="gain", verbose=-1)
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    return (imp / imp.sum()).sort_values(ascending=False)


def run_experiment(masters: dict[str, pd.DataFrame]) -> dict:
    cfg = load_config()
    mcfg = cfg["ml"]
    cost = cfg["cost"]["round_trip"]
    stock = mcfg["target_stock"]
    master = masters[stock]

    wf = walk_forward(master)
    wf.to_csv(RESULTS_DIR / "ml_walkforward.csv", encoding="utf-8-sig")

    oos = master.loc[wf.index]
    net = oos["day_ret"] - cost

    st_ml = combo_stats(wf.loc[wf["traded"], "net_ret"])
    rule_mask = condition_mask(oos, mcfg["benchmark_condition"])
    st_rule = combo_stats(net[rule_mask])
    st_all = combo_stats(net)

    yearly = (wf[wf["traded"]].groupby("year")["net_ret"]
              .agg(n="count", mean="mean", win=lambda s: (s > 0).mean()))

    gates = {
        "min_trades": st_ml["n"] >= mcfg["min_trades"],
        "beats_rule_cum": st_ml["cum_ret"] > st_rule["cum_ret"],
        "t_stat_ge_2": bool(st_ml["t_stat"] >= 2),
    }
    verdict = all(gates.values())

    imp = feature_importance(master)

    lines = [f"""# ML 실험 리포트 — 워크포워드 LightGBM vs 룰

생성일: {pd.Timestamp.today().date()} | 대상: {stock} | OOS: {mcfg['oos_start']}~
벤치마크 룰: `{mcfg['benchmark_condition']}` | 설정: config.yaml `ml` (사전 등록·동결)

## OOS 성과 비교 (비용 {cost:.1%} 차감)

| 전략 | 거래수 | 평균 | 승률 | 누적 | MDD | t-stat |
|---|---|---|---|---|---|---|
| ML (pred>0 진입) | {st_ml['n']} | {st_ml['mean']:+.3%} | {st_ml['win_rate']:.1%} | {st_ml['cum_ret']:+.1%} | {st_ml['mdd']:.1%} | {st_ml['t_stat']:.2f} |
| 룰 벤치마크 | {st_rule['n']} | {st_rule['mean']:+.3%} | {st_rule['win_rate']:.1%} | {st_rule['cum_ret']:+.1%} | {st_rule['mdd']:.1%} | {st_rule['t_stat']:.2f} |
| 매일 진입 (참고) | {st_all['n']} | {st_all['mean']:+.3%} | {st_all['win_rate']:.1%} | {st_all['cum_ret']:+.1%} | {st_all['mdd']:.1%} | {st_all['t_stat']:.2f} |

## ML 연도별 (OOS)

| 연도 | 거래수 | 평균 | 승률 |
|---|---|---|---|"""]
    for yr, r in yearly.iterrows():
        lines.append(f"| {yr} | {int(r['n'])} | {r['mean']:+.3%} | {r['win']:.1%} |")

    lines.append("\n## 피처 중요도 (gain, 전 기간 참고치)\n")
    for name, v in imp.items():
        lines.append(f"- {name}: {v:.1%}")

    lines.append("\n## 채택 기준 판정\n")
    for g, ok in gates.items():
        lines.append(f"- {g}: {'통과' if ok else '탈락'}")
    lines.append(f"\n**판정: {'채택 — ML이 룰을 이김' if verdict else '기각 — 수동 룰 유지'}**")

    (RESULTS_DIR / "ml_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"ml": st_ml, "rule": st_rule, "all": st_all,
            "gates": gates, "verdict": verdict, "importance": imp}
