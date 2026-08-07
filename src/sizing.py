"""사전 등록 2단 사이징 실험 (config.yaml sizing — 동결).

-3% 임계값은 인샘플 신호 세기 분석에서 나왔으므로, 역사적 데이터로는
전/후반 반쪽 각각에서 일관되게 우수한지(지지 여부)까지만 확인한다.
최종 채택 판정은 섀도 장부의 포워드 성과(weight 컬럼)로 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import condition_mask
from .config import RESULTS_DIR, load_config


def signal_weight(nvda_ret: pd.Series) -> pd.Series:
    """사이징 규칙: NVDA <= strong_threshold 면 strong_weight, 아니면 weak_weight."""
    s = load_config()["sizing"]
    return pd.Series(np.where(nvda_ret <= s["strong_threshold"],
                              s["strong_weight"], s["weak_weight"]),
                     index=nvda_ret.index)


def _variant_stats(net: pd.Series, weight: pd.Series) -> dict:
    """베팅 비중 적용 후 성과 (자본 대비 일간 수익 = weight * net_ret)."""
    port = weight * net
    n = len(port)
    std = port.std(ddof=1) if n > 1 else np.nan
    equity = (1 + port).cumprod()
    return {
        "n": n,
        "ret_on_capital": port.sum() / weight.sum() if weight.sum() else np.nan,
        "mean_port": port.mean(),
        "sharpe_trade": port.mean() / std if std and std > 0 else np.nan,
        "cum_ret": equity.iloc[-1] - 1 if n else np.nan,
        "mdd": (equity / equity.cummax() - 1).min() if n else np.nan,
    }


def run_sizing_experiment(masters: dict[str, pd.DataFrame]) -> dict:
    cfg = load_config()
    scfg = cfg["sizing"]
    cost = cfg["cost"]["round_trip"]
    stock = cfg["main_stock"]
    master = masters[stock]
    boundary = pd.Timestamp(cfg["split"]["boundary"])

    mask = condition_mask(master, scfg["condition"])
    trig = master[mask]
    net = trig["day_ret"] - cost
    w_tiered = signal_weight(trig["nvda_ret"])
    w_flat = pd.Series(1.0, index=trig.index)

    rows = []
    halves = {"first(2015~)": trig.index < boundary,
              "second(2021~)": trig.index >= boundary,
              "full": pd.Series(True, index=trig.index)}
    for period, sel in halves.items():
        for variant, w in [("flat", w_flat), ("tiered", w_tiered)]:
            rows.append({"period": period, "variant": variant,
                         **_variant_stats(net[sel], w[sel])})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "sizing_experiment.csv", index=False, encoding="utf-8-sig")

    def _sharpe(period, variant):
        return df[(df["period"] == period) & (df["variant"] == variant)
                  ]["sharpe_trade"].iloc[0]

    supported = (_sharpe("first(2015~)", "tiered") > _sharpe("first(2015~)", "flat")
                 and _sharpe("second(2021~)", "tiered") > _sharpe("second(2021~)", "flat"))

    lines = [f"""# 2단 사이징 실험 (사전 등록 2026-08-06)

규칙: `{scfg['condition']}` 발동일에 NVDA <= {scfg['strong_threshold']:.0%} 이면
비중 {scfg['strong_weight']}, 아니면 {scfg['weak_weight']} (flat 은 항상 1.0).

| 구간 | 변형 | N | 투입자본대비수익 | 회당 Sharpe | 누적 | MDD |
|---|---|---|---|---|---|---|"""]
    for _, r in df.iterrows():
        lines.append(f"| {r['period']} | {r['variant']} | {r['n']} "
                     f"| {r['ret_on_capital']:+.3%} | {r['sharpe_trade']:.3f} "
                     f"| {r['cum_ret']:+.1%} | {r['mdd']:.1%} |")
    lines.append(f"""
## 판정

- 역사적 지지 (전/후반 각각 tiered Sharpe > flat): **{'지지' if supported else '기각'}**
- 임계값 -3%가 인샘플에서 나온 값이므로 이 결과는 지지 증거일 뿐, **최종 채택은
  섀도 장부(weight 컬럼)의 포워드 성과가 인샘플 격차를 재현하는지로 판정한다.**
""")
    (RESULTS_DIR / "sizing_experiment.md").write_text("\n".join(lines), encoding="utf-8")
    return {"table": df, "supported": supported}
