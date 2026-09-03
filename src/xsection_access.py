"""시장 접근 업그레이드 타당성 측정 — xsection 기각 후속 진단 (측정 전용).

질문: Stage 1 이 남긴 top50-bottom50 스프레드(+1.02%/주, 비용 전)는
**숏이 실제로 가능한 종목 집합** 안에서도 성립하는가?

- 개인의 숏 수단별 유니버스 프록시 (유동성 = log_amt_63 랭크):
  * futures150: 거래대금 상위 150 — 주식선물 상장 종목 근사
  * loanable30: 거래대금 상위 30% — 대주(신용거래) 용이 종목 근사
- 각 부분집합 안에서 score 상/하위 50을 다시 뽑아 라벨 수익(y_raw, 5일
  지평) 스프레드를 잰다. 부분집합 재선택이므로 원판 성적의 재해석이 아니라
  "그 유니버스에서 운용했다면"의 근사다.
- 비용 오버레이: 왕복 0.3% x 양다리 회전율 + 대주 이자 연 3% (보수 가정).

**이 결과는 전략 채택이 아니라 접근 경로 의사결정 자료다.** 여기서 유망해
보이는 어떤 구성도 실거래 전에 새 사전 등록을 거쳐야 하며, 그 등록은 이
수치를 인샘플 정보로 취급해야 한다 (이 진단이 설계에 정보를 준 순간부터).

실행: python -m src.main xaccess (결정적 재계산 — 재실행 안전)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .xsection import build_panel, walkforward_scores

BORROW_ANNUAL = 0.03      # 대주 이자 보수 가정 (연)
COST_RT = 0.003           # 왕복 (Stage 1 등록값과 동일)


def _weekly_spread(g: pd.DataFrame, n: int) -> dict | None:
    """부분집합 g 안에서 score 상/하위 n 스프레드 + 종목 집합."""
    if len(g) < n * 2:
        return None
    top = g.nlargest(n, "score")
    bot = g.nsmallest(n, "score")
    return {"top_ret": top["y_raw"].mean(), "bot_ret": bot["y_raw"].mean(),
            "spread": top["y_raw"].mean() - bot["y_raw"].mean(),
            "top_codes": set(top["code"]), "bot_codes": set(bot["code"])}


def _series_stats(s: pd.Series) -> dict:
    n, mean, sd = len(s), s.mean(), s.std(ddof=1)
    return {"n": n, "mean": float(mean),
            "t": float(mean / sd * np.sqrt(n)) if sd > 0 else float("nan"),
            "sharpe": float(mean / sd * np.sqrt(52)) if sd > 0 else float("nan")}


def run_access_study() -> dict:
    print("=== 시장 접근 타당성 측정 (xsection 후속 진단 — 측정 전용) ===")
    panel, _opens, _exitpx, _ks = build_panel(verbose=False)
    scored = walkforward_scores(panel, verbose=False)

    subsets = {
        "full": lambda g: g,
        "futures150": lambda g: g.nlargest(150, "log_amt_63"),
        "loanable30": lambda g: g.nlargest(max(int(len(g) * 0.3), 100),
                                           "log_amt_63"),
    }
    rows = {k: [] for k in subsets}
    prev = {k: (set(), set()) for k in subsets}
    turn = {k: [] for k in subsets}
    liq_decomp = []
    for e, g in scored.groupby("entry_day"):
        for k, sel in subsets.items():
            r = _weekly_spread(sel(g), 50)
            if r is None:
                continue
            rows[k].append({"entry_day": e, **{c: r[c] for c in
                                               ("top_ret", "bot_ret", "spread")}})
            pt, pb = prev[k]
            if pt:
                turn[k].append((len(r["top_codes"] - pt)
                                + len(r["bot_codes"] - pb)) / 100)
            prev[k] = (r["top_codes"], r["bot_codes"])
        # 숏 알파의 소재: 전체 유니버스 score 하위 50이 유동성 상위 30% 안에
        # 몇 종목이나 있고, 그 안/밖의 라벨 수익은 어떤가
        bot_full = g.nsmallest(50, "score")
        liq_cut = g["log_amt_63"].quantile(0.7)
        in_liq = bot_full[bot_full["log_amt_63"] >= liq_cut]
        out_liq = bot_full[bot_full["log_amt_63"] < liq_cut]
        liq_decomp.append({"n_liq": len(in_liq),
                           "ret_liq": in_liq["y_raw"].mean(),
                           "ret_illiq": out_liq["y_raw"].mean()})

    res = {}
    for k in subsets:
        df = pd.DataFrame(rows[k]).set_index("entry_day")
        to = float(np.mean(turn[k])) if turn[k] else 1.0
        # 주간 순스프레드 추정: 총스프레드 - 양다리 교체 비용 - 대주 이자
        cost_w = COST_RT * to + BORROW_ANNUAL / 52
        net = df["spread"] - cost_w
        res[k] = {"gross": _series_stats(df["spread"]),
                  "net": _series_stats(net), "turnover": to,
                  "cost_w": cost_w, "df": df,
                  "yearly": net.groupby(net.index.year).mean()}
    ld = pd.DataFrame(liq_decomp)
    res["short_locus"] = {
        "bot50_in_liq30": float(ld["n_liq"].mean()),
        "ret_liq": float(ld["ret_liq"].mean()),
        "ret_illiq": float(ld["ret_illiq"].mean()),
    }
    _write_report(res)
    for k in subsets:
        g, n = res[k]["gross"], res[k]["net"]
        print(f"  {k:11s}: 총 {g['mean']*100:+.3f}%/주 (t={g['t']:.1f}) → "
              f"순 {n['mean']*100:+.3f}%/주 (t={n['t']:.1f}, "
              f"Sharpe {n['sharpe']:.2f}, 회전 {res[k]['turnover']:.0%})")
    sl = res["short_locus"]
    print(f"  숏 알파 소재: 전체 bottom50 중 유동성 상위30% 평균 "
          f"{sl['bot50_in_liq30']:.1f}종목 — 그 라벨수익 {sl['ret_liq']*100:+.3f}% "
          f"vs 저유동 {sl['ret_illiq']*100:+.3f}%")
    return res


def _write_report(res: dict) -> None:
    lines = [f"""# 시장 접근 업그레이드 타당성 측정 (2026-09-03 — 측정 전용)

xsection Stage 1 기각(접근 구조 원인) 후속 진단. **전략 채택 아님** — 향후
어떤 롱숏 구성도 새 사전 등록 필수이며, 그 등록은 이 수치를 인샘플로 취급한다.

생성일: {pd.Timestamp.today().date()} | 가정: 왕복 {COST_RT:.1%} x 양다리 회전,
대주 이자 연 {BORROW_ANNUAL:.0%} | 스프레드 지평 = 라벨(5거래일)

## 숏 수단별 유니버스에서의 롱숏 스프레드 (OOS 2020~2026, 주간)

| 유니버스 | 총스프레드/주 | t | 순스프레드/주 | 순 t | 순 Sharpe | 회전율 |
|---|---|---|---|---|---|---|"""]
    for k, label in [("full", "전체 (숏 불가 — 참고 상한)"),
                     ("futures150", "거래대금 상위150 (주식선물 근사)"),
                     ("loanable30", "상위 30% (대주 용이 근사)")]:
        g, n = res[k]["gross"], res[k]["net"]
        lines.append(f"| {label} | {g['mean']:+.3%} | {g['t']:.1f} "
                     f"| {n['mean']:+.3%} | {n['t']:.1f} "
                     f"| {n['sharpe']:.2f} | {res[k]['turnover']:.0%} |")
    sl = res["short_locus"]
    lines.append(f"""
## 숏 알파의 소재 (전체 유니버스 score 하위 50 분해)

- 하위 50 중 유동성 상위 30%에 드는 종목: 평균 **{sl['bot50_in_liq30']:.1f}개/주**
- 그 종목들의 라벨 수익: {sl['ret_liq']:+.3%} (음수일수록 숏 가치)
- 저유동(하위 70%) 쪽: {sl['ret_illiq']:+.3%}

## 연도별 순스프레드 (주간 평균)

| 연도 | futures150 | loanable30 |
|---|---|---|""")
    y1, y2 = res["futures150"]["yearly"], res["loanable30"]["yearly"]
    for y in sorted(set(y1.index) | set(y2.index)):
        v1 = f"{y1.get(y, float('nan')):+.3%}" if y in y1.index else "-"
        v2 = f"{y2.get(y, float('nan')):+.3%}" if y in y2.index else "-"
        lines.append(f"| {y} | {v1} | {v2} |")
    lines.append("""
## 해석 규칙 (기록 시점 고정)

- 순 t 와 Sharpe 가 유동성 상위 유니버스에서도 뚜렷이 양수면: 접근 업그레이드
  (대주/주식선물)의 기대가치가 실측으로 뒷받침된다 → 다음 단계는 실제 대주
  가능 종목 리스트 확보 후 **새 사전 등록**.
- 유동성 상위에서 스프레드가 붕괴하면: 한국 롱숏은 접근을 올려도 실익이
  작다 → 글로벌(IBKR 등) 또는 기관화 경로만 남는다.
- 프록시 한계: 실제 주식선물 상장 종목·대주 가능 종목과 유동성 랭크는
  다르다. 본 측정은 방향 판단용이며 실 리스트로 재확인이 필요하다.
""")
    out = RESULTS_DIR / "xsection_access_study.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out}", file=sys.stderr)
