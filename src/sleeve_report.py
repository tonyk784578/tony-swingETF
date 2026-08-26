"""통합 슬리브 리스크 뷰 — 전 섀도 전략을 '한 계좌' 관점으로 (측정 전용).

왜 (2026-08-26 검토에서 지적): 포트폴리오 시뮬은 trend 계열만 다루고
volbreak/overnight/rotation2 는 별도 슬리브로 빠져 있다 — 개별로는 맞는
설계지만, 실전에서 한 계좌에 담기면 전체 노출·상관·동일 ETF 중복 보유를
보는 화면이 없다. 이 모듈이 그 화면이다. 신호·판정 기준 불변.

- 현재 보유: 전 후보의 미청산 포지션 + 같은 ETF 를 여러 전략이 같은 밤
  보유하는 중복(명목 노출 배증) 표시
- 슬리브 상관: 장부 완결 트레이드의 청산일 수익을 계열(family)별 일간
  시리즈로 만들어 상관 행렬 (비활동일=현금 0%). 표본이 얇은 초기에는
  참고치 — 장부가 쌓일수록 정확해진다.
- 같은 밤 중복 보유 이력: 1일 회전 계열(volbreak/overnight)이 같은 ETF 를
  같은 진입일에 보유한 건수 (실측 — Semicon 이중 보유 등)

실행: python -m src.main sleeves
"""

from __future__ import annotations

import pandas as pd

from .config import RESULTS_DIR, load_config


def _sleeve_of(cand: dict) -> str:
    return cand.get("family", "trend")


def run_sleeves(force: bool = False) -> dict:
    from .etf_paper import candidate_states, load_etf_ledger, rotation2_state

    cfg = load_config()
    led = load_etf_ledger()
    states = candidate_states(force)
    rot = rotation2_state(force)

    # 1. 현재 보유 + 동일 ETF 중복
    pos_rows = []
    for st in states:
        if st["open_pos"]:
            pos_rows.append({"code": str(st["cand"]["code"]), "name": st["cand"]["name"],
                             "strategy": st["cand"]["strategy"],
                             "sleeve": _sleeve_of(st["cand"]),
                             "unrealized": st["open_pos"]["unrealized"]})
    if rot:
        for p in rot["open_pos"]:
            pos_rows.append({"code": "", "name": p["name"], "strategy": "rotation2",
                             "sleeve": "rotation2", "unrealized": p["unrealized"]})
    positions = pd.DataFrame(pos_rows)
    dup = pd.DataFrame()
    if not positions.empty:
        by_name = positions[positions["strategy"] != "rotation2"].groupby("name")
        dup = by_name.filter(lambda g: len(g) > 1)

    # 2. 슬리브 일간 수익 상관 (장부 기반 — 청산일 귀속 근사)
    corr = pd.DataFrame()
    sleeve_daily = {}
    if not led.empty:
        fam_of = {(c["name"], c["strategy"]): _sleeve_of(c)
                  for c in cfg["etf_paper"]["candidates"]}
        led2 = led.copy()
        led2["sleeve"] = [
            "rotation2" if s == "rotation2" else fam_of.get((n, s), "trend")
            for n, s in zip(led2["name"], led2["strategy"], strict=True)]
        for sl, g in led2.groupby("sleeve"):
            sleeve_daily[sl] = g.groupby("exit_date")["net_ret"].mean()
        if len(sleeve_daily) >= 2:
            # DataFrame 생성자가 인덱스 외부조인을 해 준다 — 비활동일은 현금(0%)
            corr = pd.DataFrame(sleeve_daily).fillna(0.0).corr()

    # 3. 같은 밤 동일 ETF 중복 보유 이력 (1일 회전 계열)
    overlap_rows = []
    if not led.empty:
        oneday = led[led["strategy"].isin(["volbreak", "overnight"])]
        for name, g in oneday.groupby("name"):
            strats = g.groupby("strategy")["entry_date"].apply(set)
            if len(strats) > 1:
                both = set.intersection(*strats.tolist())
                if both:
                    overlap_rows.append({"name": name,
                                         "strategies": "+".join(sorted(strats.index)),
                                         "nights": len(both)})
    overlaps = pd.DataFrame(overlap_rows)

    res = {"positions": positions, "dup": dup, "corr": corr, "overlaps": overlaps,
           "ledger_n": len(led)}
    _write_report(res)
    return res


def _write_report(res: dict) -> None:
    lines = [f"""# 통합 슬리브 리스크 뷰 (측정 전용)

생성일: {pd.Timestamp.today().date()} | 근거: 섀도 장부 {res['ledger_n']}건 +
현재 미청산 포지션 | 신호·판정 기준 불변 — 한 계좌 관점의 리스크 가시화

## 현재 보유 (전 슬리브)
"""]
    if res["positions"].empty:
        lines.append("- 보유 없음")
    else:
        lines += ["| ETF | 전략 | 슬리브 | 평가 |", "|---|---|---|---|"]
        for _, r in res["positions"].iterrows():
            lines.append(f"| {r['name']} | {r['strategy']} | {r['sleeve']} "
                         f"| {r['unrealized']:+.2%} |")
    if not res["dup"].empty:
        names = ", ".join(sorted(res["dup"]["name"].unique()))
        lines.append(f"\n**⚠ 동일 ETF 중복 보유: {names}** — 같은 밤 명목 노출이 "
                     "전략 수만큼 배증한다. 실전 투입 시 노출 합산 캡 적용 대상.")

    lines.append("\n## 슬리브 간 일간 수익 상관 (장부 실측 — 비활동일=0)\n")
    if res["corr"].empty:
        lines.append("- 표본 부족 (2개 이상 슬리브에 완결 트레이드 필요)")
    else:
        cols = list(res["corr"].columns)
        lines.append("| | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for i, r in res["corr"].iterrows():
            lines.append(f"| {i} | " + " | ".join(f"{v:.2f}" for v in r) + " |")
        lines.append("\n초기 표본에선 참고치 — |ρ|>0.5 쌍은 실전에서 같은 리스크 "
                     "그룹으로 취급 (등록된 운영 원칙).")

    lines.append("\n## 같은 밤 동일 ETF 중복 보유 이력 (1일 회전 계열)\n")
    if res["overlaps"].empty:
        lines.append("- 아직 없음")
    else:
        lines += ["| ETF | 전략 조합 | 중복 밤 수 |", "|---|---|---|"]
        for _, r in res["overlaps"].iterrows():
            lines.append(f"| {r['name']} | {r['strategies']} | {r['nights']} |")

    out = RESULTS_DIR / "sleeve_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out}")
