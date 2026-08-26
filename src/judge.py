"""판정 절차 실행기 — 사전 등록된 결정 규칙(config)의 코드 동결본.

왜 지금 짜는가: 기준(etf_paper.judgment)만 등록하고 계산 절차를 판정 시점(2.3년 뒤)에
구현하면, 데이터를 보면서 정렬·비용·표본 경계 같은 구현 세부를 고를 여지가 생긴다.
장부 0행인 지금 절차까지 코드로 동결해야 사전 등록이 완결된다. 이 파일의 판정
로직 변경은 사후 기준 변경과 동일하게 취급한다 — 버그 수정 외 금지.

절차 세부 (여기서 동결하는 결정들):
- **표본 경계**: "N건 도달 시점"의 표본 = 청산일(완결) 순 정렬 후 처음 N건.
  실행 시점이 늦어져 장부에 더 쌓여 있어도 처음 N건만 쓴다 — 판정이 실행
  시점에 좌우되면 안 된다. 정렬 키: (exit_date, entry_date, name) 결정적.
- **비용 재계산**: 장부 net_ret 는 왕복 0.1% 일괄 차감값 → 판정 수익률 =
  net_ret + 0.1% - 판정비용(results/etf_costs.csv judgment_cost, 동결).
  rotation2 는 월 1회 리밸런스로 회전율이 낮아 비용 민감도가 스윙 대비 무시
  가능(인샘플 비용 2배에도 t 거의 불변) — 장부 0.1% 그대로 쓴다.
- **검정**: 단측 one-sample t (H0: 평균<=0), 임계는 config 의 t 또는 알파.
  scipy 없이 계산 가능한 형태(임계 t 사전 등록)를 우선한다.

실행: python -m src.main judge — 표본 미달이면 현재 도달률만 출력 (판정 안 함).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, load_config


def first_n_by_completion(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """완결(청산일) 순 처음 n건 — 판정 표본의 결정적 경계."""
    return df.sort_values(["exit_date", "entry_date", "name"]).head(n)


def adjust_costs(df: pd.DataFrame, cost_flat: float,
                 cost_map: dict[str, float]) -> pd.Series:
    """장부 net_ret(일괄 비용) → 판정비용 반영 수익률. 매핑 없으면 일괄 유지."""
    adj = df["net_ret"] + cost_flat - df["name"].map(cost_map).fillna(cost_flat)
    return adj


def one_sided_t(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if len(r) >= 2 else float("nan")


def _load_judgment_costs(cfg: dict) -> dict[str, float]:
    path = RESULTS_DIR / "etf_costs.csv"
    if not path.exists():
        raise SystemExit("판정용 비용표가 없음 — `python -m src.main cost` 로 생성(동결) 후 재실행")
    est = pd.read_csv(path, dtype={"code": str})
    return dict(zip(est["name"], est["judgment_cost"], strict=True))


def run_judge() -> None:
    from .etf_paper import load_etf_ledger

    cfg = load_config()
    jcfg = cfg["etf_paper"]["judgment"]
    cost_flat = cfg["etf"]["cost_round_trip"]
    cost_map = _load_judgment_costs(cfg)
    led = load_etf_ledger()

    # 계열(family)별 판정 — 메커니즘이 다른 전략(추세추종/캘린더)을 한 풀에 섞으면
    # 시험이 오염되므로, 후보의 family 필드(기본 trend)로 풀을 가른다 (v2.1)
    fam_of = {(c["name"], c["strategy"]): c.get("family", "trend")
              for c in cfg["etf_paper"]["candidates"]}
    for fam, rule in jcfg["families"].items():
        members = [k for k, f in fam_of.items() if f == fam]
        fam_led = (led[pd.MultiIndex.from_frame(led[["name", "strategy"]])
                       .isin(set(members))] if len(led) else led)

        if rule.get("mode") == "per_candidate":
            # 풀링 검정 없음 (동시 트리거 상관으로 풀 t가 부풀기 때문 — config 주석).
            # 후보별 단측 t (다중성 보정 임계) — 각 후보 1회, 도달 순서대로.
            need, crit = rule["n"], rule["t"]
            print(f"=== [{fam}] 후보별 판정 (다중성 보정 단측 t>={crit} — 절차 동결본) ===")
            for name, strat in members:
                sub = (fam_led[(fam_led["name"] == name)
                               & (fam_led["strategy"] == strat)]
                       if len(fam_led) else fam_led)
                label = f"{name} {strat}"
                if len(sub) < need:
                    print(f"  {label:34s} 표본 대기 {len(sub)}/{need}")
                    continue
                r = adjust_costs(first_n_by_completion(sub, need), cost_flat, cost_map)
                t = one_sided_t(r)
                ok = bool(r.mean() > 0 and t >= crit)
                print(f"  {label:34s} 처음 {need}건 평균 {r.mean():+.3%} t={t:.2f} "
                      f"→ {'통과 — 실거래 편입 가능' if ok else '기각 — 해당 후보 제거'}")
            print()
            continue

        pooled_n, fam_t = rule["pooled_n"], rule["t"]
        print(f"=== [{fam}] 계열 판정 (후보 {len(members)}개 풀링 — 절차 동결본) ===")
        if len(fam_led) < pooled_n:
            print(f"  판정 시점 아님 — 완결 {len(fam_led)}/{pooled_n}건 "
                  f"(도달 시 이 명령이 그대로 판정)")
        else:
            sample = first_n_by_completion(fam_led, pooled_n)
            r = adjust_costs(sample, cost_flat, cost_map)
            t = one_sided_t(r)
            ok = bool(r.mean() > 0 and t >= fam_t)
            print(f"  표본: 완결 순 처음 {pooled_n}건 "
                  f"(마지막 청산 {sample['exit_date'].max().date()})")
            print(f"  풀링 평균 {r.mean():+.3%} | 단측 t={t:.2f} (임계 {fam_t})")
            verdict = ("통과 — 후보별 부호 규칙으로 채택 결정" if ok
                       else "기각 — 계열 실거래 불가 (재검정 금지)")
            print(f"  ▶ 계열 판정: {verdict}")

        if len(members) > 1:   # 단일 후보 계열은 계열 판정 = 후보 판정
            print(f"\n=== [{fam}] 후보별 채택/제거 (부호 규칙 — 계열 통과가 전제) ===")
            cand_need = jcfg["per_candidate_min"]
            for name, strat in members:
                sub = (fam_led[(fam_led["name"] == name)
                               & (fam_led["strategy"] == strat)]
                       if len(fam_led) else fam_led)
                label = f"{name} {strat}"
                if len(sub) < cand_need:
                    print(f"  {label:34s} 표본 대기 {len(sub)}/{cand_need}")
                    continue
                r = adjust_costs(first_n_by_completion(sub, cand_need),
                                 cost_flat, cost_map)
                verdict = "채택 후보 (계열 통과 시)" if r.mean() > 0 else "제거"
                print(f"  {label:34s} 처음 {cand_need}건 평균 {r.mean():+.3%} → {verdict}")
        print()

    print("=== rotation2 판정 (독립 계열 — 단독 단측 t, 1회) ===")
    r2 = cfg["etf_rotation2"]
    rot = led[led["strategy"] == "rotation2"] if len(led) else led
    need, crit = r2["judgment_min"], r2["judgment_t"]
    if len(rot) < need:
        print(f"  판정 시점 아님 — 완결 에피소드 {len(rot)}/{need}건")
    else:
        r = first_n_by_completion(rot, need)["net_ret"]   # 비용은 장부 그대로 (위 주석)
        t = one_sided_t(r)
        ok = bool(r.mean() > 0 and t >= crit)
        print(f"  처음 {need}건 평균 {r.mean():+.3%} | 단측 t={t:.2f} (임계 {crit})")
        print(f"  ▶ {'통과 — 실거래 편입 가능' if ok else '기각 — 로테이션 프로그램 종료'}")
    print(f"  (등록 시 명시: 이 검정의 검정력은 인샘플 재현 가정에서도 ~{r2['judgment_power_note']}"
          " — 표본 제약의 정직한 귀결이며, 낮다고 부호 규칙으로 바꾸면 α=50% 동전던지기가 된다)")

    _regime_composition(cfg, led)
    _execution_gate_status(cfg, led)


def _regime_composition(cfg: dict, led: pd.DataFrame) -> None:
    """판정 표본의 레짐 구성 병기 (2026-08-26 추가 — 측정·표시 전용, 기준 불변).

    지금 쌓이는 포워드 표본은 2026 폭락기에 편중된다. 판정 결과가 '어느 레짐의
    성과였나'를 해석할 수 있도록, 진입일에 해당 ETF가 폭락 구간(60일 고점 대비
    -10% 이하 — crash_study 검출 기준)이었던 비율을 계열별로 찍는다.
    """
    if led.empty:
        return
    from .data_loader import load_symbol

    code_of = {(c["name"], c["strategy"]): str(c["code"])
               for c in cfg["etf_paper"]["candidates"]}
    print("=== 판정 표본 레짐 구성 (측정 전용 — 진입일 기준 폭락 구간 비율) ===")
    fam_stats: dict[str, list[bool]] = {}
    for (name, strat), code in code_of.items():
        sub = led[(led["name"] == name) & (led["strategy"] == strat)]
        if sub.empty:
            continue
        px = load_symbol(code, "kr")["Close"]
        crash = (px / px.rolling(60).max() - 1) <= -0.10
        # asof 는 인덱스 시작 전 날짜에 NaN 을 주고 bool(NaN)=True 라 폭락으로
        # 오인된다 — notna 필터 후 집계
        vals = [crash.asof(d) for d in pd.to_datetime(sub["entry_date"])]
        flags = [bool(v) for v in vals if pd.notna(v)]
        if not flags:
            continue
        fam = next((c.get("family", "trend") for c in cfg["etf_paper"]["candidates"]
                    if c["name"] == name and c["strategy"] == strat), "trend")
        fam_stats.setdefault(fam, []).extend(flags)
    for fam, flags in sorted(fam_stats.items()):
        pct = sum(flags) / len(flags)
        print(f"  [{fam:9s}] 표본 {len(flags):3d}건 중 폭락 구간 진입 {pct:.0%} — "
              "판정 도달 시 이 비율을 해석에 병기할 것")
    print()


def _execution_gate_status(cfg: dict, led: pd.DataFrame) -> None:
    """실전 투입 게이트 상태 표시 (config etf_paper.execution_gate — 2026-08-26 동결).

    판정 통과가 전제인 2차 관문 — 여기서는 상태만 표시한다 (판정을 대체하지 않음).
    """
    gate = cfg["etf_paper"].get("execution_gate")
    if not gate:
        return
    print("=== 실전 투입 게이트 (판정 통과 후 2차 관문 — config 동결, 상태 표시만) ===")
    fill_path = RESULTS_DIR / "volbreak_fill_check.csv"
    fills = (pd.read_csv(fill_path).set_index("name")
             if fill_path.exists() else pd.DataFrame())
    for cand in cfg["etf_paper"]["candidates"]:
        name, strat = cand["name"], cand["strategy"]
        label = f"{name} {strat}"
        if strat == "volbreak":
            g = gate["volbreak"]
            if fills.empty or name not in fills.index:
                print(f"  {label:34s} fillcheck 표본 없음 — `fillcheck` 실행 필요")
                continue
            row = fills.loc[name]
            days_ok = int(row["minute_days"]) >= int(g["min_minute_days"])
            pos_ok = float(row["cons_mean"]) > 0
            state = ("충족" if days_ok and pos_ok else
                     f"미충족 (분봉 {int(row['minute_days'])}/{g['min_minute_days']}일, "
                     f"보수 평균 {row['cons_mean']:+.3%})")
            print(f"  {label:34s} 보수 체결 평균 {row['cons_mean']:+.3%} → {state}")
        else:
            g = gate["default"]
            sub = led[(led["name"] == name) & (led["strategy"] == strat)] \
                if len(led) else led
            ok_sub = sub[sub["preview"] == "ok"] if len(sub) else sub
            if not len(sub):
                print(f"  {label:34s} 표본 0건 — 대기")
                continue
            ratio = len(ok_sub) / len(sub)
            if ratio < float(g["min_ok_ratio"]):
                print(f"  {label:34s} ok 증빙 {len(ok_sub)}/{len(sub)}건 "
                      f"({ratio:.0%} < {g['min_ok_ratio']:.0%}) — 증빙 부족")
            else:
                m = float(ok_sub["net_ret"].mean())
                print(f"  {label:34s} ok 부분집합 평균 {m:+.3%} "
                      f"({len(ok_sub)}건) → {'충족' if m > 0 else '미충족'}")
    print("  (판정 통과 + 게이트 충족인 후보만 실전 투입 — 어느 한쪽도 대체 불가)")
