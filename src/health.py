"""시스템 헬스체크 + 판정 준비 상태 — 침묵 실패 감지용.

evening cron 마지막에 실행. [WARN] 라인이 있으면 사람이 확인해야 한다.
(알림 채널이 없는 동안에는 로그 grep이 유일한 통로 — grep WARN paper/logs/*.log)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, ROOT_DIR, load_config


def _warn(msg: str) -> bool:
    print(f"[WARN] {msg}")
    return False


def _ok(msg: str) -> bool:
    print(f"[OK] {msg}")
    return True


def _stale(code: str, label: str) -> bool | None:
    """캐시 지연 검사. True=최신, False=경고, None=파일 없음."""
    path = DATA_DIR / f"{code}.parquet"
    if not path.exists():
        return None
    last = pd.read_parquet(path).index.max()
    lag = int(np.busday_count(last.date(), pd.Timestamp.today().date()))
    if lag > 3:
        return _warn(f"{label} 데이터가 {lag}영업일 전({last.date()})에서 멈춤 — "
                     "cron/데이터소스 확인")
    return _ok(f"{label} 데이터 최신 ({last.date()}, {lag}영업일 전)")


def check_data_freshness() -> bool:
    """주력 종목 + 첫 ETF 후보 캐시가 최근 3영업일 내 데이터를 갖고 있는가."""
    cfg = load_config()
    stock_code = next(code for code, name in cfg["data"]["kr_stocks"].items()
                      if name == cfg["main_stock"])
    etf = cfg["etf_paper"]["candidates"][0]
    ok = True
    for code, label in [(str(stock_code), cfg["main_stock"]),
                        (str(etf["code"]), etf["name"])]:
        res = _stale(code, label)
        ok = _warn(f"{label} 캐시 없음 — download 실행 필요") if res is None else (res and ok)
    return ok


def check_evening_log() -> bool:
    """최근 5일 내 evening 실행 로그가 있는가 (WSL 꺼짐/cron 미동작 감지)."""
    log_dir = ROOT_DIR / "paper" / "logs"
    logs = sorted(log_dir.glob("*.log")) if log_dir.exists() else []
    if not logs:
        return _warn("실행 로그 없음 — cron이 한 번도 안 돌았거나 로그 경로 문제")
    latest = logs[-1]
    # 오늘 로그에 아직 evening이 없을 수 있으니 직전 로그까지 본다
    if ("evening ledger" not in latest.read_text(errors="ignore")
            and (len(logs) < 2
                 or "evening ledger" not in logs[-2].read_text(errors="ignore"))):
        return _warn("최근 로그에 evening 실행 기록 없음")
    age = (pd.Timestamp.today().normalize()
           - pd.Timestamp(latest.stem)).days
    if age > 5:
        return _warn(f"마지막 실행 로그가 {age}일 전({latest.stem}) — WSL/cron 확인")
    return _ok(f"실행 로그 최신 ({latest.stem})")


def check_revisions() -> bool:
    """최근 7일 내 과거 데이터 정정 경고가 있었는가."""
    path = DATA_DIR / "revisions.log"
    if not path.exists():
        return _ok("데이터 정정 이력 없음")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    recent = [ln for ln in lines
              if pd.Timestamp(ln[:10]) >= pd.Timestamp.today() - pd.Timedelta(days=7)]
    if recent:
        return _warn(f"최근 7일 내 데이터 정정 {len(recent)}건 — 백테스트/장부 재검토 필요 "
                     f"(data/revisions.log)")
    return _ok("최근 데이터 정정 없음")


def verdict_readiness() -> None:
    """진행 중 실험들의 판정 표본 도달률. 도달 시 실행할 명령 안내."""
    from .etf_paper import load_etf_ledger
    from .paper import load_ledger

    cfg = load_config()
    need = cfg["backtest"]["min_samples"]
    jcfg = cfg["etf_paper"].get("judgment", {})
    pooled_need = jcfg.get("pooled_checkpoint", need)
    cand_need = jcfg.get("per_candidate_min", need)
    print("\n=== 판정 준비 상태 (기준은 config etf_paper.judgment — 변경 금지) ===")

    led = load_ledger()
    n_stock = 0 if led.empty else int(
        led[led["condition"] == cfg["paper"]["conditions"][0]["condition"]]
        ["triggered"].sum())
    print(f"  개별주 주력조건 포워드: {n_stock}/{need}"
          + ("  ← 판정 가능! `paper` 요약으로 판정" if n_stock >= need else ""))

    etf_led = load_etf_ledger()
    n_etf = len(etf_led)
    print(f"  ETF 합산(계열 중간점검): {n_etf}/{pooled_need}"
          + ("  ← 중간점검 가능 (계열 생존 확인용 — 실거래 채택 근거 아님)"
             if n_etf >= pooled_need else ""))
    for cand in cfg["etf_paper"]["candidates"]:
        n = 0 if etf_led.empty else int(((etf_led["name"] == cand["name"])
                                         & (etf_led["strategy"] == cand["strategy"])).sum())
        print(f"    후보별(실거래 판정) {cand['name']} {cand['strategy']}: {n}/{cand_need}"
              + ("  ← 실거래 채택 판정 가능" if n >= cand_need else ""))

    try:
        from .align import build_master
        from .backtest import condition_mask
        from .data_loader import load_all
        from .minute_data import coverage, load_minute

        data = load_all()
        master = build_master(data, cfg["main_stock"])
        trig = master.index[condition_mask(master, cfg["paper"]["conditions"][0]["condition"])]
        n_min = len(trig.intersection(coverage(load_minute())))
        print(f"  손절 검증용 분봉 발동일: {n_min}/{cfg['stoploss']['min_days']}"
              + ("  ← 판정 가능! `stoploss` 실행" if n_min >= cfg["stoploss"]["min_days"]
                 else ""))
    except Exception as e:
        print(f"  손절 검증 집계 실패: {e}")


def run_health() -> None:
    print("=== health check ===")
    results = [check_data_freshness(), check_evening_log(), check_revisions()]
    verdict_readiness()
    n_warn = len(results) - sum(results)
    print(f"\nhealth: {'모두 정상' if n_warn == 0 else f'경고 {n_warn}건 — 위 [WARN] 확인'}")
