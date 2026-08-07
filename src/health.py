"""시스템 헬스체크 + 판정 준비 상태 — 침묵 실패 감지용.

evening cron 마지막에 실행. [WARN] 라인이 있으면 사람이 확인해야 한다.

경고 통보는 세 겹이다: 로그 라인(`grep WARN paper/logs/*.log`),
프로젝트 루트의 `ALERT.md`(경고가 있는 동안에만 존재 — 파일 존재 자체가 상태),
그리고 Windows 풍선 알림(best-effort). 판정까지 1~3년을 무인 관측하는 구조라
경고가 로그에 묻히면 표본이 조용히 깨진 채로 몇 달이 지날 수 있다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, ROOT_DIR, load_config

ALERT_PATH = ROOT_DIR / "ALERT.md"
_WARNINGS: list[str] = []


def _warn(msg: str) -> bool:
    print(f"[WARN] {msg}")
    _WARNINGS.append(msg)
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


def _notify_desktop(title: str, msg: str) -> None:
    """Windows 풍선 알림 — best-effort. 실패해도 헬스체크 결과에 영향 없음."""
    ps = Path("/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not ps.exists():
        return
    safe = msg.replace("'", " ")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$n=New-Object System.Windows.Forms.NotifyIcon;"
        "$n.Icon=[System.Drawing.SystemIcons]::Warning;$n.Visible=$true;"
        f"$n.ShowBalloonTip(20000,'{title}','{safe}',"
        "[System.Windows.Forms.ToolTipIcon]::Warning);"
        "Start-Sleep -Seconds 8;$n.Dispose()")
    try:
        subprocess.run([str(ps), "-NoProfile", "-Command", script],
                       timeout=40, capture_output=True, check=False)
    except Exception as e:            # 알림 실패가 헬스체크를 깨뜨리면 안 된다
        print(f"  (데스크톱 알림 실패 — 무시: {e})")


def _publish_alert(warnings: list[str]) -> None:
    """경고 상태를 눈에 띄는 파일로 게시. 정상으로 돌아오면 지운다."""
    if not warnings:
        ALERT_PATH.unlink(missing_ok=True)
        return
    stamp = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M")
    body = "\n".join(f"- {w}" for w in warnings)
    ALERT_PATH.write_text(
        f"# 헬스체크 경고 {len(warnings)}건\n\n"
        f"최종 감지: {stamp} (KST)\n\n{body}\n\n"
        "---\n\n"
        "이 파일은 `python -m src.main health` 가 경고를 감지하는 동안에만 존재하며,\n"
        "정상으로 돌아오면 자동으로 삭제된다. 직접 지우지 말고 원인을 먼저 확인할 것.\n",
        encoding="utf-8")
    print(f"\n[ALERT] {ALERT_PATH.name} 게시 — 경고 {len(warnings)}건")
    _notify_desktop("TonySwingETF", f"헬스체크 경고 {len(warnings)}건 — ALERT.md 확인")


def run_health() -> None:
    print("=== health check ===")
    _WARNINGS.clear()
    results = [check_data_freshness(), check_evening_log(), check_revisions()]
    verdict_readiness()
    n_warn = len(results) - sum(results)
    print(f"\nhealth: {'모두 정상' if n_warn == 0 else f'경고 {n_warn}건 — 위 [WARN] 확인'}")
    _publish_alert(_WARNINGS)
