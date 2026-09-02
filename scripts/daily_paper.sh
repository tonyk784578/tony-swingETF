#!/usr/bin/env bash
# 섀도 트래킹 일일 루틴. cron 또는 수동 실행.
#   daily_paper.sh morning : 개장 전 — 데이터 갱신 + 조건 발동 미리보기
#   daily_paper.sh close   : 15:20 — 종가 매수/청산 제출 (마감 동시호가 근사)
#   daily_paper.sh evening : 마감 후 — 데이터 갱신 + 장부 기록 + 성과 요약
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/paper/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$(date +%F).log"
find "$LOG_DIR" -name '*.log' -mtime +180 -delete   # 로그 180일 보존

cd "$ROOT"
case "${1:-evening}" in
  morning)
    {
      echo "===== morning preview $(date '+%F %T') ====="
      "$PY" -m src.main paper --preview --force
      # Phase D 실행기 (2026-08-26): 오늘 신호를 주문 계획으로 번역해 기록.
      # 모드는 config ops.trade_mode (dry_run 기본 / live_mock 전환 시 모의 제출).
      # 실패해도 아침 루틴은 계속 (증빙 누락은 exec_plan.csv 공백으로 드러남)
      "$PY" -m src.main trade --auto || echo "[WARN] trade step failed"
    } >>"$LOG" 2>&1
    ;;
  close)
    {
      echo "===== close window $(date '+%F %T') ====="
      # 15:20 실행 창 (2026-09-02): volbreak 트리거 도달·overnight 양봉 종가
      # 매수 + swing 청산 신호 종가 매도. 창 밖 실행은 파이썬 쪽 가드가 스킵.
      "$PY" -m src.main trade --auto --close-window || echo "[WARN] close window failed"
    } >>"$LOG" 2>&1
    ;;
  evening)
    {
      echo "===== evening ledger $(date '+%F %T') ====="
      "$PY" -m src.main download --force
      "$PY" -m src.main minute
      "$PY" -m src.main paper
      # 통합 슬리브 뷰 갱신 (2026-08-26): 중복 보유·슬리브 상관 일일 기록
      "$PY" -m src.main sleeves || echo "[WARN] sleeves step failed"
      "$PY" -m src.main health
      # 장부 스냅샷 자동 커밋 (포워드 기록 보호 — paper/ 경로만, 로컬 커밋)
      git -C "$ROOT" add paper/ >/dev/null 2>&1 || true
      git -C "$ROOT" commit -q -m "장부 스냅샷 $(date +%F) (자동)" -- paper/ \
        >/dev/null 2>&1 || true
      # 일일 오프사이트 백업 (2026-08-26 주간→일일 개정) — preview_signals·
      # market_snapshots 는 소급 불가 자산이라, 주간 push 는 기기 사고 시 최대
      # 1주치를 잃는다. 매일 push 로 노출 창을 하루로 줄인다.
      # 실패는 삼키되 로그에 남긴다 (health의 커밋 나이 감시와 별개)
      if git -C "$ROOT" push >/dev/null 2>&1; then
        echo "daily push: ok"
      else
        echo "[WARN] daily push failed — 수동 push 필요"
      fi
    } >>"$LOG" 2>&1
    ;;
  *)
    echo "usage: $0 {morning|close|evening}" >&2
    exit 1
    ;;
esac
