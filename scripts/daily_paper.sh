#!/usr/bin/env bash
# 섀도 트래킹 일일 루틴. cron 또는 수동 실행.
#   daily_paper.sh morning : 개장 전 — 데이터 갱신 + 조건 발동 미리보기
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
    } >>"$LOG" 2>&1
    ;;
  evening)
    {
      echo "===== evening ledger $(date '+%F %T') ====="
      "$PY" -m src.main download --force
      "$PY" -m src.main minute
      "$PY" -m src.main paper
      "$PY" -m src.main health
      # 장부 스냅샷 자동 커밋 (포워드 기록 보호 — paper/ 경로만, 로컬 커밋.
      # 변경 없으면 조용히 넘어간다. 푸시는 수동)
      git -C "$ROOT" add paper/ >/dev/null 2>&1 || true
      git -C "$ROOT" commit -q -m "장부 스냅샷 $(date +%F) (자동)" -- paper/ \
        >/dev/null 2>&1 || true
    } >>"$LOG" 2>&1
    ;;
  *)
    echo "usage: $0 {morning|evening}" >&2
    exit 1
    ;;
esac
