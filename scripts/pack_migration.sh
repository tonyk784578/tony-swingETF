#!/usr/bin/env bash
# 이관용 전체 압축 (2026-09-04 GPU 워크스테이션 이관).
#   scripts/pack_migration.sh            → ~/TonySwingETF_migration_<날짜시각>.tar.gz
#   scripts/pack_migration.sh /경로/디렉토리   → 지정 디렉토리에 생성
#
# 포함: git 저장소(.git), 소스, config, paper/ 전체(장부·로그·성공 스탬프),
#       data/ 전체(분봉 캐시는 60일 밖 소급 불가 — 반드시 포함), results/,
#       .env(자격증명 — 압축본 취급 주의), ALERT.md/STATUS.md(상태 파일).
# 제외: .venv(807MB — requirements.lock.txt 로 재생성), 캐시 디렉토리.
#
# 실행 시점: 아침 프리뷰(08:30~09:55)가 끝난 뒤 — 그날 preview_signals·
# market_snapshots 가 압축본에 들어가야 한다 (소급 불가 자산).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$HOME}"
STAMP="$(date +%Y%m%d_%H%M)"
OUT="$OUT_DIR/TonySwingETF_migration_${STAMP}.tar.gz"

cd "$ROOT"
"$ROOT/.venv/bin/pip" freeze >requirements.lock.txt 2>/dev/null || true

if [[ "$(cat paper/logs/.morning_done 2>/dev/null || true)" != "$(date +%F)" ]]; then
  echo "[WARN] 오늘($(date +%F)) 아침 프리뷰 성공 스탬프가 없다 — 오늘치 preview_signals/" >&2
  echo "       market_snapshots 미기록 상태. 평일이면 아침 루틴 완료 후 다시 압축할 것." >&2
fi

tar -C "$(dirname "$ROOT")" \
  --exclude='TonySwingETF/.venv' \
  --exclude='TonySwingETF/.pytest_cache' \
  --exclude='TonySwingETF/.ruff_cache' \
  --exclude='__pycache__' \
  -czf "$OUT" TonySwingETF

chmod 600 "$OUT"   # .env 포함 — 소유자만 읽기
echo "생성: $OUT ($(du -h "$OUT" | cut -f1))"
echo "검증: tar -tzf \"$OUT\" | wc -l  → 파일 수"
echo "체크섬: $(sha256sum "$OUT" | cut -c1-16)…  (워크스테이션에서 sha256sum 으로 대조)"
