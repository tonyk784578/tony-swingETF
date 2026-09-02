#!/usr/bin/env bash
# 15:20 실행 창 (정시 + 창 안 재시도 겸용 — 2026-09-02).
#
# 배경: volbreak 종가 매수·overnight 종가 매수·swing 종가 청산은 마감
# 동시호가(15:20~15:30) 직전에만 제출할 수 있다. 창이 10여 분뿐이라 아침/저녁
# 캐치업처럼 늦게 돌 수 없다 — crontab 이 15:21/15:24/15:27 세 번 부르고 하루
# 첫 성공 후 스탬프로 건너뛴다. 세 슬롯을 모두 놓친 날(WSL 다운)은 모의 제출만
# 빠지고 섀도 장부·판정은 영향 없다 (파이썬 쪽 창 밖 가드가 이중 방어).
#
# 스탬프는 종료코드 기반 (run_morning_catchup.sh 관례 — 로그 매칭은 실패한
# 실행을 완료로 오인한다).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$ROOT/paper/logs/.close_done"   # paper/logs/ 는 gitignore 대상
TODAY="$(date +%F)"

mkdir -p "$(dirname "$STAMP")"
if [[ "$(cat "$STAMP" 2>/dev/null || true)" == "$TODAY" ]]; then
  exit 0
fi

if "$ROOT/scripts/daily_paper.sh" close; then
  echo "$TODAY" >"$STAMP"
else
  rc=$?
  echo "[WARN] close window 실패 (rc=$rc) — 다음 슬롯에 재시도" \
    >>"$ROOT/paper/logs/$TODAY.log"
  exit "$rc"
fi
