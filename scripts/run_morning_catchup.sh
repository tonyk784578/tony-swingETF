#!/usr/bin/env bash
# 아침 프리뷰 실행기 (정시 + 캐치업 겸용).
#
# 배경: WSL 부팅이 08:30 이후인 날이 상례라(2026-08-04 09:35 / 08-05 09:02 /
# 08-06 08:38 / 08-07 08:36) 08:30 정시 크론만으로는 아침 프리뷰가 거의 매일
# 유실된다. cron 은 꺼져 있던 시각의 작업을 나중에 실행하지 않으므로 부팅 후
# 재시도가 필요하다. crontab 은 이 스크립트를 08:30~09:55 사이 5분 간격으로 부르고,
# 하루 첫 '성공' 이후로는 스탬프를 보고 스스로 건너뛴다.
#
# 스탬프는 로그 문자열이 아니라 종료코드로 찍는다. daily_paper.sh 는 헤더를 먼저
# 로그에 적고 파이썬을 실행하므로, 로그에서 "morning preview" 를 찾는 방식은
# 실패한 실행까지 완료로 오인해 재시도를 영영 막는다.
#
# 저녁(16:20)은 캐치업이 필요 없다 — 리플레이 방식이라 놓친 날을 다음 실행이
# 자동 소급 기록한다. 아침 프리뷰만 그날 못 보면 그날치가 사라진다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$ROOT/paper/logs/.morning_done"   # paper/logs/ 는 gitignore 대상
TODAY="$(date +%F)"

mkdir -p "$(dirname "$STAMP")"
if [[ "$(cat "$STAMP" 2>/dev/null || true)" == "$TODAY" ]]; then
  exit 0
fi

if "$ROOT/scripts/daily_paper.sh" morning; then
  echo "$TODAY" >"$STAMP"
else
  rc=$?
  echo "[WARN] morning preview 실패 (rc=$rc) — 다음 5분 슬롯에 재시도" \
    >>"$ROOT/paper/logs/$TODAY.log"
  exit "$rc"
fi
