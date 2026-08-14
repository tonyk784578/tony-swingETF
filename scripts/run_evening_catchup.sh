#!/usr/bin/env bash
# 저녁 장부 실행기 (정시 + 캐치업 겸용).
#
# 배경: 2026-08-14(금) 16:20 저녁 실행이 통째로 유실됐다. 아침 08:40 은 정상
# 실행됐으므로 그 사이에 WSL/cron 이 내려간 것 (cron.service 재시작 17:53).
# cron 은 꺼져 있던 시각의 작업을 나중에 실행하지 않으므로, 아침과 같은
# 재시도 구조가 저녁에도 필요하다.
#
# 유실 시 손해는 즉사가 아니라 지연이다 — 일봉·장부는 리플레이라 다음 저녁이
# 소급하고, 분봉만 yfinance 60일 한도라 장기 방치 때 영구 결손이 된다.
# 그래도 매일 도는 편이 낫고, 특히 금요일은 주간 오프사이트 push 가 걸려 있다.
#
# 창은 16:20~23:55 로 넓게 잡는다: 마감 후 언제 켜든 그날 안에 한 번은 잡는다
# (당일 봉은 16시 이후 확정 — confirmed_cutoff 규칙과 일치하므로 늦게 돌아도
# 결과가 같다). 스탬프는 아침과 같은 이유로 로그 문자열이 아니라 종료코드로 찍는다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$ROOT/paper/logs/.evening_done"   # paper/logs/ 는 gitignore 대상
TODAY="$(date +%F)"

mkdir -p "$(dirname "$STAMP")"
if [[ "$(cat "$STAMP" 2>/dev/null || true)" == "$TODAY" ]]; then
  exit 0
fi

if "$ROOT/scripts/daily_paper.sh" evening; then
  echo "$TODAY" >"$STAMP"
else
  rc=$?
  echo "[WARN] evening ledger 실패 (rc=$rc) — 다음 10분 슬롯에 재시도" \
    >>"$ROOT/paper/logs/$TODAY.log"
  exit "$rc"
fi
