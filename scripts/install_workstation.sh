#!/usr/bin/env bash
# 워크스테이션 1회 설치 (2026-09-04 이관). 압축본을 푼 디렉토리 안에서 실행:
#   scripts/install_workstation.sh
# 하는 일: 시간대 확인 → 자격증명 권한 → venv 재생성 → git 신원/push 자격증명 →
#          검증(pytest·verify·health) → 오늘 아침 프리뷰 미실행이면 지금 실행 →
#          crontab 5줄 등록(경로는 실제 위치로 치환). 재실행해도 안전(멱등).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ok()   { echo "  [ok]   $*"; }
warn() { echo "  [WARN] $*"; }
die()  { echo "  [STOP] $*" >&2; exit 1; }

echo "== 1/7 시간대 =="
if [[ "$(date +%Z)" != "KST" ]]; then
  if command -v timedatectl >/dev/null && sudo -n true 2>/dev/null; then
    sudo timedatectl set-timezone Asia/Seoul
  elif command -v timedatectl >/dev/null; then
    echo "  시간대를 KST 로 바꿉니다 (sudo 암호):"; sudo timedatectl set-timezone Asia/Seoul
  fi
  [[ "$(date +%Z)" == "KST" ]] || die "시간대가 KST 가 아니다 ($(date +%Z)). 'sudo timedatectl set-timezone Asia/Seoul' 후 재실행."
fi
ok "시간대 $(date '+%Z %F %T')"

echo "== 2/7 자격증명 권한 =="
[[ -f .env ]] || die ".env 가 없다 — 압축본이 불완전하다."
chmod 600 .env .env.bak.premigration data/.kis_token.json 2>/dev/null || true
ok ".env 600"

echo "== 3/7 파이썬 venv =="
PYBIN=""
for c in python3.12 python3.13 python3.11 python3; do
  command -v "$c" >/dev/null && { PYBIN="$c"; break; }
done
[[ -n "$PYBIN" ]] || die "python3 가 없다. 'sudo apt-get install -y python3.12 python3.12-venv libgomp1' 후 재실행."
if [[ ! -x .venv/bin/python ]]; then
  "$PYBIN" -m venv .venv || die "venv 생성 실패 — 'sudo apt-get install -y ${PYBIN}-venv' 후 재실행."
fi
.venv/bin/pip install -q --upgrade pip >/dev/null 2>&1 || true
if ! .venv/bin/pip install -q -r requirements.lock.txt 2>/tmp/pip_err.log; then
  warn "고정 버전 설치 실패 → 최소 요구사항으로 대체 (tail /tmp/pip_err.log)"
  .venv/bin/pip install -q -r requirements.txt || die "의존성 설치 실패"
fi
.venv/bin/python -c "import lightgbm" 2>/dev/null || warn "lightgbm 로드 실패 — 'sudo apt-get install -y libgomp1' (ML 재현 전용, 일일 루틴엔 무관)"
ok "venv $(.venv/bin/python --version)"

echo "== 4/7 git 신원 + push 자격증명 =="
command -v git >/dev/null || die "git 이 없다."
git config user.name  >/dev/null || git config user.name  tonyk7845
git config user.email >/dev/null || git config user.email tonyk7845@gmail.com
git config --global credential.helper store
if [[ ! -f "$HOME/.git-credentials" && -f "$ROOT/../.git-credentials" ]]; then
  mv "$ROOT/../.git-credentials" "$HOME/.git-credentials" && chmod 600 "$HOME/.git-credentials"
fi
if git push --dry-run >/dev/null 2>&1; then ok "push 가능"; else warn "push 불가 — 저녁 루틴이 [WARN] daily push failed 를 남긴다. 'git push' 를 한 번 수동 실행해 토큰 입력."; fi

echo "== 5/7 검증 =="
.venv/bin/pytest -q >/tmp/pytest.log 2>&1 && ok "pytest 통과" || { tail -5 /tmp/pytest.log; die "pytest 실패"; }
.venv/bin/python -m src.main verify >/dev/null 2>&1 && ok "날짜 정렬 verify 통과" || die "verify 실패"

echo "== 6/7 오늘 아침 프리뷰 =="
TODAY="$(date +%F)"
if [[ "$(cat paper/logs/.morning_done 2>/dev/null || true)" == "$TODAY" ]]; then
  ok "노트북에서 이미 완료 ($TODAY)"
elif [[ $(date +%u) -le 5 && 10#$(date +%H) -lt 16 ]]; then
  echo "  노트북이 못 받은 오늘치 아침 프리뷰를 지금 기록합니다 (개장 후면 late 플래그 — 정상)."
  scripts/run_morning_catchup.sh && ok "아침 프리뷰 기록 완료" || warn "아침 프리뷰 실패 — tail paper/logs/$TODAY.log"
else
  ok "오늘은 아침 루틴 대상 아님 (주말 또는 16시 이후)"
fi

echo "== 7/7 crontab =="
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v 'TonySwingETF/scripts/' >"$TMP" || true
grep -v '^#' scripts/crontab.workstation | sed "s|/home/user/Project/TonySwingETF|$ROOT|g" >>"$TMP"
crontab "$TMP" && rm -f "$TMP"
echo "$(crontab -l | grep -c 'TonySwingETF/scripts/')줄 등록:"; crontab -l | grep 'TonySwingETF/scripts/' | cut -c1-60
.venv/bin/python -m src.main health >/dev/null 2>&1 && ok "health 실행" || warn "health 경고 — ALERT.md 확인 (09-04 만료 예정인 데이터 정정 경고일 수 있음)"

echo
echo "설치 완료. 오늘 확인 지점: 15:21 창 → paper/logs/$TODAY.log 에 'close window',"
echo "16:20 이후 cat paper/logs/.evening_done == $TODAY, 로그에 'daily push: ok'."
