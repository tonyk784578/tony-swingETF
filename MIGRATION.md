# GPU 워크스테이션 이관 절차 (2026-09-04 10:00 예정)

이유: 노트북은 크론 시각(08:30 아침 / 15:20 창 / 16:20 저녁)에 맞춰 켜야 해서
유실이 잦다. 항시 가동 워크스테이션으로 옮긴다. 옮기는 것은 **파일 + crontab 5줄**뿐이고
코드·데이터 형식은 그대로다 (절대 경로 `/home/user/Project/TonySwingETF` 유지 권장 —
crontab 라인이 이 경로를 쓴다).

## 0. 이관일 아침 (노트북, 09:55 이전)

- 노트북을 08:30 전에 켜 두거나, 늦어도 09:55 전에 켠다. 아침 프리뷰
  (`preview_signals.csv`·`market_snapshots.csv`)는 **소급 불가**라 이관일 아침도
  노트북이 한 번 받아야 한다. 확인: `cat paper/logs/.morning_done` 이 오늘 날짜.
- 09-04 09:00 이후엔 워크스테이션에서 압축본을 풀기 전까지 아무것도 돌지 않는다
  (15:20 창은 현재 crontab 에 미등록이라 잃는 것도 없음).

## 1. 노트북에서 압축 (10:00)

```bash
cd /home/user/Project/TonySwingETF
scripts/pack_migration.sh          # → ~/TonySwingETF_migration_<날짜시각>.tar.gz (약 200MB)
```

- 아침 스탬프가 오늘이 아니면 경고를 띄운다 — 그 경우 아침 루틴 먼저
  (`scripts/run_morning_catchup.sh`) 돌리고 다시 압축.
- 압축본에 `.env`(KIS 자격증명)가 들어 있다. USB·직접 복사(scp)로만 옮기고
  클라우드 동기화 폴더(2Brain 등)에 두지 말 것.
- Windows 탐색기 경로: `\\wsl$\Ubuntu\home\user\TonySwingETF_migration_*.tar.gz`

## 2. 노트북 크론 정지 (압축 직후 — 중요)

두 기기가 같은 저녁 루틴을 돌리면 (a) 같은 날짜 "장부 스냅샷" 커밋이 양쪽에 생겨
git push 가 충돌하고, (b) 15:20 창·아침 실행기가 KIS 토큰을 서로 폐기시킨다.

```bash
crontab -l | sed 's|^\([^#].*TonySwingETF\)|# [2026-09-04 이관] \1|' | crontab -
crontab -l | grep TonySwingETF      # 4줄 전부 # 로 시작하는지 확인
```

(SwingETF·sandwitch-platform 라인은 그대로 둔다 — 이 프로젝트와 무관.)

## 3. 워크스테이션 설치

```bash
# 3-1 시간대 — 크론 시각·스탬프 날짜·로그 이름이 전부 로컬 시각 기준
sudo timedatectl set-timezone Asia/Seoul && date      # KST 로 찍혀야 함

# 3-2 풀기 (경로 유지)
mkdir -p /home/user/Project && cd /home/user/Project
tar -xzf ~/TonySwingETF_migration_*.tar.gz
cd TonySwingETF && chmod 600 .env .env.bak.premigration data/.kis_token.json

# 3-3 파이썬 환경 (노트북은 3.12.3 — 3.12 계열 권장, lightgbm 은 libgomp 필요)
sudo apt-get install -y python3.12 python3.12-venv libgomp1 git
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt   # 실패 시 requirements.txt 로 대체

# 3-4 git 신원 + push 자격증명 (저녁 루틴이 매일 push)
git config user.name tonyk7845 && git config user.email tonyk7845@gmail.com
git config --global credential.helper store
git push --dry-run        # 토큰 물어보면 GitHub PAT 입력 (노트북 ~/.git-credentials 복사도 가능)

# 3-5 검증
.venv/bin/pytest -q                          # 전부 통과
.venv/bin/python -m src.main verify          # 날짜 정렬 검증
.venv/bin/python -m src.main health          # 신선도·판정 준비 상태 표시
.venv/bin/python -m src.main trade --auto    # 창 밖이면 가드가 스킵 — KIS 토큰 발급 확인용

# 3-6 crontab (이 프로젝트 5줄 — 15:20 창 포함)
crontab -l 2>/dev/null > /tmp/cron.bak || true
cat /tmp/cron.bak scripts/crontab.workstation | crontab -
crontab -l | grep TonySwingETF
```

## 4. 이관일 오후 확인 (워크스테이션)

- 15:21~15:27 `paper/logs/2026-09-04.log` 에 `close window` 헤더 — 3-6 이
  15:20 전에 끝났을 때만. 못 맞추면 모의 제출만 빠지고 장부는 무영향.
- 16:20 이후 `cat paper/logs/.evening_done` == 2026-09-04, 로그에 `daily push: ok`.
  놓쳐도 23:55 까지 10분마다 재시도하고, 그것도 놓치면 다음 저녁이 소급 기록한다.
- `git log -1` 이 "장부 스냅샷 2026-09-04 (자동)".

## 달라지는 것 / 안 달라지는 것

- Windows 풍선 알림(`health._notify_desktop`)은 powershell.exe 가 없으면 조용히
  건너뛴다 — 워크스테이션이 WSL 이 아니면 알림만 없어지고 ALERT.md·로그는 동일.
- 분봉(`data/minute_*_5m.parquet`)은 60일 밖 소급 불가 — 압축본에 포함됐으므로
  풀기만 하면 연속. `data/xs/`(172MB)는 기각 실험 재현용 캐시 — 함께 옮김.
- 판정 기준·후보·장부는 어떤 것도 바뀌지 않는다. 기기 이동은 통계와 무관.
