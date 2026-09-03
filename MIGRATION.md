# GPU 워크스테이션 이관 (2026-09-04 09:30) — 사용자가 할 일 4단계

옮기는 것은 파일 하나(압축본)뿐이다. 코드·판정·장부는 바뀌지 않는다.
압축본에 `.env`(KIS 자격증명)와 GitHub 토큰이 들어 있으니 **USB 또는 scp 로만** 옮기고
클라우드 동기화 폴더(2Brain 등)에는 두지 말 것.

## 1단계 — 노트북 (09:30, 명령 1개)

```bash
scripts/pack_migration.sh --handoff
```

- 결과: `~/TonySwingETF_migration_<날짜시각>.tar.gz` (약 140MB) + 이 노트북의
  TonySwingETF 크론 라인 자동 주석 처리(두 기기 동시 실행 방지).
- 노트북이 08:30~09:30 사이에 켜져 있었으면 오늘 아침 프리뷰가 압축본에 들어간다.
  못 켰어도 괜찮다 — 3단계에서 워크스테이션이 이어받아 기록한다 (경고 문구는 무시).

## 2단계 — 파일 옮기기

Windows 탐색기: `\\wsl$\Ubuntu\home\user\TonySwingETF_migration_*.tar.gz` → USB
또는 `scp ~/TonySwingETF_migration_*.tar.gz <계정>@<워크스테이션>:~/`

## 3단계 — 워크스테이션 (명령 2개)

```bash
mkdir -p ~/Project && cd ~/Project && tar -xzf ~/TonySwingETF_migration_*.tar.gz
cd TonySwingETF && scripts/install_workstation.sh
```

설치 스크립트가 순서대로 처리하고 각 단계에 [ok]/[WARN]/[STOP] 을 찍는다:
시간대 KST 확인(아니면 sudo 암호 물어봄) → 자격증명 권한 → 파이썬 venv 재생성
(수 분) → git 신원·push 자격증명 → pytest·verify 검증 → **오늘 아침 프리뷰가
없으면 지금 실행** → crontab 5줄 등록(15:20 창 포함, 경로는 실제 위치로 자동 치환).

[STOP] 이 뜨면 그 줄의 안내대로 처리 후 같은 명령을 다시 실행하면 된다(멱등).
사전 준비가 필요할 수 있는 것: `sudo apt-get install -y python3.12 python3.12-venv libgomp1 git`

## 4단계 — 당일 확인 (워크스테이션, 저녁)

```bash
cat paper/logs/.evening_done        # 2026-09-04 이면 저녁 장부 완료
grep -E "daily push|WARN" paper/logs/2026-09-04.log
```

- 15:21 창은 3단계가 15:20 전에 끝났을 때만 돈다. 놓치면 모의 제출만 빠지고 장부 무영향.
- 저녁은 23:55 까지 10분마다 재시도. 그날 통째로 놓쳐도 다음 저녁이 소급 기록.
- 이후 노트북 쪽에는 아무것도 남기지 않아도 된다 (크론은 1단계에서 정지됨).

## 알아둘 것

- Windows 풍선 알림은 powershell 이 없는 기기에서 자동으로 건너뛴다. ALERT.md·로그는 동일.
- 분봉 캐시(60일 밖 소급 불가)와 xs 캐시가 압축본에 포함돼 있어 연속성이 끊기지 않는다.
- 되돌리기: 노트북에서 `crontab -e` 로 `# [이관 ...]` 접두어를 지우면 원상복구.
