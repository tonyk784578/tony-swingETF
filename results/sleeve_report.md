# 통합 슬리브 리스크 뷰 (측정 전용)

생성일: 2026-09-02 | 근거: 섀도 장부 41건 +
현재 미청산 포지션 | 신호·판정 기준 불변 — 한 계좌 관점의 리스크 가시화

## 현재 보유 (전 슬리브)

| ETF | 전략 | 슬리브 | 평가 |
|---|---|---|---|
| KODEX_Gold | tom | tom | -2.92% |
| KODEX_Semicon | overnight | overnight | +0.00% |
| KODEX_Semicon | volbreak | volbreak | -1.04% |
| KODEX_Semicon | rotation2 | rotation2 | +161.53% |
| KODEX200 | rotation2 | rotation2 | +78.92% |
| TIGER_OilH | rotation2 | rotation2 | +14.88% |

**⚠ 동일 ETF 중복 보유: KODEX_Semicon** — 같은 밤 명목 노출이 전략 수만큼 배증한다. 실전 투입 시 노출 합산 캡 적용 대상.

## 슬리브 간 일간 수익 상관 (장부 실측 — 비활동일=0)

| | overnight | trend | volbreak |
|---|---|---|---|
| overnight | 1.00 | 0.10 | 0.14 |
| trend | 0.10 | 1.00 | -0.92 |
| volbreak | 0.14 | -0.92 | 1.00 |

초기 표본에선 참고치 — |ρ|>0.5 쌍은 실전에서 같은 리스크 그룹으로 취급 (등록된 운영 원칙).

## 같은 밤 동일 ETF 중복 보유 이력 (1일 회전 계열)

| ETF | 전략 조합 | 중복 밤 수 |
|---|---|---|
| KODEX_Semicon | overnight+volbreak | 1 |
