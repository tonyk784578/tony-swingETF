# 미국장 갭 패턴 백테스트 리포트

생성일: 2026-08-06 | 기간: 2015-01-01 ~ 현재 | 거래비용: 왕복 0.2%

## 방법론

- 전략: 조건 충족 시 **당일 시가 매수 → 당일 종가 매도**, 순수익 = (종가/시가 − 1) − 비용
- 룩어헤드 방지: 미국 수익률은 한국 거래일 D보다 **미국 현지 날짜가 앞선 마지막 봉**만 사용
  (전수 검증 통과). 국내 지표는 D−1까지만, 당일 시가는 갭 계산에만 사용.
- 조건: SOX/NVDA/KOSPI 수익률 구간, 20일 이평 위/아래, 시가 갭 구간 — 단독 + 2개 조합
  + 사전 지정 3개 조합. N < 30 조건은 순위 평가 제외.
- 과최적화 검증: 2021-01-01 기준 전/후반 분할, 전반부 상위 조건의 후반부 성과 확인.

## SamsungElec — 상위 조건 (N ≥ 30)

| 조건 | N | 평균순수익 | 승률 | 누적수익 | MDD | t-stat |
|---|---|---|---|---|---|---|
| nvda_down2 & gap_up_small | 74 | +0.454% | 48.6% | +38.5% | -8.0% | 2.45 |
| sox_down1 & gap_up_small | 105 | +0.312% | 50.5% | +36.8% | -10.1% | 1.96 |
| nvda_down2 & gap_down_big | 183 | +0.241% | 51.4% | +51.3% | -13.8% | 1.94 |
| sox_flat & nvda_down2 | 89 | +0.218% | 48.3% | +20.3% | -7.7% | 1.42 |
| nvda_down2 & kospi_down | 267 | +0.198% | 49.1% | +63.3% | -19.7% | 1.90 |
| kospi_down & gap_down_big | 170 | +0.163% | 52.9% | +27.3% | -24.4% | 1.02 |
| nvda_down2 & ma20_below | 279 | +0.093% | 45.5% | +25.3% | -29.9% | 0.99 |
| sox_down1 & gap_down_big | 235 | +0.064% | 48.9% | +11.5% | -26.1% | 0.52 |
| nvda_down2 | 490 | +0.063% | 45.9% | +28.0% | -43.2% | 0.87 |
| sox_down1 & kospi_down | 383 | +0.032% | 47.8% | +5.9% | -39.2% | 0.34 |
| sox_down1 & nvda_down2 | 391 | +0.024% | 45.3% | +4.1% | -41.5% | 0.29 |
| nvda_down2 & ma20_above | 211 | +0.024% | 46.4% | +2.1% | -32.7% | 0.21 |
| gap_down_big & ma20_below | 180 | +0.022% | 51.1% | +1.0% | -35.9% | 0.16 |
| sox_flat & gap_down_big | 102 | -0.014% | 56.9% | -3.5% | -21.8% | -0.07 |
| nvda_down2 & gap_down | 387 | -0.016% | 45.5% | -10.1% | -36.4% | -0.22 |

![equity](equity_curves_SamsungElec.png)

## SKHynix — 상위 조건 (N ≥ 30)

| 조건 | N | 평균순수익 | 승률 | 누적수익 | MDD | t-stat |
|---|---|---|---|---|---|---|
| nvda_down2 & kospi_down | 267 | +0.428% | 56.2% | +193.9% | -18.8% | 3.22 |
| nvda_down2 & gap_down_big | 257 | +0.247% | 53.3% | +79.2% | -13.6% | 1.99 |
| nvda_down2 & ma20_below | 264 | +0.215% | 53.0% | +65.5% | -13.5% | 1.60 |
| sox_down1 & nvda_down2 | 391 | +0.213% | 52.9% | +111.2% | -15.9% | 2.01 |
| nvda_down2 & gap_down | 405 | +0.209% | 53.1% | +116.1% | -17.6% | 2.16 |
| sox_down1 & gap_up_small | 71 | +0.196% | 50.7% | +12.7% | -19.7% | 0.69 |
| sox_down1 & kospi_down | 383 | +0.192% | 51.7% | +87.3% | -44.8% | 1.58 |
| sox_down1 & gap_down & kospi_down | 306 | +0.188% | 52.3% | +64.7% | -41.3% | 1.47 |
| nvda_down2 | 490 | +0.185% | 52.4% | +122.7% | -17.5% | 1.98 |
| sox_flat & nvda_down2 | 89 | +0.165% | 51.7% | +14.0% | -12.6% | 0.82 |
| nvda_down2 & ma20_above | 226 | +0.150% | 51.8% | +34.6% | -23.9% | 1.17 |
| kospi_down & gap_down_big | 269 | +0.087% | 51.3% | +17.7% | -34.2% | 0.62 |
| sox_down1 & gap_down_big | 358 | +0.076% | 51.1% | +20.8% | -27.9% | 0.66 |
| nvda_down2 & gap_up_small | 60 | +0.024% | 46.7% | -0.1% | -11.4% | 0.08 |
| sox_down1 & ma20_below | 367 | +0.017% | 49.0% | -3.3% | -35.1% | 0.14 |

![equity](equity_curves_SKHynix.png)

## SamsungElecMech — 상위 조건 (N ≥ 30)

| 조건 | N | 평균순수익 | 승률 | 누적수익 | MDD | t-stat |
|---|---|---|---|---|---|---|
| nvda_down2 & gap_up_small | 90 | +0.385% | 51.1% | +38.2% | -8.2% | 1.63 |
| nvda_flat & gap_down_big | 120 | +0.326% | 55.0% | +38.9% | -22.7% | 1.09 |
| gap_down_big & ma20_above | 187 | +0.306% | 54.0% | +63.1% | -20.9% | 1.39 |
| kospi_down & gap_down_big | 172 | +0.301% | 55.8% | +54.3% | -32.0% | 1.27 |
| sox_down1 & gap_down_big | 252 | +0.198% | 52.0% | +47.3% | -25.1% | 1.05 |
| sox_flat & gap_down_big | 66 | +0.195% | 59.1% | +9.7% | -19.9% | 0.49 |
| nvda_down2 & kospi_down | 267 | +0.186% | 50.9% | +48.5% | -33.0% | 1.12 |
| sox_flat & nvda_down2 | 89 | +0.168% | 46.1% | +13.6% | -18.8% | 0.71 |
| sox_down1 & gap_down & kospi_down | 288 | +0.135% | 49.7% | +33.4% | -43.2% | 0.87 |
| nvda_down2 & gap_down_big | 201 | +0.118% | 52.2% | +18.0% | -19.9% | 0.63 |
| gap_down_big | 338 | +0.115% | 53.0% | +26.0% | -43.5% | 0.70 |
| nvda_down2 & ma20_above | 241 | +0.065% | 46.9% | +9.1% | -36.8% | 0.42 |
| sox_down1 & kospi_down | 383 | +0.025% | 48.0% | -4.4% | -49.4% | 0.18 |
| nvda_flat & gap_up_big | 287 | +0.020% | 40.8% | -6.8% | -44.0% | 0.11 |
| nvda_up3 & gap_up_small | 143 | +0.009% | 47.6% | -1.2% | -25.9% | 0.06 |

![equity](equity_curves_SamsungElecMech.png)

## Split 검증 (전반부 상위 10개 → 후반부)

| 종목 | 조건 | 전반 순위 | 전반 평균 | 후반 평균 | 후반 N | 부호 유지 |
|---|---|---|---|---|---|---|
| SamsungElec | gap_down_big & ma20_below | 1 | +0.524% | -0.438% | 94 | X |
| SamsungElec | kospi_down & gap_down_big | 2 | +0.473% | -0.169% | 82 | X |
| SamsungElec | nvda_down2 & gap_down_big | 3 | +0.410% | +0.083% | 95 | O |
| SamsungElec | nvda_down2 & ma20_below | 4 | +0.367% | -0.087% | 168 | X |
| SamsungElec | sox_down1 & gap_down_big | 5 | +0.363% | -0.173% | 131 | X |
| SamsungElec | nvda_flat & gap_down_big | 6 | +0.345% | -0.592% | 77 | X |
| SamsungElec | gap_down_big | 7 | +0.316% | -0.325% | 189 | X |
| SamsungElec | sox_flat & gap_down_big | 8 | +0.294% | -0.491% | 40 | X |
| SamsungElec | sox_flat & nvda_down2 | 9 | +0.274% | +0.161% | 44 | O |
| SamsungElec | nvda_down2 & kospi_down | 10 | +0.271% | +0.141% | 149 | O |
| SKHynix | nvda_down2 & kospi_down | 1 | +0.460% | +0.403% | 149 | O |
| SKHynix | sox_down1 & gap_down & kospi_down | 2 | +0.377% | +0.041% | 172 | O |
| SKHynix | sox_down1 & gap_up_small | 3 | +0.357% | +0.000% | 32 | O |
| SKHynix | sox_down1 & kospi_down | 4 | +0.331% | +0.080% | 211 | O |
| SKHynix | sox_down1 & nvda_down2 | 5 | +0.273% | +0.173% | 234 | O |
| SKHynix | nvda_down2 & gap_down_big | 6 | +0.255% | +0.242% | 162 | O |
| SKHynix | nvda_down2 & ma20_below | 7 | +0.252% | +0.183% | 143 | O |
| SKHynix | nvda_down2 & gap_down | 8 | +0.223% | +0.200% | 241 | O |
| SKHynix | kospi_down & gap_down_big | 9 | +0.195% | -0.002% | 147 | X |
| SKHynix | sox_down1 & gap_down_big | 10 | +0.193% | +0.001% | 217 | O |
| SamsungElecMech | sox_flat & gap_down_big | 1 | +0.607% | -0.149% | 36 | X |
| SamsungElecMech | kospi_down & gap_down_big | 2 | +0.542% | +0.115% | 97 | O |
| SamsungElecMech | sox_down1 & gap_down & kospi_down | 3 | +0.456% | -0.121% | 160 | X |
| SamsungElecMech | gap_down_big & ma20_below | 4 | +0.381% | -0.580% | 79 | X |
| SamsungElecMech | nvda_up3 & gap_up_small | 5 | +0.356% | -0.214% | 87 | X |
| SamsungElecMech | nvda_flat & gap_down_big | 6 | +0.326% | +0.326% | 66 | O |
| SamsungElecMech | nvda_down2 & gap_up_small | 7 | +0.292% | +0.465% | 48 | O |
| SamsungElecMech | gap_down_big | 8 | +0.291% | +0.005% | 208 | O |
| SamsungElecMech | sox_down1 & gap_down_big | 9 | +0.283% | +0.147% | 157 | O |
| SamsungElecMech | sox_down1 & kospi_down | 10 | +0.254% | -0.161% | 211 | X |

## 로버스트니스 — 생존 조건 비용 민감도

| 종목 | 조건 | 왕복비용 | N | 평균순수익 | 승률 | t-stat |
|---|---|---|---|---|---|---|
| SKHynix | nvda_down2 & kospi_down | 0.2% | 267 | +0.428% | 56.2% | 3.22 |
| SKHynix | nvda_down2 & kospi_down | 0.3% | 267 | +0.328% | 54.7% | 2.47 |
| SKHynix | nvda_down2 & kospi_down | 0.5% | 267 | +0.128% | 49.1% | 0.96 |
| SKHynix | sox_down1 & nvda_down2 | 0.2% | 391 | +0.213% | 52.9% | 2.01 |
| SKHynix | sox_down1 & nvda_down2 | 0.3% | 391 | +0.113% | 51.9% | 1.07 |
| SKHynix | sox_down1 & nvda_down2 | 0.5% | 391 | -0.087% | 45.8% | -0.82 |
| SKHynix | nvda_down2 & gap_down | 0.2% | 405 | +0.209% | 53.1% | 2.16 |
| SKHynix | nvda_down2 & gap_down | 0.3% | 405 | +0.109% | 51.6% | 1.13 |
| SKHynix | nvda_down2 & gap_down | 0.5% | 405 | -0.091% | 45.2% | -0.93 |

## 로버스트니스 — 생존 조건 연도별 안정성

| 종목 | 조건 | 연수 | 음수 연수 | 최악 연도 | 최악 평균 |
|---|---|---|---|---|---|
| SKHynix | nvda_down2 & gap_down | 12 | 2 | 2015 | -0.073% |
| SKHynix | nvda_down2 & kospi_down | 12 | 1 | 2024 | -0.180% |
| SKHynix | sox_down1 & nvda_down2 | 12 | 2 | 2024 | -0.064% |

연도별 상세는 `robustness_yearly.csv` 참고.

## 결론

- 전반부 상위 조건 30개 중 후반부에서 양(+)의 평균수익 부호를 유지한 조건: **17개** (57%)
- 이 중 전체 기간 t-stat ≥ 2 (통계적으로 유의미한 수준)인 조건: **3개**
  - SKHynix: `nvda_down2 & kospi_down` (전반 +0.460% / 후반 +0.403%)
  - SKHynix: `sox_down1 & nvda_down2` (전반 +0.273% / 후반 +0.173%)
  - SKHynix: `nvda_down2 & gap_down` (전반 +0.223% / 후반 +0.200%)
- **판정: 위 조건들은 비용 차감 후에도 표본 수·후반부 유지·유의성 3박자를 충족한다. 실전 후보 여부는 위 로버스트니스 표(비용 민감도·연도별 안정성)로 판단할 것 — 특히 왕복 0.3%에서 t ≥ 2를 유지하는지가 관건이다.**
