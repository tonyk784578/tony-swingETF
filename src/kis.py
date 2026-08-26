"""KIS OpenAPI 모의투자 클라이언트 — Phase D 최소 구현 (2026-08-26 계좌 이관).

계좌 내력: SwingETF가 쓰던 주식 모의계좌를 이관받음 (SwingETF 자동매매 종료
— 크론·스크립트 가드 처리 완료). 자격증명은 프로젝트 루트 .env
(KIS_APP_KEY / KIS_APP_SECRET / KIS_ACCOUNT_NO / KIS_IS_PAPER=true).

**모의 도메인 하드코딩 — 실계좌 도메인은 이 모듈에 존재하지 않는다.**
'판정 전 실거래 금지'를 코드 수준에서 강제하는 장치다. 판정 통과 +
실전 투입 게이트(config etf_paper.execution_gate) 충족 후에만 실계좌
지원을 추가한다 (그때도 별도 승인 코드 경로로).

토큰: KIS는 신규 발급 시 기존 토큰을 무효화하고 발급을 분당 1회 제한한다.
→ 파일 캐시(data/.kis_token.json, 600) 재사용, 만료 10분 전 선제 갱신,
EGW00123(만료) 수신 시 디스크 캐시 재로드 후 재발급 (SwingETF 실측 관례).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta

import requests

from .config import DATA_DIR, ROOT_DIR

MOCK_BASE = "https://openapivts.koreainvestment.com:29443"   # 모의투자 전용
_TOKEN_CACHE = DATA_DIR / ".kis_token.json"
_REFRESH_MARGIN = timedelta(minutes=10)

# 거래 TR (모의 접두 V 고정 — SwingETF broker/schemas.py 레지스트리와 동일 접미)
TR = {"balance": "VTTC8434R", "order_buy": "VTTC0802U", "order_sell": "VTTC0801U"}


def _load_env() -> dict:
    env = {}
    for line in (ROOT_DIR / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


class KIS:
    def __init__(self) -> None:
        env = _load_env()
        if env.get("KIS_IS_PAPER", "").lower() != "true":
            raise SystemExit("KIS_IS_PAPER=true 가 아님 — 이 클라이언트는 모의 전용")
        self.key = env["KIS_APP_KEY"]
        self.secret = env["KIS_APP_SECRET"]
        acct = env["KIS_ACCOUNT_NO"].replace("-", "")
        self.cano, self.prdt = acct[:8], acct[8:10]
        self._token: str | None = None
        self._expires = datetime.min
        self._load_token_cache()

    # -- 토큰 ----------------------------------------------------------- #
    def _valid(self) -> bool:
        return self._token is not None and datetime.now() < self._expires - _REFRESH_MARGIN

    def _load_token_cache(self) -> None:
        try:
            d = json.loads(_TOKEN_CACHE.read_text())
            self._token, self._expires = d["token"], datetime.fromisoformat(d["expires_at"])
        except (OSError, KeyError, ValueError):
            pass

    def _issue(self) -> None:
        r = requests.post(f"{MOCK_BASE}/oauth2/tokenP",
                          json={"grant_type": "client_credentials",
                                "appkey": self.key, "appsecret": self.secret}, timeout=15)
        r.raise_for_status()
        d = r.json()
        if "access_token" not in d:
            raise RuntimeError(f"토큰 응답 오류: {d}")
        self._token = d["access_token"]
        self._expires = datetime.now() + timedelta(seconds=int(d.get("expires_in", 86400)))
        _TOKEN_CACHE.write_text(json.dumps(
            {"token": self._token, "expires_at": self._expires.isoformat()}))
        _TOKEN_CACHE.chmod(0o600)
        print(f"[kis] 토큰 발급 (만료 {self._expires:%H:%M})", file=sys.stderr)

    def _headers(self, tr_id: str) -> dict:
        if not self._valid():
            self._issue()
        return {"authorization": f"Bearer {self._token}",
                "appkey": self.key, "appsecret": self.secret, "tr_id": tr_id}

    def _request(self, method: str, path: str, tr_id: str, *,
                 params: dict | None = None, body: dict | None = None) -> dict:
        for attempt in (1, 2):
            r = requests.request(method, f"{MOCK_BASE}{path}",
                                 headers=self._headers(tr_id),
                                 params=params, json=body, timeout=15)
            d = r.json()
            if d.get("msg_cd") == "EGW00123" and attempt == 1:   # 토큰 만료 — 승계/재발급
                self._load_token_cache()
                if not self._valid():
                    self._token = None
                continue
            if r.status_code != 200 or d.get("rt_cd") != "0":
                raise RuntimeError(f"KIS 오류 {r.status_code}/{d.get('msg_cd')}: "
                                   f"{d.get('msg1', '').strip()}")
            return d
        raise RuntimeError("KIS 요청 실패 (토큰 재시도 소진)")

    # -- 조회/주문 ------------------------------------------------------- #
    def balance(self) -> dict:
        """잔고 요약 + 보유 목록. 반환: {'cash', 'total', 'holdings': [...]}"""
        d = self._request("GET", "/uapi/domestic-stock/v1/trading/inquire-balance",
                          TR["balance"],
                          params={"CANO": self.cano, "ACNT_PRDT_CD": self.prdt,
                                  "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
                                  "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
                                  "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "01",
                                  "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""})
        s = (d.get("output2") or [{}])[0]
        holdings = [{"code": h["pdno"], "name": h["prdt_name"],
                     "qty": int(h["hldg_qty"]), "value": int(h["evlu_amt"])}
                    for h in d.get("output1", []) if int(h["hldg_qty"]) > 0]
        return {"cash": int(s.get("dnca_tot_amt", 0)),
                "total": int(s.get("tot_evlu_amt", 0)), "holdings": holdings}

    def _hashkey(self, body: dict) -> str:
        r = requests.post(f"{MOCK_BASE}/uapi/hashkey",
                          headers={"appkey": self.key, "appsecret": self.secret},
                          json=body, timeout=15)
        r.raise_for_status()
        return r.json()["HASH"]

    def order_cash(self, code: str, qty: int, side: str, price: int = 0) -> dict:
        """현금 주문. side: buy|sell, price=0 → 시장가. 반환: output(주문번호 등)."""
        if qty <= 0:
            raise ValueError(f"수량 오류: {qty}")
        body = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "PDNO": code,
                "ORD_DVSN": "01" if price <= 0 else "00",
                "ORD_QTY": str(qty), "ORD_UNPR": str(max(price, 0))}
        tr = TR["order_buy"] if side == "buy" else TR["order_sell"]
        if not self._valid():
            self._issue()
        headers = self._headers(tr)
        headers["hashkey"] = self._hashkey(body)
        r = requests.post(f"{MOCK_BASE}/uapi/domestic-stock/v1/trading/order-cash",
                          headers=headers, json=body, timeout=15)
        d = r.json()
        if r.status_code != 200 or d.get("rt_cd") != "0":
            raise RuntimeError(f"주문 실패 {d.get('msg_cd')}: {d.get('msg1', '').strip()}")
        return d.get("output", {})
