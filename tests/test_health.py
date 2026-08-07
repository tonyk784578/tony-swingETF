"""헬스체크 — PS 알림 인젝션 방어 및 정제 함수 테스트."""

from src.health import _ps_quote


def test_single_quotes_removed():
    # PS 단일따옴표 문자열의 유일한 탈출 문자
    assert "'" not in _ps_quote("경고 'test' 발생")


def test_control_chars_stripped():
    s = _ps_quote("경고\r\n줄바꿈\x00널")
    assert "\n" not in s and "\r" not in s and "\x00" not in s


def test_length_capped():
    assert len(_ps_quote("x" * 500)) == 200
    assert len(_ps_quote("t" * 500, limit=60)) == 60


def test_injection_payload_neutralized():
    # 닫는 따옴표로 문자열을 탈출해 명령을 잇는 페이로드가 무력화되는지
    payload = "'); Remove-Item -Recurse C:\\ ; ('"
    safe = _ps_quote(payload)
    assert "'" not in safe          # 문자열 탈출 불가
    # $, 백틱, 세미콜론은 단일따옴표 문자열 내에서 리터럴 — 남아 있어도 무해
