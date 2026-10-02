from __future__ import annotations

import asyncio
import json
import random

import pytest

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    FishHttpError,
    bearer,
    fish_backoff_s,
    fish_backoff_seconds,
    fish_non_json,
    fish_retry_pause,
    fish_sleep_before_retry,
    fish_unreachable,
    parse_asr_body,
    parse_fish_error,
    should_retry_fish_status,
)
from fish_audio_suite_kit.http_errors import (
    FISH_BACKOFF_CAP_S,
    FISH_RETRY_AFTER_CAP_S,
    fish_error_body,
    fish_non_object,
    fish_transport_error,
)


def test_fish_error_message_is_utf8() -> None:
    body = fish_error_body(502, "bad \ud800 byte")
    assert "\ud800" not in str(body["message"])
    str(body["message"]).encode("utf-8")
    err = FishHttpError(502, "bad \ud800 byte")
    err.message.encode("utf-8")


def test_retry_pause_stops_on_the_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("fish_audio_suite_kit.http_errors.asyncio.sleep", fake_sleep)

    async def run() -> tuple[bool, bool]:
        return await fish_retry_pause(0), await fish_retry_pause(4)

    early, last = asyncio.run(run())
    assert early is False
    assert last is True
    assert slept == [1.0]


def test_transport_error_timeout_and_blank() -> None:
    status, message = fish_transport_error(TimeoutError("late"), timed_out=True)
    assert status == 504
    assert message == "Fish request timed out"
    status, message = fish_transport_error(None, timed_out=False)
    assert status == 502
    assert message == "Fish upstream unreachable"
    status, message = fish_transport_error(ConnectionError("reset"), timed_out=False)
    assert message == "reset"


def test_fish_error_ignores_a_body_status_that_is_not_an_error() -> None:
    success = parse_fish_error(500, {"message": "nope", "status": 200})
    assert success["status"] == 500
    assert success["message"] == "nope"
    assert parse_fish_error(429, {"message": "slow", "status": 0})["status"] == 429
    assert parse_fish_error(500, {"message": "nope", "status": 402})["status"] == 402


def test_fish_validation_array_uses_the_field_messages() -> None:
    body = [
        {"loc": ["body", "text"], "msg": "Field required"},
        {"loc": ["body", "format"], "msg": "unexpected format"},
    ]
    assert parse_fish_error(422, body)["message"] == "Field required; unexpected format"
    mixed = [
        {"loc": ["body", "text"], "msg": "Field required"},
        {"loc": ["body", "format"]},
        "nope",
    ]
    assert parse_fish_error(422, mixed)["message"] == "Field required"
    encoded = parse_fish_error(422, json.dumps(body).encode())
    assert encoded["status"] == 422
    assert encoded["message"] == "Field required; unexpected format"
    assert parse_fish_error(422, b"[1, 2]")["message"] == "[1, 2]"
    wrapped = {"detail": [{"loc": ["body", "input"], "msg": "field required", "type": "missing"}]}
    assert parse_fish_error(422, wrapped)["message"] == "field required"
    named = {"detail": [{"loc": ["body", "voice"], "message": "field required"}]}
    assert parse_fish_error(422, named)["message"] == "field required"
    assert parse_fish_error(400, {"message": ["bad voice", "too long"]})["message"] == (
        "bad voice; too long"
    )


def test_one_validation_object_uses_the_field_message() -> None:
    body = {"detail": {"loc": ["body", "input"], "msg": "field required", "type": "missing"}}
    assert parse_fish_error(422, body)["message"] == "field required"
    hidden = {"message": {"message": None}, "detail": {"msg": "bad voice"}}
    assert parse_fish_error(400, hidden)["message"] == "bad voice"
    nested = {"message": {"message": [{"msg": "field required"}, {"msg": "bad format"}]}}
    assert parse_fish_error(422, nested)["message"] == "field required; bad format"
    words = {"detail": {"message": ["bad voice", "too long"]}}
    assert parse_fish_error(400, words)["message"] == "bad voice; too long"


def test_blank_fish_message_does_not_hide_detail() -> None:
    assert parse_fish_error(400, {"message": "", "detail": "voice not found"})["message"] == (
        "voice not found"
    )
    assert parse_fish_error(400, {"message": "  ", "detail": "voice not found"})["message"] == (
        "voice not found"
    )
    assert parse_fish_error(502, {"message": "", "error": {"message": "upstream"}})["message"] == (
        "upstream"
    )
    kept = parse_fish_error(400, {"message": "bad voice", "detail": "ignored"})
    assert kept["message"] == "bad voice"


def test_fish_error_falls_back_when_the_body_is_empty() -> None:
    assert parse_fish_error(500, "")["message"] == "HTTP 500"
    assert parse_fish_error(500, None)["message"] == "HTTP 500"
    assert parse_fish_error(500, b'{"message": "down"}')["message"] == "down"
    assert parse_fish_error(500, {"message": {"message": None}})["message"] == "error"


def test_parse_asr_body_text_or_502() -> None:
    data, text = parse_asr_body({"text": "hello"})
    assert data["text"] == "hello"
    assert text == "hello"
    assert parse_asr_body({})[1] == ""
    assert parse_asr_body({"text": None})[1] == ""
    with pytest.raises(FishHttpError) as missing:
        parse_asr_body(["hello"])
    assert missing.value.status == 502
    with pytest.raises(FishHttpError) as bad_text:
        parse_asr_body({"text": ["hello"]})
    assert bad_text.value.message == "Fish returned a non-object body"


def test_bearer_drops_a_key_that_would_split_the_header() -> None:
    assert bearer("sk-test") == "Bearer sk-test"
    assert bearer("  sk-test  ") == "Bearer sk-test"
    assert bearer("sk-test\r\nX-Injected: 1") == "Bearer "
    assert "\n" not in bearer("sk-test\nX-Injected: 1")
    assert bearer("sk-\ud800") == "Bearer "


def test_fish_error_shape_and_retry_policy() -> None:
    assert should_retry_fish_status(429)
    assert should_retry_fish_status(503)
    assert not should_retry_fish_status(400)
    assert not should_retry_fish_status(401)
    assert not should_retry_fish_status(402)
    assert not should_retry_fish_status(404)
    assert fish_backoff_seconds(0) == 1.0
    assert fish_backoff_seconds(2) == 4.0
    assert parse_fish_error(401, {"message": "Invalid Token", "status": 401}) == {
        "message": "Invalid Token",
        "status": 401,
    }
    assert parse_fish_error(400, b"not json")["message"] == "not json"
    assert parse_fish_error(502, {"error": {"message": "upstream"}})["message"] == "upstream"
    assert fish_error_body(402, "Insufficient credits")["status"] == 402


class _FixedRandom(random.Random):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def random(self) -> float:
        return self.value


@pytest.mark.parametrize(
    ("attempt", "draw", "expected"),
    [(0, 0.0, 0.5), (0, 1.0, 1.0), (3, 0.0, 4.0), (3, 1.0, 8.0), (20, 1.0, FISH_BACKOFF_CAP_S)],
)
def test_backoff_is_exponential_with_equal_jitter(
    attempt: int, draw: float, expected: float
) -> None:
    rng = _FixedRandom(draw)
    assert fish_backoff_s(attempt, rng=rng) == pytest.approx(expected)


def test_backoff_honors_retry_after_up_to_a_cap() -> None:
    rng = _FixedRandom(0.0)
    assert fish_backoff_s(0, retry_after=7, rng=rng) == 7
    assert fish_backoff_s(0, retry_after=10_000, rng=rng) == FISH_RETRY_AFTER_CAP_S
    assert fish_backoff_s(0, retry_after=-1, rng=rng) == 0.5
    assert fish_backoff_s(0, retry_after=float("nan"), rng=rng) == 0.5
    assert 0.5 <= fish_backoff_s(0) <= 1.0


def test_sleep_before_retry_reports_whether_to_try_again(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    assert asyncio.run(fish_sleep_before_retry(0, retry_after=3, rng=_FixedRandom(0.0))) is True
    assert slept == [3]
    assert asyncio.run(fish_sleep_before_retry(FISH_RETRY_ATTEMPTS - 1)) is False
    assert slept == [3]


def test_fish_http_error_constructors_match_the_tuple_helpers() -> None:
    for built, pair in (
        (FishHttpError.unreachable(), fish_unreachable()),
        (FishHttpError.non_json(), fish_non_json()),
        (FishHttpError.non_object(), fish_non_object()),
    ):
        assert (built.status, built.message) == pair
    timed_out = FishHttpError.timed_out()
    assert (timed_out.status, timed_out.message) == (504, "Fish request timed out")
