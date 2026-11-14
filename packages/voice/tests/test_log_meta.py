"""What a debug line may show of a Fish message: sizes and plain metadata, never text or keys."""

from __future__ import annotations

from fish_audio_suite_voice.logutil.meta import public_meta, ws_event_view


def test_ws_event_view_strips_audio() -> None:
    view = ws_event_view({"event": "audio", "audio": b"\x00" * 12, "time": 1.5})
    assert view == {"event": "audio", "audio_bytes": 12, "time": 1.5}


def test_public_meta_drops_text_and_secrets() -> None:
    meta = public_meta(
        {
            "text": "hello there friend",
            "duration": 1.2,
            "language": "en",
            "Authorization": "Bearer secret",
            "segments": [{}, {}],
        }
    )
    assert "text" not in meta
    assert meta["text_chars"] == 18
    assert meta["duration"] == 1.2
    assert meta["language"] == "en"
    assert meta["segments_len"] == 2
    assert "Authorization" not in meta


def test_public_meta_drops_a_key_named_like_a_secret() -> None:
    body = public_meta({"api_key": "sk-secret", "language": "en"})
    assert "api_key" not in body
    assert body["language"] == "en"
