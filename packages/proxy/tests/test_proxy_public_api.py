"""The proxy's public surface: what each module exports and what the types promise."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import typing
from collections.abc import Iterator
from types import ModuleType

import httpx
import pytest
from starlette.requests import Request

from fish_audio_suite_kit import AsrFormat, FishErrorBody
from fish_audio_suite_proxy.audio import (
    SUPPORTED_FORMATS,
    AudioDecodeError,
    ClientFormat,
    decode_audio_b64,
    fish_audio_format,
)
from fish_audio_suite_proxy.errors import (
    ProxyError,
    provider_json_error,
    proxy_error_response,
)
from fish_audio_suite_proxy.request_fields import read_format
from fish_audio_suite_proxy.settings import ProxySettings
from fish_audio_suite_proxy.speech import (
    ClipError,
    PackedTts,
    SpeechControls,
)
from fish_audio_suite_proxy.transcribe import (
    ASR_FORMATS,
    InboundAsr,
    read_asr_format,
)
from fish_audio_suite_proxy.upstream import FishHttp, RetryPolicy

_MODULES = (
    "audio",
    "body_limit",
    "errors",
    "models",
    "phrases",
    "request_fields",
    "server",
    "settings",
    "speech",
    "transcribe",
    "upstream",
)


def _module(name: str) -> ModuleType:
    return importlib.import_module(f"fish_audio_suite_proxy.{name}")


def _exports(module: ModuleType) -> list[str]:
    declared: object = vars(module).get("__all__")
    assert isinstance(declared, list), f"{module.__name__} has no __all__"
    # By convention a module's __all__ is a list of names.
    return typing.cast(list[str], declared)


@pytest.mark.parametrize("name", _MODULES)
def test_every_module_lists_its_public_names_and_each_one_exists(name: str) -> None:
    module = _module(name)
    for item in _exports(module):
        assert hasattr(module, item), f"{name}.__all__ names {item}, which does not exist"


def _type_names(annotation: object) -> Iterator[str]:
    if isinstance(annotation, type):
        yield annotation.__name__
    for argument in typing.get_args(annotation):
        yield from _type_names(argument)


def _public_callables_and_classes() -> Iterator[tuple[str, object]]:
    for name in _MODULES:
        module = _module(name)
        for item in _exports(module):
            value = getattr(module, item)
            if callable(value) and not item.isupper():
                yield f"{name}.{item}", value


def _hints(value: object) -> dict[str, object]:
    """Resolved annotations of a function, or of a dataclass's fields."""
    if isinstance(value, type) and not dataclasses.is_dataclass(value):
        return {}
    try:
        return dict(typing.get_type_hints(value))
    except TypeError:
        return {}


@pytest.mark.parametrize(("label", "value"), list(_public_callables_and_classes()))
def test_no_public_signature_mentions_a_private_type(label: str, value: object) -> None:
    hints = _hints(value)
    for key, annotation in hints.items():
        for type_name in _type_names(annotation):
            assert not type_name.startswith("_"), f"{label} {key} uses private type {type_name}"


def test_clip_error_is_a_proxy_error_with_status_400() -> None:
    error = ClipError("reference audio is empty")
    assert isinstance(error, AudioDecodeError)
    assert isinstance(error, ProxyError)
    assert error.status == 400
    assert error.message == "reference audio is empty"
    response = proxy_error_response(Request({"type": "http"}), error)
    assert response.status_code == 400
    with pytest.raises(ProxyError):
        decode_audio_b64("")


def test_the_audio_decoder_names_the_field_it_read() -> None:
    with pytest.raises(AudioDecodeError, match=r"^audio is empty$"):
        decode_audio_b64(None)
    with pytest.raises(AudioDecodeError, match=r"^audio is not valid base64$"):
        decode_audio_b64("!!!!")
    with pytest.raises(AudioDecodeError, match=r"^input_audio is empty$") as caught:
        decode_audio_b64("  ", field="input_audio")
    assert caught.value.status == 400
    assert decode_audio_b64(b"RIFF") == b"RIFF"


@pytest.mark.parametrize(
    "instance",
    [
        SpeechControls("s2.1-pro", 1.0, "mp3", "normal", 200, 50),
        PackedTts({}, {"json": {}}, "audio/mpeg"),
        InboundAsr(b"x", "a.wav", "audio/wav", None, "", "json", ()),
        RetryPolicy(),
        ProxySettings(),
    ],
)
def test_the_public_value_types_are_frozen_and_slotted(instance: object) -> None:
    assert dataclasses.is_dataclass(instance)
    assert not hasattr(instance, "__dict__")
    field_name = dataclasses.fields(instance)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field_name, None)


def test_formats_the_client_may_ask_for_and_the_format_fish_makes() -> None:
    assert set(SUPPORTED_FORMATS) == set(typing.get_args(ClientFormat))
    assert fish_audio_format("pcm16") == "pcm"
    for fmt in ("mp3", "opus", "pcm", "wav"):
        assert fish_audio_format(typing.cast(ClientFormat, fmt)) == fmt
    assert read_format({"format": "PCM16"}, "mp3") == "pcm16"


def test_asr_formats_are_the_kit_literal_and_the_reader_returns_a_member() -> None:
    assert typing.get_args(AsrFormat) == ASR_FORMATS
    for name in ASR_FORMATS:
        assert read_asr_format(f" {name.upper()} ") == name
    with pytest.raises(ProxyError, match="unsupported response_format"):
        read_asr_format("xml")


def test_a_parsed_fish_error_becomes_a_provider_error_with_its_own_status() -> None:
    response = provider_json_error(FishErrorBody.of(429, "slow down"))
    assert response.status_code == 429
    assert response.body == (
        b'{"error":{"code":429,"message":"slow down","type":"provider_error",'
        b'"metadata":{"provider_name":"fish-audio"}}}'
    )


def test_an_httpx_client_satisfies_the_fish_http_protocol() -> None:
    http = httpx.AsyncClient()
    client: FishHttp = http
    try:
        request = client.build_request("POST", "https://example.test/x", json={"a": 1})
        assert isinstance(request, httpx.Request)
    finally:
        asyncio.run(http.aclose())
