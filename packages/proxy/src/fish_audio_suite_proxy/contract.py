"""OpenAI audio fields translated into Fish TTS and ASR bodies."""

from fish_audio_suite_proxy.errors import json_error, json_from_upstream, openai_error_body
from fish_audio_suite_proxy.fields import (
    SILENT_MP3,
    catalog_ids,
    chunk_length_hi,
    pcm_sample_rate,
    pick_format,
    pick_reference_id,
    prepare_tts_text,
    resolve_asr_model,
    resolve_tts_model,
    runtime_defaults,
    upstream_trace_headers,
)
from fish_audio_suite_proxy.speech import pack_tts, speech_controls
from fish_audio_suite_proxy.transcribe import (
    asr_upload,
    caption_cues,
    form_strings,
    read_asr,
    transcription_body,
)

__all__ = [
    "SILENT_MP3",
    "asr_upload",
    "caption_cues",
    "catalog_ids",
    "chunk_length_hi",
    "form_strings",
    "json_error",
    "json_from_upstream",
    "openai_error_body",
    "pack_tts",
    "pcm_sample_rate",
    "pick_format",
    "pick_reference_id",
    "prepare_tts_text",
    "read_asr",
    "resolve_asr_model",
    "resolve_tts_model",
    "runtime_defaults",
    "speech_controls",
    "transcription_body",
    "upstream_trace_headers",
]
