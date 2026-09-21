"""Unofficial Fish Audio text helpers. Not affiliated with Fish Audio."""

from fish_audio_suite_kit.captions import CaptionCue, format_as_srt, format_as_vtt
from fish_audio_suite_kit.cues import ensure_lead_cue, normalize_cues, spread_cues
from fish_audio_suite_kit.defaults import (
    DEFAULT_SYSTEM_PROMPT,
    LatencySnapshot,
    SuiteDefaults,
)
from fish_audio_suite_kit.http_errors import (
    FISH_RETRY_ATTEMPTS,
    FishHttpError,
    fish_backoff_seconds,
    fish_error_body,
    parse_fish_error,
    should_retry_fish_status,
)
from fish_audio_suite_kit.text_filters import (
    extract_quoted_speech,
    is_asr_hallucination,
    is_backchannel,
    is_quit_utterance,
    is_tts_junk,
    next_tts_cut,
    scrub_asr,
    scrub_tts,
    skip_empty_delta,
)
from fish_audio_suite_kit.trace_context import (
    canonical_traceparent,
    make_traceparent,
    trace_id_of,
    w3c_trace_headers,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "FISH_RETRY_ATTEMPTS",
    "CaptionCue",
    "FishHttpError",
    "LatencySnapshot",
    "SuiteDefaults",
    "canonical_traceparent",
    "ensure_lead_cue",
    "extract_quoted_speech",
    "fish_backoff_seconds",
    "fish_error_body",
    "format_as_srt",
    "format_as_vtt",
    "is_asr_hallucination",
    "is_backchannel",
    "is_quit_utterance",
    "is_tts_junk",
    "make_traceparent",
    "next_tts_cut",
    "normalize_cues",
    "parse_fish_error",
    "scrub_asr",
    "scrub_tts",
    "should_retry_fish_status",
    "skip_empty_delta",
    "spread_cues",
    "trace_id_of",
    "w3c_trace_headers",
]
