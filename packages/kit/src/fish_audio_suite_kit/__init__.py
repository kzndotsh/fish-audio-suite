"""Shared Fish Audio helpers for proxy and voice.

Cue tags, TTS and ASR scrubbing, sentence cuts, turn judging, shared defaults,
environment readers, the Fish error shape and retry policy, and W3C trace headers.
No HTTP client, no audio, no API key. Unofficial. Not affiliated with Fish Audio.
"""

from fish_audio_suite_kit._charsets import cut_at_control, is_header_safe, utf8_text
from fish_audio_suite_kit._speech_floor import DEFAULT_SHORT_WORDS
from fish_audio_suite_kit._values import clamp_number, parse_flag, parse_number
from fish_audio_suite_kit._version import read_version
from fish_audio_suite_kit.asr_text import (
    DEFAULT_BACKCHANNELS,
    DEFAULT_QUIT_PHRASES,
    asr_language_hint,
    asr_tone,
    is_asr_hallucination,
    is_backchannel,
    is_caption_watermark,
    is_quit_utterance,
    is_same_utterance,
    scrub_asr,
    without_watermark_segments,
)
from fish_audio_suite_kit.captions import CaptionCue, format_as_srt, format_as_vtt
from fish_audio_suite_kit.catalog import (
    CHUNK_LENGTH_CLOUD_HI,
    CHUNK_LENGTH_LO,
    FISH_TTS_MODEL_IDS,
    MIN_CHUNK_LENGTH_HI,
    MIN_CHUNK_LENGTH_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    UNIT_INTERVAL_HI,
    UNIT_INTERVAL_LO,
    chunk_length_hi,
    known_latency,
    known_mp3_bitrate,
    known_opus_bitrate,
    normalize_tts_model,
    takes_directions,
)
from fish_audio_suite_kit.cue_style import CueMode, CueStyle
from fish_audio_suite_kit.cues import (
    MoodCarry,
    ensure_lead_cue,
    last_emotion,
    normalize_cues,
    official_cue,
    split_cues,
    strip_cue_tags,
)
from fish_audio_suite_kit.cuts import ends_sentence, next_tts_cut, sentence_cuts, split_tts_piece
from fish_audio_suite_kit.defaults import SuiteDefaults
from fish_audio_suite_kit.dialogue import extract_quoted_speech
from fish_audio_suite_kit.env import (
    env_base,
    env_bool,
    env_float,
    env_int,
    env_text,
    env_token,
)
from fish_audio_suite_kit.http_errors import (
    FISH_ASR_PATH,
    FISH_RETRY_ATTEMPTS,
    FISH_TTS_PATH,
    FishAuthError,
    FishErrorBody,
    FishHttpError,
    FishRateLimitError,
    FishTimeoutError,
    FishUpstreamError,
    bearer,
    describe_request_error,
    fish_attempt_exhausted,
    fish_backoff_s,
    fish_request_id,
    parse_asr_body,
    parse_fish_error,
    retry_after_s,
    should_retry_fish_status,
)
from fish_audio_suite_kit.junk import is_tts_junk
from fish_audio_suite_kit.literals import (
    AsrFormat,
    AudioFormat,
    ChatRole,
    FishLatency,
    TtsModel,
)
from fish_audio_suite_kit.payloads import (
    AsrBody,
    AsrSegment,
    AsrWord,
    ChatMessage,
    OpenAIErrorBody,
    OpenAIErrorDetail,
)
from fish_audio_suite_kit.prompt import DEFAULT_SEED_EXCHANGE, DEFAULT_SYSTEM_PROMPT
from fish_audio_suite_kit.scrub_tts import scrub_tts
from fish_audio_suite_kit.stream_holds import tts_hold_at
from fish_audio_suite_kit.timing import MS_PER_S, LatencySnapshot, elapsed_ms
from fish_audio_suite_kit.trace_context import (
    ensure_trace_headers,
    make_traceparent,
    trace_id_of,
)
from fish_audio_suite_kit.turn import TurnJudgement, TurnReason, judge_turn
from fish_audio_suite_kit.urls import is_insecure_fish_base, strip_base

__version__ = read_version()

__all__ = [
    "CHUNK_LENGTH_CLOUD_HI",
    "CHUNK_LENGTH_LO",
    "DEFAULT_BACKCHANNELS",
    "DEFAULT_QUIT_PHRASES",
    "DEFAULT_SEED_EXCHANGE",
    "DEFAULT_SHORT_WORDS",
    "DEFAULT_SYSTEM_PROMPT",
    "FISH_ASR_PATH",
    "FISH_RETRY_ATTEMPTS",
    "FISH_TTS_MODEL_IDS",
    "FISH_TTS_PATH",
    "MIN_CHUNK_LENGTH_HI",
    "MIN_CHUNK_LENGTH_LO",
    "MS_PER_S",
    "TTS_SPEED_HI",
    "TTS_SPEED_LO",
    "UNIT_INTERVAL_HI",
    "UNIT_INTERVAL_LO",
    "AsrBody",
    "AsrFormat",
    "AsrSegment",
    "AsrWord",
    "AudioFormat",
    "CaptionCue",
    "ChatMessage",
    "ChatRole",
    "CueMode",
    "CueStyle",
    "FishAuthError",
    "FishErrorBody",
    "FishHttpError",
    "FishLatency",
    "FishRateLimitError",
    "FishTimeoutError",
    "FishUpstreamError",
    "LatencySnapshot",
    "MoodCarry",
    "OpenAIErrorBody",
    "OpenAIErrorDetail",
    "SuiteDefaults",
    "TtsModel",
    "TurnJudgement",
    "TurnReason",
    "__version__",
    "asr_language_hint",
    "asr_tone",
    "bearer",
    "chunk_length_hi",
    "clamp_number",
    "cut_at_control",
    "describe_request_error",
    "elapsed_ms",
    "ends_sentence",
    "ensure_lead_cue",
    "ensure_trace_headers",
    "env_base",
    "env_bool",
    "env_float",
    "env_int",
    "env_text",
    "env_token",
    "extract_quoted_speech",
    "fish_attempt_exhausted",
    "fish_backoff_s",
    "fish_request_id",
    "format_as_srt",
    "format_as_vtt",
    "is_asr_hallucination",
    "is_backchannel",
    "is_caption_watermark",
    "is_header_safe",
    "is_insecure_fish_base",
    "is_quit_utterance",
    "is_same_utterance",
    "is_tts_junk",
    "judge_turn",
    "known_latency",
    "known_mp3_bitrate",
    "known_opus_bitrate",
    "last_emotion",
    "make_traceparent",
    "next_tts_cut",
    "normalize_cues",
    "normalize_tts_model",
    "official_cue",
    "parse_asr_body",
    "parse_fish_error",
    "parse_flag",
    "parse_number",
    "retry_after_s",
    "scrub_asr",
    "scrub_tts",
    "sentence_cuts",
    "should_retry_fish_status",
    "split_cues",
    "split_tts_piece",
    "strip_base",
    "strip_cue_tags",
    "takes_directions",
    "trace_id_of",
    "tts_hold_at",
    "utf8_text",
    "without_watermark_segments",
]
