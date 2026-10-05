"""A real recording of a quiet mic: RMS per 90 ms block, taken with ``scripts/mic_levels.py``.

The first two seconds are the room, then a person says a few sentences with pauses. The room
sits near -61 dB, speech at -35 to -45 dB and the loudest block at -28 dB, so this mic is
much quieter than the full-scale audio of a TTS voice.
"""

from __future__ import annotations

BLOCK_S = 0.09

# fmt: off
RMS: tuple[float, ...] = (
    487, 235, 437, 133, 357, 281, 153, 184, 146, 96, 102, 102, 133, 25, 56, 34, 35, 34, 31, 27,
    39, 237, 1258, 524, 221, 44, 203, 447, 580, 332, 440, 544, 576, 364, 23, 28, 31, 26, 30, 29,
    30, 37, 32, 26, 48, 29, 34, 26, 29, 28, 18, 22, 22, 27, 35, 29, 39, 40, 30, 47, 72, 26, 82,
    317, 598, 361, 64, 82, 244, 222, 186, 254, 286, 307, 147, 26, 24, 31, 20, 356, 75, 396, 342,
    101, 77, 28, 41, 61, 26, 41, 77, 100, 122, 457, 286, 188, 151, 328, 346, 250, 289, 356, 203,
    30, 76, 83, 46, 81, 137, 40, 88, 40, 31, 22, 85, 52, 73, 42, 67, 74, 22, 44, 34, 63, 93, 74,
    620, 241, 78,
)
# fmt: on

SPEECH_FROM = 200.0  # a block above this is a person talking
QUIET_BELOW = 60.0  # a block below this is the room
