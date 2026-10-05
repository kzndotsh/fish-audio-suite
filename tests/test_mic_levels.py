"""The numbers `scripts/mic_levels.py` reports. The recording itself needs a mic and is not run."""

from __future__ import annotations

import pytest
from mic_levels import main, percentile, summarize, to_db


def test_a_level_is_reported_in_db_below_full_scale() -> None:
    assert to_db(32768.0) == pytest.approx(0.0)
    assert to_db(3276.8) == pytest.approx(-20.0)
    assert to_db(0.0) == -90.0
    assert to_db(-5.0) == -90.0  # nonsense is silence


def test_a_percentile_is_one_of_the_values_and_ends_at_the_extremes() -> None:
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 1.0) == 5.0
    assert percentile([7.0], 0.9) == 7.0
    assert percentile(values, 5.0) == 5.0  # out of range is clamped
    assert percentile(values, -1.0) == 1.0


def test_a_summary_separates_the_room_from_the_speech_and_finds_the_bump() -> None:
    room = [-55.0, -56.0, -54.0, -55.0, -57.0, -55.0, -56.0, -54.0, -55.0, -56.0]
    speech = [-30.0, -26.0, -28.0, -24.0, -29.0]
    bump = [-8.0]
    stats = summarize(room + speech + bump)
    assert -58.0 <= stats["noise"] <= -54.0
    assert stats["speech"] == pytest.approx(
        sum(speech + bump) / 6
    )  # everything well above the room
    assert stats["peak"] == -8.0
    assert stats["spread"] == pytest.approx(-8.0 - stats["speech"])
    assert stats["active"] == pytest.approx(6 / 16)
    assert stats["p95"] >= stats["speech"] - 10.0


def test_a_recording_of_only_room_has_no_speech_to_average() -> None:
    stats = summarize([-55.0, -56.0, -54.0, -55.0])
    assert stats["active"] == 0.0
    assert stats["speech"] == stats["noise"]  # nothing above the room, so nothing to average


def test_the_script_explains_itself_without_opening_the_mic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--seconds" in capsys.readouterr().out
