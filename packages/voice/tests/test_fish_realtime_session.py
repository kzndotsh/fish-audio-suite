"""Fish realtime speech to text on one connection kept open across turns, driven by fake sockets."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import ormsgpack
import pytest

from fish_audio_suite_voice.asr import fish_realtime
from fish_audio_suite_voice.asr.fish_realtime import (
    FishRealtimeSession,
    realtime_config,
    unsaid_lines,
)
from fish_audio_suite_voice.asr.pipeline import spoken_lines
from fish_audio_suite_voice.asr.turn_stream import (
    ContextLine,
    SttStreamError,
    TurnDropped,
    TurnEnded,
    TurnMessage,
    TurnStarted,
    TurnUpdate,
)
from fish_audio_suite_voice.config.settings import load_config
from fish_audio_suite_voice.config.tune import STT_TERM_CHARS, STT_TERMS_MAX, SttTune
from fish_audio_suite_voice.session.conversation import Conversation

USER = ContextLine("user", "hi")
BOT = ContextLine("assistant", "hello")


def _pack(kind: str, **fields: Any) -> bytes:
    return ormsgpack.packb({"type": kind, "seq": 1, **fields})


class _LiveSocket:
    """A connection whose frames arrive when the test feeds them."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._incoming: asyncio.Queue[bytes | None] = asyncio.Queue()

    def feed(self, kind: str, **fields: Any) -> None:
        self._incoming.put_nowait(_pack(kind, **fields))

    def hang_up(self) -> None:
        self._incoming.put_nowait(None)

    async def send(self, data: bytes) -> None:
        self.sent.append(ormsgpack.unpackb(data))

    def kinds(self) -> list[str]:
        return [message["type"] for message in self.sent]

    def __aiter__(self) -> AsyncIterator[bytes]:
        async def generate() -> AsyncIterator[bytes]:
            while (frame := await self._incoming.get()) is not None:
                yield frame

        return generate()

    async def close(self) -> None:
        self.closed = True
        self._incoming.put_nowait(None)


class _Fish:
    """Hands out a new fake socket for each connection, and keeps them all."""

    def __init__(self) -> None:
        self.sockets: list[_LiveSocket] = []

    async def connect(self, url: str, *, key: str) -> _LiveSocket:
        assert (url, key) == ("wss://example/v1/asr/realtime", "secret")
        socket = _LiveSocket()
        self.sockets.append(socket)
        return socket

    @property
    def last(self) -> _LiveSocket:
        return self.sockets[-1]


def _session(
    fish: _Fish,
    context: Callable[[], list[ContextLine]],
    *,
    idle_s: int = 120,
    terms: tuple[str, ...] = (),
    clock: Callable[[], float] = lambda: 0.0,
) -> FishRealtimeSession:
    return FishRealtimeSession(
        "wss://example/v1/asr/realtime",
        "secret",
        context=context,
        sample_rate=16000,
        terms=terms,
        idle_s=idle_s,
        connect=fish.connect,
        clock=clock,
    )


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _committed_turn(
    session: FishRealtimeSession, fish: _Fish, index: int, text: str
) -> list[TurnMessage]:
    """Open a turn, send audio, have Fish commit it, and close it."""
    turn = session.turn()
    await turn.open()
    await turn.send_audio(b"\x01\x00" * 160)
    fish.last.feed("speech_start", turn_index=index)
    fish.last.feed("draft", turn_index=index, text=text[:2])
    fish.last.feed("committed", turn_index=index, text=text, reason="eot")
    heard = [message async for message in turn.messages()]
    await turn.close()
    return heard


def test_a_turn_reuses_the_connection_and_fish_is_told_only_what_it_lacks() -> None:
    fish = _Fish()
    chat: list[ContextLine] = [ContextLine("assistant", "Hi, I'm here.")]

    async def run() -> list[TurnMessage]:
        session = _session(fish, lambda: list(chat), terms=("Bart",))
        heard = await _committed_turn(session, fish, 0, "hi there")
        chat.extend([ContextLine("user", "hi there"), ContextLine("assistant", "Hello!")])
        await _committed_turn(session, fish, 1, "how are you")
        await session.close()
        return heard

    heard = asyncio.run(run())
    assert heard == [TurnStarted(""), TurnUpdate("hi"), TurnEnded("hi there", 1.0, "model")]
    assert len(fish.sockets) == 1  # one connection for both turns
    sent = fish.last.sent
    assert sent[0] == realtime_config(
        sample_rate=16000, history=[ContextLine("assistant", "Hi, I'm here.")], terms=("Bart",)
    )
    # Fish added "hi there" itself when it committed it, so only the reply is new to it.
    assert {"type": "context_add", "items": [{"role": "assistant", "text": "Hello!"}]} in sent
    assert fish.last.kinds()[-1] == "audio_done"
    assert fish.last.closed


def test_a_turn_fish_did_not_commit_closes_the_connection_and_the_next_opens_a_new_one() -> None:
    fish = _Fish()

    async def run() -> None:
        session = _session(fish, lambda: [USER, BOT])
        turn = session.turn()
        await turn.open()
        fish.last.feed("speech_start", turn_index=0)
        fish.last.feed("draft", turn_index=0, text="wai")
        await _settle()
        await turn.close()  # stopped before Fish committed it, as when a line is typed
        assert fish.sockets[0].closed
        await _committed_turn(session, fish, 0, "wait")
        await session.close()

    asyncio.run(run())
    assert len(fish.sockets) == 2
    assert fish.sockets[1].sent[0]["context"]["history"] == [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "hello"},
    ]


def test_an_edited_conversation_is_told_again_on_a_new_connection() -> None:
    fish = _Fish()
    chat = [USER, BOT]

    async def run() -> None:
        session = _session(fish, lambda: list(chat))
        await _committed_turn(session, fish, 0, "and then")
        chat[1] = ContextLine("assistant", "hello, edited")
        await _committed_turn(session, fish, 0, "again")
        await session.close()

    asyncio.run(run())
    assert len(fish.sockets) == 2
    assert fish.sockets[1].sent[0]["context"]["history"][1]["text"] == "hello, edited"


def test_what_fish_says_about_an_earlier_turn_never_reaches_the_next_one() -> None:
    fish = _Fish()

    async def run() -> list[TurnMessage]:
        session = _session(fish, list)
        await _committed_turn(session, fish, 0, "first")
        turn = session.turn()
        await turn.open()
        fish.last.feed("draft", turn_index=0, text="late words about the first")
        fish.last.feed("speech_start", turn_index=1)
        fish.last.feed("committed", turn_index=1, text="second", reason="eot")
        heard = [message async for message in turn.messages()]
        await turn.close()
        await session.close()
        return heard

    assert asyncio.run(run()) == [TurnStarted(""), TurnEnded("second", 1.0, "model")]
    assert len(fish.sockets) == 1


def test_a_turn_fish_opened_after_the_last_one_ended_gets_a_new_connection() -> None:
    fish = _Fish()

    async def run() -> None:
        session = _session(fish, list)
        await _committed_turn(session, fish, 0, "first")
        fish.last.feed("speech_start", turn_index=1)  # from audio sent after the commit
        await _settle()
        await _committed_turn(session, fish, 0, "second")
        await session.close()

    asyncio.run(run())
    assert len(fish.sockets) == 2


def test_no_audio_goes_up_once_fish_committed_the_turn() -> None:
    fish = _Fish()

    async def run() -> None:
        session = _session(fish, list)
        turn = session.turn()
        await turn.open()
        fish.last.feed("speech_start", turn_index=0)
        fish.last.feed("committed", turn_index=0, text="done", reason="eot")
        _ = [message async for message in turn.messages()]
        await turn.send_audio(b"\x01\x00" * 160)
        await turn.close()
        await session.close()

    asyncio.run(run())
    assert "audio" not in fish.last.kinds()


def test_ending_a_turn_fish_never_heard_start_drops_it_at_once_and_one_it_heard_is_finalized() -> (
    None
):
    fish = _Fish()

    async def run() -> tuple[list[TurnMessage], list[TurnMessage]]:
        session = _session(fish, list)
        quiet = session.turn()
        await quiet.open()
        await quiet.force_end_turn()
        dropped = [message async for message in quiet.messages()]
        await quiet.close()
        assert "finalize" not in fish.last.kinds()
        spoken = session.turn()
        await spoken.open()
        fish.last.feed("speech_start", turn_index=0)
        await _settle()
        await spoken.force_end_turn()
        fish.last.feed("committed", turn_index=0, text="ok", reason="finalize")
        heard = [message async for message in spoken.messages()]
        await spoken.close()
        await session.close()
        return dropped, heard

    dropped, heard = asyncio.run(run())
    assert dropped == [TurnDropped()]
    assert heard == [TurnStarted(""), TurnEnded("ok", 1.0, "manual")]
    # The turn dropped here closed its connection: Fish may still open a turn for it.
    assert len(fish.sockets) == 2
    assert "finalize" in fish.last.kinds()


def test_fish_closing_the_connection_ends_the_turn_and_the_next_one_reconnects() -> None:
    fish = _Fish()

    async def run() -> list[TurnMessage]:
        session = _session(fish, list)
        turn = session.turn()
        await turn.open()
        fish.last.feed("speech_start", turn_index=0)
        fish.last.hang_up()
        heard = [message async for message in turn.messages()]
        await turn.close()
        await _committed_turn(session, fish, 0, "back")
        await session.close()
        return heard

    assert asyncio.run(run()) == [TurnStarted("")]
    assert len(fish.sockets) == 2


def test_an_idle_connection_sends_keepalives_then_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fish_realtime, "_KEEPALIVE_S", 0.001)
    fish = _Fish()
    now = [0.0]

    async def run() -> None:
        session = _session(fish, list, idle_s=60, clock=lambda: now[0])
        await _committed_turn(session, fish, 0, "hi")
        now[0] = 30.0
        await asyncio.sleep(0.02)
        assert "keepalive" in fish.last.kinds()
        assert not fish.last.closed
        now[0] = 61.0
        await asyncio.sleep(0.02)
        assert fish.last.closed
        await _committed_turn(session, fish, 0, "back")  # the next turn opens a new one
        await session.close()

    asyncio.run(run())
    assert len(fish.sockets) == 2


def test_an_idle_limit_of_zero_closes_the_connection_after_every_turn() -> None:
    fish = _Fish()

    async def run() -> None:
        session = _session(fish, list, idle_s=0)
        await _committed_turn(session, fish, 0, "one")
        assert fish.last.closed
        await _committed_turn(session, fish, 0, "two")
        await session.close()

    asyncio.run(run())
    assert len(fish.sockets) == 2


def test_a_session_near_fishs_hour_limit_is_replaced_before_a_turn() -> None:
    fish = _Fish()
    now = [0.0]

    async def run() -> None:
        session = _session(fish, list, idle_s=3000, clock=lambda: now[0])
        await _committed_turn(session, fish, 0, "one")
        now[0] = 3400.0
        await _committed_turn(session, fish, 0, "two")
        await session.close()

    asyncio.run(run())
    assert len(fish.sockets) == 2


def test_closing_a_session_that_never_opened_does_nothing() -> None:
    fish = _Fish()

    async def run() -> None:
        session = _session(fish, list)
        await session.close()
        await session.close()

    asyncio.run(run())
    assert fish.sockets == []


def test_the_settings_carry_the_last_32_lines_each_cut_to_2000_characters() -> None:
    lines = [ContextLine("user", f"line {n}") for n in range(40)]
    lines.append(ContextLine("assistant", "x" * 2100 + "end"))
    config: dict[str, Any] = realtime_config(sample_rate=16000, history=lines, terms=("Bart",))
    history: list[dict[str, str]] = config["context"]["history"]
    assert len(history) == 32
    assert history[0] == {"role": "user", "text": "line 9"}
    assert len(history[-1]["text"]) == 2000
    assert history[-1]["text"].endswith("end")
    assert config["context"]["terms"] == ["Bart"]


def test_unsaid_lines_skips_what_fish_heard_and_spots_any_change_to_what_it_was_told() -> None:
    mm = ContextLine("user", "mm-hmm")
    # Fish heard "hi" itself, and an "mm-hmm" the chat skipped, which it keeps.
    assert unsaid_lines([], [USER, mm], [USER, BOT]) == ([BOT], [mm])
    assert unsaid_lines([], [], [USER, BOT]) == ([USER, BOT], [])
    assert unsaid_lines([USER, BOT], [], [USER, BOT]) == ([], [])
    assert unsaid_lines([USER, BOT], [], [USER]) is None  # the reply was deleted
    assert unsaid_lines([BOT], [], [ContextLine("assistant", "edited")]) is None
    assert unsaid_lines([USER], [], [ContextLine("user", "edited")]) is None  # a user line too
    assert unsaid_lines([USER], [], []) is None  # a new chat


def test_an_edited_user_line_fish_heard_itself_is_told_again_on_a_new_connection() -> None:
    fish = _Fish()
    chat: list[ContextLine] = []

    async def run() -> None:
        session = _session(fish, lambda: list(chat))
        await _committed_turn(session, fish, 0, "hello there")
        chat.extend([ContextLine("user", "hello there"), BOT])
        await _committed_turn(session, fish, 1, "next")  # told only the reply: one connection
        chat[0] = ContextLine("user", "hello, edited")
        await _committed_turn(session, fish, 0, "again")
        await session.close()

    asyncio.run(run())
    assert len(fish.sockets) == 2
    assert fish.sockets[1].sent[0]["context"]["history"][0]["text"] == "hello, edited"


def test_no_audio_goes_up_once_fish_committed_the_turn_even_before_it_is_read() -> None:
    fish = _Fish()

    async def run() -> None:
        session = _session(fish, list)
        turn = session.turn()
        await turn.open()
        fish.last.feed("speech_start", turn_index=0)
        fish.last.feed("committed", turn_index=0, text="done", reason="eot")
        await _settle()  # the reader has it; the turn has not read it yet
        await turn.send_audio(b"\x01\x00" * 160)
        await turn.close()
        await session.close()

    asyncio.run(run())
    assert "audio" not in fish.last.kinds()


def test_a_judge_whose_connection_closed_lets_the_gate_stop_the_reply() -> None:
    fish = _Fish()

    async def run() -> str | None:
        session = _session(fish, list)
        judge = session.judge()
        await judge.start()
        fish.last.hang_up()
        await _settle()
        verdict = judge.take(1.0)
        await judge.finish()
        await session.close()
        return verdict

    assert asyncio.run(run()) == "turn"


def test_ending_a_reply_never_waits_long_on_a_stuck_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fish_realtime, "_END_REPLY_WAIT_S", 0.05)
    fish = _Fish()

    async def run() -> None:
        session = _session(fish, list)
        judge = session.judge()
        await judge.start()
        stuck = asyncio.Event()

        async def hang(data: bytes) -> None:
            del data
            await stuck.wait()

        judge.hear(b"\x00\x00")
        fish.last.send = hang
        judge.hear(b"\x00\x00")
        await _settle()
        await asyncio.wait_for(judge.finish(), 1.0)  # the pump and the goodbye are both stuck
        assert fish.last.closed
        await session.close()

    asyncio.run(run())


def test_spoken_lines_are_the_visible_turns_without_cues_one_line_each() -> None:
    chat = Conversation("system")
    chat.add_user_line("first")
    chat.add_user_line("second")  # never answered, so it joins the first
    chat.add_reply("[happy] Hello <whisper>there</whisper>!", interrupted=False)
    chat.add_user_line("hide me")
    chat.add_reply("hidden", interrupted=False)
    chat.set_hidden(chat.turns[-1].id, hidden=True)
    lines = spoken_lines(chat)
    assert [(line.role, line.text) for line in lines] == [
        ("user", "first"),
        ("user", "second"),
        ("assistant", "Hello there!"),
    ]


def test_stt_terms_are_read_from_a_comma_list_dropping_any_fish_would_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FISH_VOICE_STT_TERMS", f" Bart, ,Lisa,{'x' * (STT_TERM_CHARS + 1)}")
    monkeypatch.setenv("FISH_VOICE_STT_IDLE", "45")
    tune = SttTune.from_env()
    assert tune.terms == ("Bart", "Lisa")
    assert tune.idle_s == 45


@pytest.mark.parametrize(
    "kwargs",
    [
        {"terms": ("x" * (STT_TERM_CHARS + 1),)},
        {"terms": (" ",)},
        {"terms": tuple(f"t{n}" for n in range(STT_TERMS_MAX + 1))},
        {"idle_s": -1},
    ],
)
def test_stt_tune_refuses_terms_or_an_idle_limit_fish_or_from_env_would_not_take(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="SttTune"):
        SttTune(**kwargs)


def test_a_reply_judge_tells_fish_the_reply_plays_sends_the_mic_and_reads_its_verdict() -> None:
    fish = _Fish()

    async def run() -> tuple[str | None, str | None]:
        session = _session(fish, lambda: [USER])
        judge = session.judge()
        await judge.start()
        frame = b"\x02\x00" * 480
        judge.hear(frame)
        await _settle()
        fish.last.feed("speech_start", turn_index=0)
        fish.last.feed("user_backchannel", turn_index=0, audio_ms=300)
        await _settle()
        first = await asyncio.to_thread(judge.take, 1.0)  # the barge-in thread asks
        again = judge.take(1.0)  # taken once
        await judge.finish()
        await session.close()
        return first, again

    first, again = asyncio.run(run())
    assert (first, again) == ("backchannel", None)
    kinds = fish.last.kinds()
    assert kinds[:2] == ["config", "agent_speaking"]
    assert fish.last.sent[1] == {"type": "agent_speaking", "speaking": True}
    assert {"type": "audio", "data": b"\x02\x00" * 480} in fish.last.sent
    assert {"type": "agent_speaking", "speaking": False} in fish.last.sent


def test_taking_the_turn_outweighs_an_acknowledgement_and_old_verdicts_do_not_count() -> None:
    fish = _Fish()
    now = [0.0]

    async def run() -> tuple[str | None, str | None]:
        session = _session(fish, list, clock=lambda: now[0])
        judge = session.judge()
        await judge.start()
        fish.last.feed("user_turn_taken", turn_index=0)
        fish.last.feed("user_backchannel", turn_index=0)  # later, but the turn still counts
        await _settle()
        both = judge.take(1.0)
        fish.last.feed("user_turn_taken", turn_index=1)
        await _settle()
        now[0] = 5.0
        stale = judge.take(1.0)
        await judge.finish()
        await session.close()
        return both, stale

    assert asyncio.run(run()) == ("turn", None)


def test_a_judge_that_cannot_reach_fish_leaves_the_gate_to_stop_the_reply() -> None:
    async def refuse(url: str, *, key: str) -> _LiveSocket:
        del url, key
        raise SttStreamError("Fish realtime speech to text refused the connection (HTTP 503)")

    async def run() -> str | None:
        session = FishRealtimeSession(
            "wss://x", "k", context=list, sample_rate=16000, connect=refuse
        )
        judge = session.judge()
        await judge.start()
        judge.hear(b"\x00\x00")  # dropped: nothing to send it on
        verdict = judge.take(1.0)
        await judge.finish()
        await session.close()
        return verdict

    assert asyncio.run(run()) == "turn"


def test_fish_messages_about_speech_over_a_reply_are_ignored_outside_one() -> None:
    fish = _Fish()

    async def run() -> str | None:
        session = _session(fish, list)
        await _committed_turn(session, fish, 0, "hi")
        fish.last.feed("user_turn_taken", turn_index=1)  # no reply playing
        await _settle()
        verdict = session.take_verdict(10.0)
        await session.close()
        return verdict

    assert asyncio.run(run()) is None


def test_the_fish_judge_needs_fish_realtime_and_otherwise_falls_back_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FISH_VOICE_BARGE_JUDGE", "fish")
    monkeypatch.setenv("FISH_VOICE_STT", "deepgram")
    assert load_config().barge.judge == "local"
    monkeypatch.setenv("FISH_VOICE_STT", "fish-realtime")
    assert load_config().barge.judge == "fish"


def test_a_failure_sending_the_mic_leaves_the_gate_to_decide_and_ends_the_reply_quietly() -> None:
    fish = _Fish()

    async def run() -> str | None:
        session = _session(fish, list)
        judge = session.judge()
        await judge.start()

        async def broken(data: bytes) -> None:
            del data
            raise RuntimeError("socket in a state nobody expected")

        fish.last.send = broken
        judge.hear(b"\x00\x00")
        await _settle()
        verdict = judge.take(1.0)
        fish.last.send = _LiveSocket.send.__get__(fish.last)
        await judge.finish()  # does not raise
        await session.close()
        return verdict

    assert asyncio.run(run()) == "turn"
