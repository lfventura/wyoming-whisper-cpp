import argparse
import asyncio
import logging
import struct
import math
from collections import deque
from typing import Optional

import numpy as np

from asyncio.subprocess import Process

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

END_OF_TEXT = "<|endoftext|>"

# Longest wait for one utterance's transcript before giving up on it
TRANSCRIPT_TIMEOUT = 60


class WhisperCppProcess:
    """The single simple-server process shared by every client connection.

    simple-server answers each NaN end marker with one transcript followed by
    <|endoftext|>, in order. One reader matches those answers to the utterances
    in the order their markers were written, so a client that disconnects early
    can never leave its answer behind for the next client to read.
    """

    def __init__(self, proc: Process) -> None:
        self.proc = proc
        # Held from a client's first audio chunk until its transcript arrives
        self.lock = asyncio.Lock()
        self._pending: deque[asyncio.Future] = deque()
        self._reader_task = asyncio.create_task(self._read_transcripts())

    async def write_audio(self, float32_audio: bytes) -> None:
        self.proc.stdin.write(float32_audio)
        await self.proc.stdin.drain()

    async def end_utterance(self) -> asyncio.Future:
        """Write the end marker and return a future for the utterance's transcript."""
        future = asyncio.get_running_loop().create_future()
        self._pending.append(future)
        self.proc.stdin.write(struct.pack("f", math.nan))
        await self.proc.stdin.drain()
        _LOGGER.debug("Wrote NaN marker to whisper process stdin")
        return future

    async def _read_transcripts(self) -> None:
        text_lines = []
        while True:
            line = await self.proc.stdout.readline()
            _LOGGER.debug("Read line: %s", line)
            if not line:
                _LOGGER.error("whisper process closed its output")
                break
            decoded = line.decode(errors="replace").strip()
            if decoded != END_OF_TEXT:
                if decoded:
                    text_lines.append(decoded)
                continue
            text = " ".join(text_lines).strip()
            text_lines = []
            if not self._pending:
                _LOGGER.warning("Dropped transcript with no utterance waiting for it: %r", text)
                continue
            future = self._pending.popleft()
            if not future.done():
                future.set_result(text)
        for future in self._pending:
            if not future.done():
                future.set_exception(ConnectionError("whisper process closed its output"))
        self._pending.clear()


class WhisperCppEventHandler(AsyncEventHandler):
    """Event handler that streams audio to whisper-stream for low-latency transcription."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model_args: list[str],
        whisper: WhisperCppProcess,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model_args = model_args
        self.whisper = whisper
        self.audio_converter = AudioChunkConverter(
            rate=16000,
            width=2,
            channels=1,
        )
        self._language = self.cli_args.language
        # True while this client holds the whisper process with audio written
        self._streaming = False

    async def _start_utterance(self) -> None:
        if not self._streaming:
            await self.whisper.lock.acquire()
            self._streaming = True

    async def _finish_utterance(self) -> Optional[str]:
        """End the current utterance and wait for its transcript, then free the process."""
        if not self._streaming:
            return None
        try:
            future = await self.whisper.end_utterance()
            return await asyncio.wait_for(asyncio.shield(future), TRANSCRIPT_TIMEOUT)
        finally:
            self._streaming = False
            self.whisper.lock.release()

    async def handle_event(self, event: Event) -> bool:
        if AudioStart.is_type(event.type):
            await self._start_utterance()
            return True

        if AudioChunk.is_type(event.type):
            await self._start_utterance()
            chunk = AudioChunk.from_event(event)
            chunk = self.audio_converter.convert(chunk)

            # Convert audio chunk to float32 PCM format
            audio_data = np.frombuffer(chunk.audio, dtype=np.int16).astype(np.float32) / 32768.0
            await self.whisper.write_audio(audio_data.tobytes())
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")
            try:
                text = await self._finish_utterance()
            except (asyncio.TimeoutError, ConnectionError):
                _LOGGER.exception("No transcript from whisper process")
                text = None
            # Always answer, even with no speech, or the client waits until it times out
            await self.write_event(Transcript(text=text or "").event())
            _LOGGER.debug("Completed streaming transcription: %r", text)
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True

    async def disconnect(self) -> None:
        if self._streaming:
            # Client left mid-utterance: close its audio off so it is not glued to the next client's
            _LOGGER.debug("Client disconnected mid-utterance, discarding its audio")
            try:
                await self._finish_utterance()
            except (asyncio.TimeoutError, ConnectionError):
                _LOGGER.exception("No transcript from whisper process")
