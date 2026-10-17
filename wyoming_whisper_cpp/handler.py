import argparse
import asyncio
import logging
import struct
import math
import numpy as np

from asyncio.subprocess import Process

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


class WhisperCppEventHandler(AsyncEventHandler):
    """Event handler that streams audio to whisper-stream for low-latency transcription."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model_args: list[str],
        model_proc: Process,
        model_proc_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model_args = model_args
        self.model_proc = model_proc
        self.model_proc_lock = model_proc_lock
        self.audio_converter = AudioChunkConverter(
            rate=16000,
            width=2,
            channels=1,
        )
        self._language = self.cli_args.language
        self._stream_reader_task = None

    async def _read_transcription_stream(self) -> None:
        """Reads partial transcription output from the streaming process and sends transcription events."""
        text_lines = []
        while True:
            line = await self.model_proc.stdout.readline()
            _LOGGER.debug("Read line: %s", line)
            if not line:
                break
            decoded = line.decode().strip()
            if decoded == "<|endoftext|>":
                break
            if decoded:
                text_lines.append(decoded)
                current_text = " ".join(text_lines).strip()
                await self.write_event(Transcript(text=current_text).event())
        if not text_lines:
            # No speech (e.g. VAD dropped everything): still answer, or the client waits until it times out
            await self.write_event(Transcript(text="").event())
        _LOGGER.debug("Completed streaming transcription")

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            chunk = self.audio_converter.convert(chunk)

            # Convert audio chunk to float32 PCM format
            audio_data = np.frombuffer(chunk.audio, dtype=np.int16).astype(np.float32) / 32768.0
            float32_audio = audio_data.tobytes()

            # Send the float32 audio chunk to the streaming process.
            self.model_proc.stdin.write(float32_audio)
            #_LOGGER.debug("Wrote float32 audio chunk of %d bytes to whisper process stdin", len(float32_audio))
            await self.model_proc.stdin.drain()
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")
            await self.model_proc.stdin.drain()
            self.model_proc.stdin.write(struct.pack("f", math.nan))
            _LOGGER.debug("Wrote NaN marker to whisper process stdin")
            await self.model_proc.stdin.drain()

            #if self.model_proc and self.model_proc.stdin:
                # Write a NaN marker to signal end-of-segment.
            #    self.model_proc.stdin.write(struct.pack("f", math.nan))
            #    _LOGGER.debug("Wrote NaN marker to whisper process stdin")
            # Wait for the process to finish processing the current segment.
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if self._stream_reader_task is None:
            self._stream_reader_task = asyncio.create_task(self._read_transcription_stream())

        return True