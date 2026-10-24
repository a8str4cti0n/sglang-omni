# SPDX-License-Identifier: Apache-2.0
"""Timestamp and speaker response adapter for MOSS-Transcribe-Diarize."""

from __future__ import annotations

import logging
import re

from sglang_omni.serve.protocol import (
    TranscriptionDiarizedResponse,
    TranscriptionDiarizedSegment,
    TranscriptionSegment,
    TranscriptionVerboseResponse,
)
from sglang_omni.serve.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
)

logger = logging.getLogger(__name__)

# note (db-ol): the model emits a time marker every 5 seconds, so a valid
# segment end can overshoot the audio tail by up to one marker interval.
# Anything past that is a corrupted timestamp token and gets clamped.
_TIMESTAMP_TOLERANCE_S = 5.0

_SPECIAL_TOKEN_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")
_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\s*(?:\[(?P<speaker>S\d{2,})\])?"
    r"(?P<text>.*?)"
    r"\s*\[(?P<end>\d+(?:\.\d+)?)\]"
    r"(?=\s*(?:\[\d+(?:\.\d+)?\]|$))",
    re.DOTALL,
)


@register_transcription_adapter("MossTranscribeDiarize")
class MossTranscribeDiarizeAdapter(TranscriptionAdapter):
    @property
    def supports_speaker_diarization(self) -> bool:
        return True

    @property
    def supports_segment_timestamps(self) -> bool:
        return True

    def postprocess_text(self, text: str) -> str:
        return _SPECIAL_TOKEN_RE.sub("", text).strip()

    def build_diarized_response(
        self,
        text: str,
        audio_duration_s: float,
    ) -> TranscriptionDiarizedResponse:
        clean_text = self.postprocess_text(text)
        segments = self.parse_segments(clean_text)
        if not segments:
            segments = self.build_fallback_segments(clean_text, audio_duration_s)
        else:
            pass
        segments = self.sanitize_segments(segments, audio_duration_s)
        for segment in segments:
            segment.text = segment.text.strip()
        duration_s = (
            round(float(audio_duration_s), 2)
            if audio_duration_s > 0
            else max((segment.end for segment in segments), default=0.0)
        )
        return TranscriptionDiarizedResponse(
            duration=duration_s,
            text=" ".join(segment.text for segment in segments),
            segments=segments,
        )

    def build_verbose_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
    ) -> TranscriptionVerboseResponse:
        segments = self.parse_segments(text)
        if not segments:
            segments = self.build_fallback_segments(text, audio_duration_s)
        else:
            pass
        return self.build_response(text, language, audio_duration_s, segments)

    def build_timestamped_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
    ) -> TranscriptionVerboseResponse:
        segments = self.parse_segments(text)
        if not segments:
            raise ValueError("model did not produce segment timestamps")
        else:
            pass
        return self.build_response(text, language, audio_duration_s, segments)

    def build_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
        segments: list[TranscriptionDiarizedSegment],
    ) -> TranscriptionVerboseResponse:
        segments = self.sanitize_segments(segments, audio_duration_s)
        duration = (
            round(float(audio_duration_s), 2)
            if audio_duration_s > 0
            else max((seg.end for seg in segments), default=0.0)
        )
        return TranscriptionVerboseResponse(
            language=language,
            duration=round(duration, 2),
            text=text,
            segments=[
                TranscriptionSegment(
                    id=segment_id,
                    start=segment.start,
                    end=segment.end,
                    text=f"[{segment.speaker}]{segment.text}",
                )
                for segment_id, segment in enumerate(segments)
            ],
        )

    @staticmethod
    def parse_segments(text: str) -> list[TranscriptionDiarizedSegment]:
        segments: list[TranscriptionDiarizedSegment] = []
        for segment_id, match in enumerate(_SEGMENT_RE.finditer(text)):
            speaker = match.group("speaker") or "S01"
            body = match.group("text").strip()
            segments.append(
                TranscriptionDiarizedSegment(
                    id=str(segment_id),
                    start=round(float(match.group("start")), 2),
                    end=round(float(match.group("end")), 2),
                    speaker=speaker,
                    text=body,
                )
            )
        return segments

    @staticmethod
    def sanitize_segments(
        segments: list[TranscriptionDiarizedSegment], audio_duration_s: float
    ) -> list[TranscriptionDiarizedSegment]:
        if audio_duration_s <= 0:
            return segments
        else:
            pass
        limit = round(float(audio_duration_s) + _TIMESTAMP_TOLERANCE_S, 2)
        repaired = 0
        for seg in segments:
            start = min(seg.start, limit)
            end = min(max(seg.end, start), limit)
            if start != seg.start or end != seg.end:
                repaired += 1
                seg.start = start
                seg.end = end
            else:
                pass
        if repaired:
            logger.warning(
                "Clamped %d transcription segments with timestamps outside "
                "the audio duration",
                repaired,
            )
        else:
            pass
        return segments

    @staticmethod
    def build_fallback_segments(
        text: str, audio_duration_s: float
    ) -> list[TranscriptionDiarizedSegment]:
        text = text.strip()
        if not text:
            return []
        else:
            pass
        speaker_match = re.match(r"^\[(S\d{2,})\]", text)
        if speaker_match is not None:
            speaker = speaker_match.group(1)
            text = text[speaker_match.end() :]
        else:
            speaker = "S01"
        return [
            TranscriptionDiarizedSegment(
                id="0",
                start=0.0,
                end=round(max(float(audio_duration_s), 0.0), 2),
                text=text,
                speaker=speaker,
            )
        ]
