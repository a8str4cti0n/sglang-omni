# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MOSS-Transcribe-Diarize verbose_json adapter."""

from __future__ import annotations

import pytest

from sglang_omni.serve.transcription_adapters import resolve_adapter
from sglang_omni.serve.transcription_adapters.base import DefaultTranscriptionAdapter
from sglang_omni.serve.transcription_adapters.moss_transcribe_diarize import (
    _TIMESTAMP_TOLERANCE_S,
    MossTranscribeDiarizeAdapter,
)


def moss_adapter() -> MossTranscribeDiarizeAdapter:
    return MossTranscribeDiarizeAdapter()


def test_resolve_adapter_matches_moss_architecture() -> None:
    adapter = resolve_adapter(["MossTranscribeDiarizeForConditionalGeneration"])
    assert isinstance(adapter, MossTranscribeDiarizeAdapter)


def test_resolve_adapter_falls_back_to_default() -> None:
    assert isinstance(resolve_adapter(["SomethingElse"]), DefaultTranscriptionAdapter)
    assert isinstance(resolve_adapter([]), DefaultTranscriptionAdapter)
    assert isinstance(resolve_adapter(None), DefaultTranscriptionAdapter)


def test_postprocess_strips_special_tokens_only() -> None:
    adapter = moss_adapter()
    raw = "<|im_start|>[0.12][S01] hello[3.27]<|im_end|>"
    assert adapter.postprocess_text(raw) == "[0.12][S01] hello[3.27]"


def test_parse_multi_speaker_segments() -> None:
    adapter = moss_adapter()
    text = "[0.00][S01] Hello there.[1.20][1.30][S02] How are you.[3.00]"
    resp = adapter.build_verbose_response(
        text=text, language=None, audio_duration_s=3.0
    )
    assert [(s.id, s.start, s.end) for s in resp.segments] == [
        (0, 0.0, 1.2),
        (1, 1.3, 3.0),
    ]
    assert resp.segments[0].text == "[S01]Hello there."
    assert resp.segments[1].text == "[S02]How are you."
    assert resp.language is None


def test_fallback_when_no_markup() -> None:
    adapter = moss_adapter()
    resp = adapter.build_verbose_response(
        text="plain transcript no markers", language="en", audio_duration_s=4.5
    )
    assert len(resp.segments) == 1
    seg = resp.segments[0]
    assert seg.start == 0.0
    assert seg.end == 4.5
    assert seg.text == "[S01]plain transcript no markers"
    assert resp.duration == 4.5


@pytest.mark.parametrize(
    "speakers", [("", "", ""), ("[S02]", "", "[S03]"), ("", "[S02]", "")]
)
def test_timestamped_segments_without_speaker_tags(
    speakers: tuple[str, str, str],
) -> None:
    """Missing speaker tags must not merge segments or leave timestamps in speech."""
    text = (
        f"[13.44]{speakers[0]}xin chào[15.44]"
        f"[33.61]{speakers[1]}văn bản này không có thẻ người nói[35.61]"
        f"[38.90]{speakers[2]}nó không được phân tích chính xác[40.90]"
    )
    response = moss_adapter().build_verbose_response(text, None, 41.22)
    assert response.text == text
    assert [
        (segment.id, segment.start, segment.end, segment.text)
        for segment in response.segments
    ] == [
        (0, 13.44, 15.44, f"{speakers[0] or '[S01]'}xin chào"),
        (
            1,
            33.61,
            35.61,
            f"{speakers[1] or '[S01]'}văn bản này không có thẻ người nói",
        ),
        (2, 38.90, 40.90, f"{speakers[2] or '[S01]'}nó không được phân tích chính xác"),
    ]


def test_empty_text_yields_no_segments() -> None:
    adapter = moss_adapter()
    resp = adapter.build_verbose_response(
        text="   ", language="en", audio_duration_s=2.0
    )
    assert resp.segments == []


@pytest.mark.parametrize("speaker_tag, speaker", [("", "S01"), ("[S02]", "S02")])
def test_diarized_response_separates_speaker_from_speech(
    speaker_tag: str,
    speaker: str,
) -> None:
    """Diarized JSON must expose speaker identity without embedding model markup in speech."""
    response = moss_adapter().build_diarized_response(
        f"<|im_start|>[13.44]{speaker_tag} xin chào[15.44]"
        "[33.61][S03] chào bạn[35.61]<|im_end|>",
        36.0,
    )
    assert response.model_dump(exclude_none=True) == {
        "task": "transcribe",
        "duration": 36.0,
        "text": "xin chào chào bạn",
        "segments": [
            {
                "type": "transcript.text.segment",
                "id": "0",
                "start": 13.44,
                "end": 15.44,
                "speaker": speaker,
                "text": "xin chào",
            },
            {
                "type": "transcript.text.segment",
                "id": "1",
                "start": 33.61,
                "end": 35.61,
                "speaker": "S03",
                "text": "chào bạn",
            },
        ],
    }


@pytest.mark.parametrize(
    "text, expected_speaker", [("hello", "S01"), ("[S02]hello", "S02")]
)
def test_diarized_response_without_timestamps(text: str, expected_speaker: str) -> None:
    """Whole-audio fallback must keep a separate speaker label."""
    response = moss_adapter().build_diarized_response(text, 4.5)
    assert response.text == "hello"
    assert [
        (segment.start, segment.end, segment.speaker, segment.text)
        for segment in response.segments
    ] == [(0.0, 4.5, expected_speaker, "hello")]


def test_diarized_response_empty_transcript() -> None:
    """Silence must not create a fabricated speaker segment."""
    response = moss_adapter().build_diarized_response("<|im_end|>", 2.0)
    assert response.text == ""
    assert response.segments == []


def test_diarized_response_sanitizes_timestamps() -> None:
    """Diarized timestamps must receive the same bounds repair as verbose JSON."""
    response = moss_adapter().build_diarized_response("[0][S02]hello[999]", 4.5)
    assert response.segments[0].end == 4.5 + _TIMESTAMP_TOLERANCE_S


def test_default_adapter_single_segment() -> None:
    adapter = DefaultTranscriptionAdapter()
    resp = adapter.build_verbose_response(
        text="hello world", language=None, audio_duration_s=1.5
    )
    assert len(resp.segments) == 1
    assert resp.segments[0].text == "hello world"
    assert resp.segments[0].end == 1.5


def test_sanitize_clamps_timestamp_beyond_audio_duration() -> None:
    adapter = moss_adapter()
    duration_s = 4601.3
    text = "[0.00][S01] Hello.[1.20][3797.61][S02] Trade in phones.[4809.84]"
    resp = adapter.build_verbose_response(
        text=text, language=None, audio_duration_s=duration_s
    )
    assert len(resp.segments) == 2
    assert resp.segments[0].start == 0.0
    assert resp.segments[0].end == 1.2
    assert resp.segments[1].start == 3797.61
    assert resp.segments[1].end == round(duration_s + _TIMESTAMP_TOLERANCE_S, 2)


def test_sanitize_keeps_end_within_tolerance_of_duration() -> None:
    adapter = moss_adapter()
    duration_s = 10.0
    end = duration_s + _TIMESTAMP_TOLERANCE_S / 2
    text = f"[0.00][S01] Hello.[{end:.2f}]"
    resp = adapter.build_verbose_response(
        text=text, language=None, audio_duration_s=duration_s
    )
    assert resp.segments[0].end == round(end, 2)
