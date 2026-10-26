from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from moxtral.errors import InputError
from moxtral.models import (
    DEFAULT_TRANSCRIPTION_MODEL,
    ApiResult,
    Confidence,
    InputSource,
    OcrRequest,
    OcrSourceKind,
    Operation,
    OutputFormat,
    SavedResult,
    SourceKind,
    TableFormat,
    TimestampGranularity,
    TranscriptionRequest,
    build_ocr_request,
    build_transcription_request,
    model_family,
    ocr_request_metadata,
    transcription_request_metadata,
)


@pytest.fixture
def source() -> InputSource:
    return InputSource(
        kind=SourceKind.FILE,
        value="/tmp/input.bin",
        filename="input.bin",
        path=Path("/tmp/input.bin"),
    )


def test_enum_values_are_stable() -> None:
    assert [kind.value for kind in SourceKind] == ["file", "url"]
    assert [kind.value for kind in OcrSourceKind] == ["document", "image"]
    assert [operation.value for operation in Operation] == ["ocr", "transcription"]
    assert [output.value for output in OutputFormat] == ["md", "json", "both"]


def test_request_dataclasses_are_frozen_and_slotted(source: InputSource) -> None:
    ocr = OcrRequest(source=source, model="ocr")
    transcription = TranscriptionRequest(source=source, model="audio")

    for request, field in (
        (source, "filename"),
        (ocr, "model"),
        (transcription, "model"),
    ):
        assert not hasattr(request, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(request, field, "changed")


def test_api_result_is_frozen_and_slotted(source: InputSource) -> None:
    result = ApiResult(
        operation=Operation.OCR,
        source=source,
        request_metadata={"model": "ocr"},
        response={"pages": []},
        created_at=datetime(2026, 7, 2, tzinfo=UTC),
    )

    assert result.operation is Operation.OCR
    assert result.request_metadata == {"model": "ocr"}
    assert result.response == {"pages": []}
    assert not hasattr(result, "__dict__")
    field = "operation"
    with pytest.raises(FrozenInstanceError):
        setattr(result, field, Operation.TRANSCRIPTION)


def test_saved_result_is_frozen_and_slotted(tmp_path: Path) -> None:
    markdown = tmp_path / "result.md"
    json = tmp_path / "result.json"
    saved = SavedResult(markdown=markdown, json=json)

    assert saved.markdown == markdown
    assert saved.json == json
    assert not hasattr(saved, "__dict__")
    field = "markdown"
    with pytest.raises(FrozenInstanceError):
        setattr(saved, field, None)


def test_api_result_requires_aware_creation_time(source: InputSource) -> None:
    with pytest.raises(ValueError, match=r"created_at.*timezone-aware"):
        ApiResult(
            operation=Operation.OCR,
            source=source,
            request_metadata={},
            response={},
            created_at=datetime(2026, 7, 2),
        )


def test_ocr_builder_populates_all_options(source: InputSource) -> None:
    request = build_ocr_request(
        source=source,
        model="mistral-ocr-latest",
        pages=" 0, 2-4 ",
        table_format="markdown",
        extract_header=True,
        extract_footer=True,
        include_images=True,
        image_limit=10,
        image_min_size=32,
        include_blocks=True,
        confidence="word",
        timeout_seconds=30,
    )

    assert request == OcrRequest(
        source=source,
        model="mistral-ocr-latest",
        pages="0,2-4",
        table_format="markdown",
        extract_header=True,
        extract_footer=True,
        include_images=True,
        image_limit=10,
        image_min_size=32,
        include_blocks=True,
        confidence="word",
        timeout_ms=30_000,
    )


@pytest.mark.parametrize(
    "pages",
    ["", "0,", ",0", "0,,2", "-1", "1--2", "4-2", "one", "1-2-3"],
)
def test_invalid_page_syntax_is_rejected(
    source: InputSource,
    pages: str,
) -> None:
    with pytest.raises(InputError, match="--pages"):
        build_ocr_request(source=source, model="ocr", pages=pages)


@pytest.mark.parametrize("pages", ["0", "0,2,4", "0-0", "1-3", " 0 , 2-4 "])
def test_valid_page_syntax_is_normalized(
    source: InputSource,
    pages: str,
) -> None:
    request = build_ocr_request(source=source, model="ocr", pages=pages)

    assert request.pages == ",".join(part.strip() for part in pages.strip().split(","))


@pytest.mark.parametrize(
    "pages",
    [
        "9" * 5000,
        f"1-{'9' * 5000}",
    ],
    ids=["single", "range"],
)
def test_huge_page_numbers_are_translated_to_input_error(
    source: InputSource,
    pages: str,
) -> None:
    with pytest.raises(InputError, match=r"--pages"):
        build_ocr_request(source=source, model="ocr", pages=pages)


@pytest.mark.parametrize("model", ["", " ", "\t\n"])
def test_blank_ocr_model_is_rejected(source: InputSource, model: str) -> None:
    with pytest.raises(InputError, match="--model"):
        build_ocr_request(source=source, model=model)


@pytest.mark.parametrize("name", ["table_format", "confidence"])
def test_ocr_choice_values_are_checked_at_runtime(
    source: InputSource,
    name: str,
) -> None:
    if name == "table_format":
        with pytest.raises(InputError, match="--table-format"):
            build_ocr_request(
                source=source,
                model="ocr",
                table_format=cast(TableFormat, "csv"),
            )
    else:
        with pytest.raises(InputError, match="--confidence"):
            build_ocr_request(
                source=source,
                model="ocr",
                confidence=cast(Confidence, "line"),
            )


@pytest.mark.parametrize(
    ("name", "value"),
    [("image_limit", -1), ("image_min_size", -1)],
)
def test_image_controls_must_be_nonnegative(
    source: InputSource,
    name: str,
    value: int,
) -> None:
    with pytest.raises(InputError, match=f"--{name.replace('_', '-')}"):
        if name == "image_limit":
            build_ocr_request(
                source=source,
                model="ocr",
                include_images=True,
                image_limit=value,
            )
        else:
            build_ocr_request(
                source=source,
                model="ocr",
                include_images=True,
                image_min_size=value,
            )


@pytest.mark.parametrize("name", ["image_limit", "image_min_size"])
def test_image_controls_require_including_images(
    source: InputSource,
    name: str,
) -> None:
    with pytest.raises(InputError, match="--include-images"):
        if name == "image_limit":
            build_ocr_request(source=source, model="ocr", image_limit=1)
        else:
            build_ocr_request(source=source, model="ocr", image_min_size=1)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_ocr_timeout_must_be_positive_and_finite(
    source: InputSource,
    timeout: float,
) -> None:
    with pytest.raises(InputError, match="--timeout"):
        build_ocr_request(source=source, model="ocr", timeout_seconds=timeout)


def test_sub_millisecond_timeout_rounds_up_to_positive_millisecond(
    source: InputSource,
) -> None:
    request = build_ocr_request(
        source=source,
        model="ocr",
        timeout_seconds=0.0001,
    )

    assert request.timeout_ms == 1


def test_ocr_timeout_too_large_for_milliseconds_is_rejected(
    source: InputSource,
) -> None:
    with pytest.raises(InputError, match=r"--timeout.*too large"):
        build_ocr_request(
            source=source,
            model="ocr",
            timeout_seconds=1e306,
        )


def test_transcription_builder_populates_and_deduplicates_timestamps(
    source: InputSource,
) -> None:
    request = build_transcription_request(
        source=source,
        model="voxtral-mini-latest",
        temperature=0.25,
        diarize=True,
        context_bias=("Mistral", "Codestral"),
        timestamps=("word", "segment", "word"),
        timeout_seconds=15,
    )

    assert request == TranscriptionRequest(
        source=source,
        model="voxtral-mini-latest",
        temperature=0.25,
        diarize=True,
        context_bias=("Mistral", "Codestral"),
        timestamps=("word", "segment"),
        timeout_ms=15_000,
    )


@pytest.mark.parametrize("model", ["", " ", "\t\n"])
def test_blank_transcription_model_is_rejected(
    source: InputSource,
    model: str,
) -> None:
    with pytest.raises(InputError, match="--model"):
        build_transcription_request(source=source, model=model)


@pytest.mark.parametrize("language", ["", " ", "\t\n"])
def test_blank_language_is_rejected(
    source: InputSource,
    language: str,
) -> None:
    with pytest.raises(InputError, match="--language"):
        build_transcription_request(source=source, model="audio", language=language)


def test_v2_language_with_timestamps_rejected(source: InputSource) -> None:
    with pytest.raises(InputError, match="Voxtral 2") as info:
        build_transcription_request(
            source=source,
            model="voxtral-mini-latest",
            language="en",
            timestamps=("segment",),
        )
    assert str(info.value) == (
        "--language cannot be combined with --timestamps on Voxtral 2 models."
    )


def test_v3_word_timestamps_rejected(source: InputSource) -> None:
    with pytest.raises(InputError, match="segment") as info:
        build_transcription_request(
            source=source,
            model="voxtral-small-transcribe-3",
            timestamps=("word",),
        )
    assert str(info.value) == (
        "--timestamps word is not supported by Voxtral 3 models; use segment."
    )


def test_v3_word_with_segment_also_rejected(source: InputSource) -> None:
    with pytest.raises(InputError, match="segment"):
        build_transcription_request(
            source=source,
            model="voxtral-mini-transcribe-3",
            timestamps=("segment", "word"),
        )


def test_v3_language_with_segment_allowed(source: InputSource) -> None:
    request = build_transcription_request(
        source=source,
        model="voxtral-small-transcribe-3",
        language="de",
        timestamps=("segment",),
    )
    assert request.language == "de"
    assert request.timestamps == ("segment",)


def test_unknown_model_has_no_family_rules(source: InputSource) -> None:
    request = build_transcription_request(
        source=source,
        model="my-finetune",
        language="de",
        timestamps=("word",),
        diarize=True,
    )
    assert request.language == "de"
    assert request.timestamps == ("word",)
    assert request.diarize is True


def test_v3_diarize_adds_segment_once(source: InputSource) -> None:
    added = build_transcription_request(
        source=source, model="voxtral-small-transcribe-3", diarize=True
    )
    assert added.timestamps == ("segment",)
    present = build_transcription_request(
        source=source,
        model="voxtral-small-transcribe-3",
        diarize=True,
        timestamps=("segment",),
    )
    assert present.timestamps == ("segment",)


def test_v2_diarize_does_not_add_segment(source: InputSource) -> None:
    request = build_transcription_request(
        source=source, model="voxtral-mini-latest", diarize=True
    )
    assert request.timestamps == ()


def test_request_metadata_reflects_auto_added_segment(source: InputSource) -> None:
    request = build_transcription_request(
        source=source, model="voxtral-mini-transcribe-3", diarize=True
    )
    assert transcription_request_metadata(request)["timestamps"] == ["segment"]


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("voxtral-small-transcribe-3", "v3"),
        ("voxtral-mini-transcribe-3", "v3"),
        ("voxtral-mini-transcribe-3-0", "v3"),
        ("voxtral-mini-transcribe-latest", "v3"),
        ("voxtral-small-transcribe-latest", "v3"),
        ("voxtral-mini-transcribe-realtime-3", "v3"),
        ("voxtral-mini-latest", "v2"),
        ("voxtral-mini-2602", "v2"),
        ("voxtral-mini-2507", "v2"),
        ("voxtral-mini-transcribe-2507", "v2"),
        ("voxtral-small-latest", "v2"),
        ("voxtral-small-2507", "v2"),
        ("voxtral-mini-transcribe-realtime-2602", "v2"),
        ("voxtral-mini-2602-x", "v2"),
        ("voxtral-mini-latest-x", "unknown"),
        ("my-finetune", "unknown"),
        ("audio", "unknown"),
    ],
)
def test_model_family(model: str, family: str) -> None:
    assert model_family(model) == family


def test_default_transcription_model() -> None:
    assert DEFAULT_TRANSCRIPTION_MODEL == "voxtral-mini-transcribe-3"


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -float("inf")])
def test_temperature_must_be_finite(
    source: InputSource,
    temperature: float,
) -> None:
    with pytest.raises(InputError, match="--temperature"):
        build_transcription_request(
            source=source,
            model="audio",
            temperature=temperature,
        )


def test_temperature_has_no_undocumented_range(source: InputSource) -> None:
    request = build_transcription_request(
        source=source,
        model="audio",
        temperature=-100.5,
    )

    assert request.temperature == -100.5


def test_context_bias_accepts_at_most_100_values(source: InputSource) -> None:
    accepted = tuple(f"value-{index}" for index in range(100))
    assert (
        build_transcription_request(
            source=source,
            model="audio",
            context_bias=accepted,
        ).context_bias
        == accepted
    )

    with pytest.raises(InputError, match="--context-bias"):
        build_transcription_request(
            source=source,
            model="audio",
            context_bias=(*accepted, "one-too-many"),
        )


def test_context_bias_rejects_blank_values(source: InputSource) -> None:
    with pytest.raises(InputError, match="--context-bias"):
        build_transcription_request(
            source=source,
            model="audio",
            context_bias=("valid", " "),
        )


def test_context_bias_preserves_values_and_order(source: InputSource) -> None:
    values = (" first ", "second", "first")

    request = build_transcription_request(
        source=source,
        model="audio",
        context_bias=values,
    )

    assert request.context_bias == values


def test_timestamp_values_are_checked_at_runtime(source: InputSource) -> None:
    with pytest.raises(InputError, match="--timestamps"):
        build_transcription_request(
            source=source,
            model="audio",
            timestamps=(cast(TimestampGranularity, "sentence"),),
        )


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_transcription_timeout_must_be_positive_and_finite(
    source: InputSource,
    timeout: float,
) -> None:
    with pytest.raises(InputError, match="--timeout"):
        build_transcription_request(
            source=source,
            model="audio",
            timeout_seconds=timeout,
        )


def test_transcription_timeout_too_large_for_milliseconds_is_rejected(
    source: InputSource,
) -> None:
    with pytest.raises(InputError, match=r"--timeout.*too large"):
        build_transcription_request(
            source=source,
            model="audio",
            timeout_seconds=1e306,
        )


def test_ocr_request_metadata_mirrors_request(source: InputSource) -> None:
    request = build_ocr_request(
        source=source,
        model="mistral-ocr-latest",
        pages="0,2-4",
        table_format="html",
        extract_header=True,
        include_images=True,
        image_limit=3,
        confidence="word",
        timeout_seconds=10,
    )
    assert ocr_request_metadata(request) == {
        "model": "mistral-ocr-latest",
        "pages": "0,2-4",
        "table_format": "html",
        "extract_header": True,
        "extract_footer": False,
        "include_images": True,
        "image_limit": 3,
        "image_min_size": None,
        "include_blocks": False,
        "confidence": "word",
        "retries": 3,
        "timeout_ms": 10_000,
    }


def test_transcription_request_metadata_mirrors_request(source: InputSource) -> None:
    request = build_transcription_request(
        source=source,
        model="voxtral-mini-latest",
        temperature=0.2,
        diarize=True,
        context_bias=("Mistral",),
        timestamps=("segment",),
        timeout_seconds=10,
    )
    assert transcription_request_metadata(request) == {
        "model": "voxtral-mini-latest",
        "language": None,
        "temperature": 0.2,
        "diarize": True,
        "context_bias": ["Mistral"],
        "timestamps": ["segment"],
        "retries": 3,
        "timeout_ms": 10_000,
    }


def test_requests_default_to_three_retries(source: InputSource) -> None:
    ocr = build_ocr_request(source=source, model="m")
    transcription = build_transcription_request(source=source, model="m")

    assert ocr.retries == 3
    assert transcription.retries == 3


@pytest.mark.parametrize("retries", [-1, True, 2.0])
def test_invalid_retries_are_rejected(source: InputSource, retries: object) -> None:
    with pytest.raises(InputError, match="--retries"):
        build_ocr_request(
            source=source,
            model="m",
            retries=retries,  # type: ignore[arg-type]
        )
    with pytest.raises(InputError, match="--retries"):
        build_transcription_request(
            source=source,
            model="m",
            retries=retries,  # type: ignore[arg-type]
        )


def test_request_metadata_includes_retries(source: InputSource) -> None:
    ocr = build_ocr_request(source=source, model="m", retries=0)
    transcription = build_transcription_request(source=source, model="m", retries=5)

    assert ocr_request_metadata(ocr)["retries"] == 0
    assert transcription_request_metadata(transcription)["retries"] == 5
