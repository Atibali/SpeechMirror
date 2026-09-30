from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

DEFAULT_DEVICE = os.environ.get("SPEECHMIRROR_DEVICE", "cpu")


class AlignmentUnavailable(RuntimeError):
    pass


class AlignmentError(ValueError):
    pass


@lru_cache(maxsize=2)
def _load_whisperx_model(language: str, device: str):
    try:
        import whisperx
    except Exception as exc:
        raise AlignmentUnavailable(
            "WhisperX could not be imported. Install its optional dependencies and "
            "model as described in README.md, or choose alignment_mode='estimate' "
            f"for explicitly approximate demo timestamps. Details: {exc}"
        ) from exc

    try:
        return whisperx, *whisperx.load_align_model(
            language_code=language, device=device
        )
    except Exception as exc:
        raise AlignmentUnavailable(
            f"WhisperX could not load the {language!r} alignment model on {device}: {exc}"
        ) from exc


def validate_alignment(
    words: list[dict], transcript: str, duration: float
) -> list[dict]:
    expected = transcript.split()
    if len(words) != len(expected):
        raise AlignmentError(
            f"Word alignment coverage mismatch: expected {len(expected)} words, "
            f"received {len(words)}."
        )
    validated: list[dict] = []
    last_start = -1.0
    for index, (word, expected_word) in enumerate(zip(words, expected)):
        try:
            start = float(word["start"])
            end = float(word["end"])
            token = str(word["word"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise AlignmentError(f"Invalid word alignment at index {index}.") from exc
        if token.casefold().strip(".,!?;:'\"()[]{}") != expected_word.casefold().strip(".,!?;:'\"()[]{}"):
            raise AlignmentError(
                f"Alignment transcript mismatch at word {index + 1}: "
                f"expected {expected_word!r}, received {token!r}."
            )
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise AlignmentError(f"Invalid start/end bounds for aligned word {token!r}.")
        if start < last_start:
            raise AlignmentError("Word alignment timestamps are not monotonically ordered.")
        if end > duration + 0.1:
            raise AlignmentError(f"Aligned word {token!r} exceeds audio duration.")
        raw_confidence = word.get("score", word.get("confidence"))
        confidence = None
        if raw_confidence is not None:
            try:
                candidate_confidence = float(raw_confidence)
                if np.isfinite(candidate_confidence) and 0 <= candidate_confidence <= 1:
                    confidence = candidate_confidence
            except (TypeError, ValueError):
                confidence = None
        validated.append(
            {
                "word": expected_word,
                "start": start,
                "end": end,
                "confidence": confidence,
            }
        )
        last_start = start
    return validated


def align_transcript_to_audio(
    signal: np.ndarray,
    sample_rate: int,
    transcript: str,
    *,
    method: str = "whisperx",
    language: str = "en",
    device: str = DEFAULT_DEVICE,
) -> tuple[list[dict], str]:
    transcript = " ".join(transcript.split())
    words = transcript.split()
    if not words:
        raise AlignmentError("Transcript must contain at least one word.")
    if sample_rate <= 0 or len(signal) == 0:
        raise AlignmentError("Audio signal and sample rate must be valid.")

    duration = len(signal) / sample_rate
    if method == "estimate":
        estimated = [
            {
                "word": word,
                "start": duration * index / len(words),
                "end": duration * (index + 1) / len(words),
            }
            for index, word in enumerate(words)
        ]
        return validate_alignment(estimated, transcript, duration), "uniform_estimate"
    if method != "whisperx":
        raise AlignmentError(f"Unsupported alignment method: {method}")

    whisperx, model, metadata = _load_whisperx_model(language, device)
    segment = [{"start": 0.0, "end": duration, "text": transcript}]
    try:
        result = whisperx.align(
            segment,
            model,
            metadata,
            np.asarray(signal, dtype=np.float32),
            device,
            return_char_alignments=False,
        )
    except Exception as exc:
        raise AlignmentError(f"WhisperX forced alignment failed: {exc}") from exc
    aligned_words = result.get("word_segments", [])
    return validate_alignment(aligned_words, transcript, duration), "whisperx_forced"
