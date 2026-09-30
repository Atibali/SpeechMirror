from __future__ import annotations

import hashlib
import json
import os
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import backend.config as backend_config
from backend.config import CACHE_ROOT, ensure_directories
from backend.services.alignment import align_transcript_to_audio, validate_alignment
from speechmirror.evaluation import compare_aligned_features, load_pipeline_config
from speechmirror.features import SAMPLE_RATE, extract_feature_frame, load_audio


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transcript_sha256(transcript: str) -> str:
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()


def _alignment_model_version(
    method: str, model_id: str = "default-for-language", device: str = "cpu"
) -> str:
    if method == "estimate":
        return "uniform-estimate-v1"
    try:
        whisperx_version = metadata.version("whisperx")
    except metadata.PackageNotFoundError:
        whisperx_version = "unknown"
    return f"whisperx-{whisperx_version}:{model_id}:{device}"


def _cached_features(path: str | Path, signal: np.ndarray, sample_rate: int):
    candidate_path = Path(path).resolve()
    try:
        candidate_path.relative_to(backend_config.DATASET_ROOT.resolve())
    except ValueError:
        return extract_feature_frame(signal, sample_rate)

    ensure_directories()
    config = load_pipeline_config()
    feature_config = json.dumps(
        config["features"], sort_keys=True, separators=(",", ":")
    )
    cache_key = hashlib.sha256(
        f"{_file_sha256(path)}|{sample_rate}|{feature_config}|feature-v3".encode("utf-8")
    ).hexdigest()
    cache_path = CACHE_ROOT / f"{cache_key}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    features = extract_feature_frame(signal, sample_rate)
    with tempfile.NamedTemporaryFile(
        dir=CACHE_ROOT, suffix=".parquet.tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        features.to_parquet(temporary, index=False)
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return features


def _cached_alignment(
    path: str | Path,
    signal: np.ndarray,
    sample_rate: int,
    transcript: str,
    method: str,
    language: str,
    device: str,
) -> tuple[list[dict], str]:
    candidate_path = Path(path).resolve()
    try:
        candidate_path.relative_to(backend_config.DATASET_ROOT.resolve())
    except ValueError:
        return align_transcript_to_audio(
            signal,
            sample_rate,
            transcript,
            method=method,
            language=language,
            device=device,
        )

    ensure_directories()
    audio_hash = _file_sha256(path)
    alignment_config = load_pipeline_config()["alignment"]
    configured_model = alignment_config.get("model_id", "default-for-language")
    key = hashlib.sha256(
        f"{audio_hash}|{_transcript_sha256(transcript)}|{method}|{language}|"
        f"{json.dumps(alignment_config, sort_keys=True, separators=(',', ':'))}|"
        f"{_alignment_model_version(method, configured_model, device)}|align-v3".encode("utf-8")
    ).hexdigest()
    cache_path = CACHE_ROOT / f"{key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return (
            validate_alignment(
                cached["words"], transcript, len(signal) / sample_rate
            ),
            cached["method"],
        )
    words, actual_method = align_transcript_to_audio(
        signal,
        sample_rate,
        transcript,
        method=method,
        language=language,
        device=device,
    )
    payload = json.dumps({"words": words, "method": actual_method}, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=CACHE_ROOT,
        suffix=".json.tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return words, actual_method


def analyze_audio_pair(
    ideal_audio_path: str | Path,
    participant_audio_path: str | Path,
    transcript: str,
    baseline_transcript: str,
    baseline_id: str = "dataset-sample",
    alignment_method: str = "whisperx",
    language: str = "en",
    device: str | None = None,
) -> dict[str, Any]:
    transcript = " ".join(transcript.split())
    baseline_transcript = " ".join(baseline_transcript.split())
    if not transcript or not baseline_transcript:
        raise ValueError("A non-empty exact transcript is required.")
    device = device or os.environ.get("SPEECHMIRROR_DEVICE", "cpu")

    ideal_signal, ideal_sr = load_audio(ideal_audio_path, sample_rate=SAMPLE_RATE)
    participant_signal, participant_sr = load_audio(
        participant_audio_path, sample_rate=SAMPLE_RATE
    )
    ideal_features = _cached_features(ideal_audio_path, ideal_signal, ideal_sr)
    participant_features = _cached_features(
        participant_audio_path, participant_signal, participant_sr
    )
    if not ideal_features["speech_activity"].any():
        raise ValueError("Baseline audio has no detected speech activity.")
    if not participant_features["speech_activity"].any():
        raise ValueError(
            "Participant audio has no detected speech activity; record the speech again "
            "or adjust microphone input level."
        )
    ideal_alignment, ideal_method = _cached_alignment(
        ideal_audio_path,
        ideal_signal,
        ideal_sr,
        baseline_transcript,
        alignment_method,
        language,
        device,
    )
    participant_alignment, participant_method = _cached_alignment(
        participant_audio_path,
        participant_signal,
        participant_sr,
        transcript,
        alignment_method,
        language,
        device,
    )
    if ideal_method != participant_method:
        raise RuntimeError("Baseline and participant audio used different alignment methods.")

    config = load_pipeline_config()
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    model_version = _alignment_model_version(
        alignment_method,
        config["alignment"].get("model_id", "default-for-language"),
        device,
    )
    input_hash = hashlib.sha256(
        "|".join(
            [
                _file_sha256(ideal_audio_path),
                _file_sha256(participant_audio_path),
                _transcript_sha256(transcript),
                _transcript_sha256(baseline_transcript),
                baseline_id,
                config_hash,
                model_version,
                device,
            ]
        ).encode("utf-8")
    ).hexdigest()
    result = compare_aligned_features(
        ideal_features=ideal_features,
        participant_features=participant_features,
        ideal_alignment=ideal_alignment,
        participant_alignment=participant_alignment,
        transcript=transcript,
        baseline_transcript=baseline_transcript,
        sample_id=baseline_id,
        input_hash=input_hash,
        alignment_method=participant_method,
        configuration_hash=config_hash,
        alignment_model_version=model_version,
    )
    result["audio"] = {
        "ideal_duration_seconds": round(len(ideal_signal) / ideal_sr, 3),
        "participant_duration_seconds": round(
            len(participant_signal) / participant_sr, 3
        ),
        "sample_rate": SAMPLE_RATE,
        "ideal_sha256": _file_sha256(ideal_audio_path),
        "participant_sha256": _file_sha256(participant_audio_path),
    }
    result["alignment_quality"] = {
        "ideal_coverage": round(len(ideal_alignment) / max(len(transcript.split()), 1), 3),
        "participant_coverage": round(
            len(participant_alignment) / max(len(transcript.split()), 1), 3
        ),
        "participant_mean_confidence": (
            round(
                float(
                    np.mean(
                        [
                            word["confidence"]
                            for word in participant_alignment
                            if word["confidence"] is not None
                        ]
                    )
                ),
                3,
            )
            if any(word["confidence"] is not None for word in participant_alignment)
            else None
        ),
    }
    active_flatness = participant_features.loc[
        participant_features["speech_activity"], "flatness"
    ]
    result["quality_warnings"] = []
    if not active_flatness.empty and float(active_flatness.median()) > 0.75:
        result["quality_warnings"].append(
            "High spectral flatness was detected in active audio; inspect background noise "
            "and microphone conditions before trusting the score."
        )
    if result["alignment_quality"]["participant_mean_confidence"] is not None and (
        result["alignment_quality"]["participant_mean_confidence"] < 0.4
    ):
        result["quality_warnings"].append(
            "Word alignment confidence is low; review timestamps and flaw regions manually."
        )
    return result


def feature_records(audio_path: str | Path) -> list[dict]:
    signal, sample_rate = load_audio(audio_path, sample_rate=SAMPLE_RATE)
    features = _cached_features(audio_path, signal, sample_rate)
    features = features.where(features.notna(), None).astype(object)
    return features.to_dict(orient="records")
