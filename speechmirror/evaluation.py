from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from speechmirror.features import aggregate_word_features

PIPELINE_VERSION = "0.2.0"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "features": {
        "sample_rate": 16000,
        "frame_length": 1024,
        "hop_length": 320,
        "pitch_frame_length": 2048,
        "n_mfcc": 13,
    },
    "alignment": {
        "sample_rate": 16000,
        "language": "en",
        "model_id": "whisperx-default-for-language",
    },
    "detector": {
        "pace_relative_threshold": 0.25,
        "pause_delta_seconds": 0.6,
        "pitch_range_reduction": 0.45,
        "energy_z_threshold": 2.0,
        "energy_scale_floor_db": 3.0,
        "minimum_duration_seconds": 0.25,
        "minimum_alignment_confidence": 0.4,
        "minimum_speech_activity_fraction": 0.15,
        "pitch_spike_threshold_semitones": 4.0,
        "minimum_voiced_fraction": 0.4,
        "spectral_centroid_delta_hz": 1200.0,
        "spectral_flatness_delta": 0.2,
        "consistency_cv_delta": 0.5,
    },
    "severity_scales": {
        "pace_relative_delta_saturation": 1.0,
        "pause_delta_seconds_saturation": 2.0,
        "pitch_range_reduction_saturation": 0.8,
        "pitch_spike_semitone_saturation": 8.0,
        "energy_robust_z_saturation": 4.0,
        "clarity_delta_saturation": 3.0,
        "cadence_cv_saturation": 1.5,
    },
    "rubric_weights": {
        "pacing": 0.25,
        "pitch": 0.20,
        "energy": 0.20,
        "clarity": 0.15,
        "pausing": 0.10,
        "consistency": 0.10,
    },
}


def load_pipeline_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not config_path.exists():
        return DEFAULT_CONFIG
    config = json.loads(config_path.read_text(encoding="utf-8"))
    from speechmirror.features import (
        FRAME_HOP,
        FRAME_LENGTH,
        N_MFCC,
        PITCH_FRAME_LENGTH,
        SAMPLE_RATE,
    )

    expected_features = {
        "sample_rate": SAMPLE_RATE,
        "frame_length": FRAME_LENGTH,
        "hop_length": FRAME_HOP,
        "pitch_frame_length": PITCH_FRAME_LENGTH,
        "n_mfcc": N_MFCC,
    }
    if config.get("features") != expected_features:
        raise ValueError("Feature configuration does not match the active extraction parameters.")
    weights = config["rubric_weights"]
    if not weights or any(float(value) < 0 for value in weights.values()):
        raise ValueError("Rubric weights must be a non-empty set of non-negative values.")
    if not np.isclose(sum(map(float, weights.values())), 1.0):
        raise ValueError("Rubric weights must sum to 1.")
    return config


def normalize_transcript(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.casefold()))


def _safe_relative_delta(actual: float, baseline: float) -> float:
    return (actual - baseline) / max(abs(baseline), 1e-6)


def _word_records(
    ideal_features,
    participant_features,
    ideal_alignment: list[dict],
    participant_alignment: list[dict],
) -> list[dict]:
    ideal_words = aggregate_word_features(ideal_features, ideal_alignment)
    participant_words = aggregate_word_features(participant_features, participant_alignment)
    if len(ideal_words) != len(participant_words):
        raise ValueError("Baseline and participant alignment word counts differ.")
    records = []
    for ideal, participant in zip(ideal_words, participant_words):
        if normalize_transcript(ideal["word"]) != normalize_transcript(participant["word"]):
            raise ValueError(
                "Baseline and participant alignment transcripts differ at "
                f"{ideal['word']!r} vs {participant['word']!r}."
            )
        records.append(
            {
                "word": participant["word"],
                "start": participant["start"],
                "end": participant["end"],
                "baseline": ideal,
                "participant": participant,
            }
        )
    return records


def compare_aligned_features(
    ideal_features,
    participant_features,
    ideal_alignment: list[dict],
    participant_alignment: list[dict],
    transcript: str,
    baseline_transcript: str,
    sample_id: str,
    input_hash: str,
    alignment_method: str,
    configuration_hash: str,
    alignment_model_version: str = "unknown",
) -> dict:
    if normalize_transcript(transcript) != normalize_transcript(baseline_transcript):
        raise ValueError("Participant transcript must exactly match the selected baseline transcript.")
    if not ideal_alignment or not participant_alignment:
        raise ValueError("Word alignment returned no words.")

    config = load_pipeline_config()
    detector = config["detector"]
    severity_scales = config["severity_scales"]
    word_records = _word_records(
        ideal_features, participant_features, ideal_alignment, participant_alignment
    )
    baseline_speech = ideal_features.loc[
        ideal_features["speech_activity"].astype(bool), "energy_db"
    ].to_numpy(dtype=np.float64)
    if baseline_speech.size == 0:
        raise ValueError("Baseline audio has no detected voiced frames for energy normalization.")
    energy_reference_db = float(np.median(baseline_speech))
    energy_mad_db = float(np.median(np.abs(baseline_speech - energy_reference_db)))
    energy_scale_db = max(
        1.4826 * energy_mad_db,
        float(detector.get("energy_scale_floor_db", 3.0)),
    )
    candidates: list[dict] = []
    dimension_severities: dict[str, list[float]] = {
        key: [] for key in config["rubric_weights"]
    }
    evidence_rows: list[dict] = []

    def add_flaw(
        index: int,
        kind: str,
        dimension: str,
        severity: float,
        explanation: str,
        evidence: dict,
        start: float | None = None,
        end: float | None = None,
    ) -> None:
        row = word_records[index]
        candidates.append(
            {
                "flaw_id": "",
                "feature": kind.lower(),
                "type": kind,
                "dimension": dimension,
                "start": round(float(row["start"] if start is None else start), 3),
                "end": round(float(row["end"] if end is None else end), 3),
                "severity": round(float(severity), 3),
                "evidence": evidence,
                "explanation": explanation,
                "action": _action_for(kind),
                "transcript": row["word"],
            }
        )

    for index, row in enumerate(word_records):
        base = row["baseline"]
        actual = row["participant"]
        word_duration = float(actual["duration"])
        confidence = actual.get("confidence")
        low_confidence = (
            (
                confidence is not None
                and confidence < detector.get("minimum_alignment_confidence", 0.4)
            )
            or actual["speech_activity_fraction"]
            < detector.get("minimum_speech_activity_fraction", 0.15)
        )

        base_rate = 1.0 / max(float(base["duration"]), 1e-6)
        actual_rate = 1.0 / max(word_duration, 1e-6)
        rate_delta = _safe_relative_delta(actual_rate, base_rate)
        if not low_confidence and abs(rate_delta) >= detector["pace_relative_threshold"]:
            kind = "PACE_FAST" if rate_delta > 0 else "PACE_SLOW"
            severity = min(
                abs(rate_delta) / severity_scales["pace_relative_delta_saturation"], 1.0
            )
            add_flaw(
                index,
                kind,
                "pacing",
                severity,
                f"This word was delivered {abs(rate_delta):.1%} "
                f"{'faster' if rate_delta > 0 else 'slower'} than the aligned baseline.",
                {
                    "baseline_words_per_second": round(base_rate, 3),
                    "participant_words_per_second": round(actual_rate, 3),
                    "relative_delta": round(rate_delta, 4),
                },
            )

        base_pause = float(base["pause_after"])
        actual_pause = float(actual["pause_after"])
        pause_delta = actual_pause - base_pause
        if not low_confidence and pause_delta >= detector["pause_delta_seconds"]:
            severity = min(
                pause_delta / severity_scales["pause_delta_seconds_saturation"], 1.0
            )
            add_flaw(
                index,
                "PAUSE_LONG",
                "pausing",
                severity,
                f"The pause after this word is {pause_delta:.2f}s longer than the baseline.",
                {
                    "baseline_pause_seconds": round(base_pause, 3),
                    "participant_pause_seconds": round(actual_pause, 3),
                    "delta_seconds": round(pause_delta, 3),
                },
                start=float(actual["end"]),
                end=float(actual["end"]) + actual_pause,
            )
        elif not low_confidence and -pause_delta >= detector["pause_delta_seconds"]:
            severity = min(
                abs(pause_delta) / severity_scales["pause_delta_seconds_saturation"], 1.0
            )
            add_flaw(
                index,
                "PAUSE_MISSING",
                "pausing",
                severity,
                f"The pause after this word is {abs(pause_delta):.2f}s shorter than the baseline.",
                {
                    "baseline_pause_seconds": round(base_pause, 3),
                    "participant_pause_seconds": round(actual_pause, 3),
                    "delta_seconds": round(pause_delta, 3),
                },
                start=float(actual["end"]),
                end=float(actual["end"]) + base_pause,
            )

        base_pitch_range = float(base["pitch_range_semitones"])
        actual_pitch_range = float(actual["pitch_range_semitones"])
        voiced_fraction = float(actual["voiced_fraction"])
        if (
            base_pitch_range > 1.0
            and not low_confidence
            and voiced_fraction >= detector["minimum_voiced_fraction"]
            and actual_pitch_range <= base_pitch_range * (1.0 - detector["pitch_range_reduction"])
        ):
            reduction = 1.0 - actual_pitch_range / base_pitch_range
            add_flaw(
                index,
                "PITCH_FLAT",
                "pitch",
                min(reduction / severity_scales["pitch_range_reduction_saturation"], 1.0),
                f"Pitch contour range is {reduction:.1%} below the aligned baseline "
                "after speaker-relative semitone normalization.",
                {
                    "baseline_pitch_range_semitones": round(base_pitch_range, 3),
                    "participant_pitch_range_semitones": round(actual_pitch_range, 3),
                    "relative_delta": round(-reduction, 4),
                },
            )

        baseline_pitch_contour = base["pitch_relative_semitones"]
        participant_pitch_contour = actual["pitch_relative_semitones"]
        if (
            not low_confidence
            and baseline_pitch_contour is not None
            and participant_pitch_contour is not None
        ):
            pitch_delta_semitones = float(
                participant_pitch_contour - baseline_pitch_contour
            )
            if abs(pitch_delta_semitones) >= detector["pitch_spike_threshold_semitones"]:
                add_flaw(
                    index,
                    "PITCH_SPIKE",
                    "pitch",
                    min(
                        abs(pitch_delta_semitones)
                        / severity_scales["pitch_spike_semitone_saturation"],
                        1.0,
                    ),
                    f"Speaker-relative pitch contour shifts by {pitch_delta_semitones:+.1f} "
                    "semitones from the aligned baseline.",
                    {
                        "baseline_pitch_relative_semitones": round(
                            float(baseline_pitch_contour), 3
                        ),
                        "participant_pitch_relative_semitones": round(
                            float(participant_pitch_contour), 3
                        ),
                        "delta_semitones": round(pitch_delta_semitones, 3),
                    },
                )

        energy_delta = float(actual["energy_db"] - base["energy_db"])
        baseline_energy_z = float((base["energy_db"] - energy_reference_db) / energy_scale_db)
        participant_energy_z = float(
            (actual["energy_db"] - energy_reference_db) / energy_scale_db
        )
        energy_z_delta = participant_energy_z - baseline_energy_z
        if not low_confidence and abs(energy_z_delta) >= detector["energy_z_threshold"]:
            kind = "ENERGY_HIGH" if energy_z_delta > 0 else "ENERGY_LOW"
            severity = min(
                abs(energy_z_delta) / severity_scales["energy_robust_z_saturation"], 1.0
            )
            add_flaw(
                index,
                kind,
                "energy",
                severity,
                f"Baseline-normalized energy differs by {energy_z_delta:+.2f} robust z; "
                f"the raw difference is {energy_delta:+.1f} dB.",
                {
                    "baseline_energy_dbfs": round(float(base["energy_db"]), 2),
                    "participant_energy_dbfs": round(float(actual["energy_db"]), 2),
                    "delta_db": round(energy_delta, 2),
                    "baseline_energy_robust_z": round(baseline_energy_z, 3),
                    "participant_energy_robust_z": round(participant_energy_z, 3),
                    "baseline_median_dbfs": round(energy_reference_db, 2),
                    "baseline_robust_scale_db": round(energy_scale_db, 2),
                    "normalized_delta": round(energy_z_delta, 3),
                },
            )

        centroid_delta = float(actual["centroid"] - base["centroid"])
        flatness_delta = float(actual["flatness"] - base["flatness"])
        centroid_threshold = float(detector.get("spectral_centroid_delta_hz", 1200.0))
        flatness_threshold = float(detector.get("spectral_flatness_delta", 0.2))
        if not low_confidence and (
            abs(centroid_delta) >= centroid_threshold
            or abs(flatness_delta) >= flatness_threshold
        ):
            severity = min(
                max(
                    abs(centroid_delta) / max(centroid_threshold, 1e-6),
                    abs(flatness_delta) / max(flatness_threshold, 1e-6),
                ) / severity_scales["clarity_delta_saturation"],
                1.0
            )
            add_flaw(
                index,
                "CLARITY_SHIFT",
                "clarity",
                severity,
                "The aligned word's spectral profile differs materially from the reference; "
                "this is an acoustic change indicator, not a semantic or intelligibility judgment.",
                {
                    "baseline_centroid_hz": round(float(base["centroid"]), 2),
                    "participant_centroid_hz": round(float(actual["centroid"]), 2),
                    "centroid_delta_hz": round(centroid_delta, 2),
                    "flatness_delta": round(flatness_delta, 4),
                },
            )

        evidence_rows.append(
            {
                "word": row["word"],
                "alignment_confidence": confidence,
                "speech_activity_fraction": round(
                    float(actual["speech_activity_fraction"]), 3
                ),
                "start": round(float(actual["start"]), 3),
                "end": round(float(actual["end"]), 3),
                "baseline_rate": round(base_rate, 3),
                "participant_rate": round(actual_rate, 3),
                "baseline_pitch_range": round(base_pitch_range, 3),
                "participant_pitch_range": round(actual_pitch_range, 3),
                "baseline_energy_dbfs": round(float(base["energy_db"]), 2),
                "participant_energy_dbfs": round(float(actual["energy_db"]), 2),
                "baseline_pause": round(base_pause, 3),
                "participant_pause": round(actual_pause, 3),
                "baseline_mfcc_1": round(float(base["mfcc_1"]), 3),
                "participant_mfcc_1": round(float(actual["mfcc_1"]), 3),
                "baseline_centroid": round(float(base["centroid"]), 3),
                "participant_centroid": round(float(actual["centroid"]), 3),
                "mfcc_1_delta": round(float(actual["mfcc_1"] - base["mfcc_1"]), 3),
                "centroid_delta": round(float(actual["centroid"] - base["centroid"]), 3),
                "flatness_delta": round(float(actual["flatness"] - base["flatness"]), 4),
            }
        )

    consistency_delta = float(detector.get("consistency_cv_delta", 0.5))
    for start_index in range(0, len(word_records), 3):
        window = word_records[start_index : start_index + 3]
        if len(window) < 3:
            continue
        if any(
            item["participant"].get("confidence") is not None
            and item["participant"]["confidence"]
            < detector.get("minimum_alignment_confidence", 0.4)
            for item in window
        ):
            continue
        baseline_rates = np.array(
            [1.0 / max(float(item["baseline"]["duration"]), 1e-6) for item in window]
        )
        participant_rates = np.array(
            [1.0 / max(float(item["participant"]["duration"]), 1e-6) for item in window]
        )
        baseline_cv = float(np.std(baseline_rates) / max(float(np.mean(baseline_rates)), 1e-6))
        participant_cv = float(
            np.std(participant_rates) / max(float(np.mean(participant_rates)), 1e-6)
        )
        cv_delta = participant_cv - baseline_cv
        if cv_delta >= consistency_delta:
            severity = min(
                cv_delta / severity_scales["cadence_cv_saturation"], 1.0
            )
            row = word_records[start_index]
            candidates.append(
                {
                    "flaw_id": "",
                    "feature": "cadence",
                    "type": "CADENCE",
                    "dimension": "consistency",
                    "start": round(float(row["start"]), 3),
                    "end": round(float(window[-1]["end"]), 3),
                    "severity": round(severity, 3),
                    "evidence": {
                        "baseline_rate_coefficient_of_variation": round(baseline_cv, 4),
                        "participant_rate_coefficient_of_variation": round(participant_cv, 4),
                        "delta": round(cv_delta, 4),
                    },
                    "explanation": (
                        "Speech-rate variability within this three-word window is higher "
                        "than in the aligned reference."
                    ),
                    "action": _action_for("CADENCE"),
                    "transcript": " ".join(item["word"] for item in window),
                }
            )

    minimum_duration = float(detector["minimum_duration_seconds"])
    merge_gap = minimum_duration
    candidates.sort(key=lambda item: (item["type"], item["start"], item["end"]))
    merged_flaws: list[dict] = []
    for candidate in candidates:
        previous = merged_flaws[-1] if merged_flaws else None
        if (
            previous
            and previous["type"] == candidate["type"]
            and candidate["start"] <= previous["end"] + merge_gap
        ):
            previous["end"] = max(previous["end"], candidate["end"])
            previous["severity"] = round(
                (previous["severity"] + candidate["severity"]) / 2, 3
            )
            previous["transcript"] += " " + candidate["transcript"]
            for key, value in candidate["evidence"].items():
                if isinstance(value, (int, float)) and key in previous["evidence"]:
                    previous["evidence"][key] = round(
                        (previous["evidence"][key] + value) / 2, 4
                    )
        else:
            merged_flaws.append(candidate.copy())

    merged_flaws = [
        flaw
        for flaw in merged_flaws
        if flaw["end"] - flaw["start"] >= minimum_duration
    ]
    flaws = sorted(merged_flaws, key=lambda item: (item["start"], -item["severity"]))
    for index, flaw in enumerate(flaws, start=1):
        flaw["flaw_id"] = f"F{index:03d}"
        dimension_severities[flaw["dimension"]].append(flaw["severity"])

    dimensions = {
        dimension: round(
            max(0.0, 100.0 - 100.0 * (float(np.mean(severities)) if severities else 0.0)),
            2,
        )
        for dimension, severities in dimension_severities.items()
    }
    weights = config["rubric_weights"]
    overall = sum(dimensions[name] * float(weight) for name, weight in weights.items())
    actual_config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if configuration_hash != actual_config_hash:
        raise ValueError("Provided configuration hash does not match the active pipeline config.")
    result_id = hashlib.sha256(
        f"{input_hash}|{sample_id}|{PIPELINE_VERSION}|{configuration_hash}|"
        f"{alignment_method}|{alignment_model_version}".encode("utf-8")
    ).hexdigest()

    return {
        "analysis_id": result_id[:16],
        "result_hash": result_id,
        "sample_id": sample_id,
        "overall_score": round(overall, 2),
        "dimensions": dimensions,
        "flaws": flaws,
        "word_evidence": evidence_rows,
        "alignment": participant_alignment,
        "alignment_method": alignment_method,
        "transcript": transcript,
        "summary": {
            "detected_regions": len(flaws),
            "average_severity": round(
                float(np.mean([item["severity"] for item in flaws])) if flaws else 0.0,
                3,
            ),
            "rubric_weights": weights,
        },
        "reproducibility": {
            "pipeline_version": PIPELINE_VERSION,
            "configuration_hash": configuration_hash,
            "input_hash": input_hash,
            "alignment_method": alignment_method,
            "alignment_model_version": alignment_model_version,
        },
    }


def _action_for(kind: str) -> str:
    actions = {
        "PACE_FAST": "Slow down through this word group and preserve the reference phrase rhythm.",
        "PACE_SLOW": "Maintain forward momentum and avoid stretching the word beyond the reference.",
        "PAUSE_LONG": "Shorten the pause or place it at a deliberate phrase boundary.",
        "PAUSE_MISSING": "Add a short pause after this phrase to restore separation and emphasis.",
        "PITCH_FLAT": "Use more pitch movement to emphasize the key phrase.",
        "PITCH_SPIKE": "Keep pitch movement controlled and closer to the reference contour.",
        "CADENCE": "Keep a steadier rhythm across the phrase and place pauses deliberately.",
        "CLARITY_SHIFT": "Check articulation, microphone distance, and room conditions in this phrase.",
        "ENERGY_LOW": "Increase vocal projection while keeping the microphone distance consistent.",
        "ENERGY_HIGH": "Reduce projection slightly and keep the loudness closer to the reference.",
    }
    return actions[kind]
