from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import normalize_features


def _contiguous_windows(mask: np.ndarray) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            windows.append((start, idx - 1))
            start = None
    if start is not None:
        windows.append((start, len(mask) - 1))
    return windows


def _feature_mentions(feature: str, mean_delta: float) -> tuple[str, str]:
    if feature == "pitch":
        if mean_delta < 0:
            return "pitch_drop", "Pitch falls below the baseline reference signal and reduces vocal clarity."
        return "pitch_rise", "Pitch rises above the baseline reference, which can indicate tension or strain."
    if feature == "energy":
        if mean_delta > 0:
            return "energy_spike", "Energy is elevated above the baseline, increasing perceived stress or arousal."
        return "energy_drop", "Energy unexpectedly falls below the baseline, suggesting weaker delivery or reduced intensity."
    if feature == "speech_rate":
        if mean_delta > 0:
            return "tempo_spike", "Speech rate has accelerated beyond the ideal pacing window."
        return "tempo_drop", "Speech rate is slower than the reference and may indicate hesitations."
    if feature == "pause_rate":
        if mean_delta > 0:
            return "pause_irregularity", "Pause intervals are unusually long or uneven relative to baseline timing."
        return "pause_reduction", "Pause variability is reduced compared with the expected baseline rhythm."
    return "feature_shift", "The delivery differs from the reference in a measurable acoustic dimension."


def compare_pair(ideal_frame: pd.DataFrame, participant_frame: pd.DataFrame) -> dict:
    ideal_norm = normalize_features(ideal_frame).reset_index(drop=True)
    participant_norm = normalize_features(participant_frame).reset_index(drop=True)

    if len(ideal_norm) != len(participant_norm):
        target_length = min(len(ideal_norm), len(participant_norm))
        ideal_norm = ideal_norm.iloc[:target_length].reset_index(drop=True)
        participant_norm = participant_norm.iloc[:target_length].reset_index(drop=True)

    delta_frame = pd.DataFrame({
        "time": ideal_norm["time"].to_numpy(dtype=np.float64),
        "energy": participant_norm["energy"].to_numpy() - ideal_norm["energy"].to_numpy(),
        "pitch": participant_norm["pitch"].to_numpy() - ideal_norm["pitch"].to_numpy(),
        "speech_rate": participant_norm["speech_rate"].to_numpy() - ideal_norm["speech_rate"].to_numpy(),
        "pause_rate": participant_norm["pause_rate"].to_numpy() - ideal_norm["pause_rate"].to_numpy(),
    })

    thresholds = {
        "pitch": 0.8,
        "energy": 0.7,
        "speech_rate": 0.7,
        "pause_rate": 0.7,
    }

    flaw_cards: list[dict] = []
    score_losses: list[float] = []
    for feature, threshold in thresholds.items():
        series = delta_frame[feature].to_numpy(dtype=np.float64)
        mean_delta = float(np.mean(series))
        abs_series = np.abs(series)
        mask = abs_series > threshold
        if not np.any(mask):
            continue
        for start_idx, end_idx in _contiguous_windows(mask):
            window_vals = series[start_idx : end_idx + 1]
            local_mean = float(np.mean(window_vals))
            start_time = float(delta_frame["time"].iloc[start_idx])
            end_time = float(delta_frame["time"].iloc[end_idx])
            severity = float(np.clip(np.abs(local_mean) / 2.0, 0.0, 1.0))
            feature_name, rationale = _feature_mentions(feature, mean_delta)
            flaw_cards.append({
                "feature": feature,
                "type": feature_name,
                "start": start_time,
                "end": end_time,
                "severity": round(severity, 3),
                "explanation": rationale,
            })
            score_losses.append(severity)

    score_losses = [min(1.0, value) for value in score_losses] or [0.0]
    aggregate_severity = float(np.mean(score_losses))
    final_score = max(0.0, 100.0 - aggregate_severity * 75.0)

    flaws = sorted(flaw_cards, key=lambda item: item["start"])
    return {
        "score": round(final_score, 2),
        "flaws": flaws,
        "deltas": delta_frame.to_dict(orient="records"),
        "summary": {
            "detected_regions": len(flaws),
            "average_severity": round(aggregate_severity, 3),
            "thresholds": thresholds,
        },
    }


def export_results(result: dict, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
