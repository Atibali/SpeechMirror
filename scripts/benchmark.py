from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import DATASET_ROOT
from backend.services.audio_service import analyze_audio_pair
from backend.services.dataset_service import load_manifest

LABEL_TYPES = {
    "SPEECH_RATE": {"PACE_FAST", "PACE_SLOW"},
    "PACE": {"PACE_FAST", "PACE_SLOW"},
    "PAUSE": {"PAUSE_LONG", "PAUSE_MISSING"},
    "PAUSE_DURATION": {"PAUSE_LONG", "PAUSE_MISSING"},
    "PAUSE_INTERVAL": {"PAUSE_LONG", "PAUSE_MISSING"},
    "PAUSE_RATIO": {"PAUSE_LONG", "PAUSE_MISSING"},
    "PITCH": {"PITCH_FLAT", "PITCH_SPIKE"},
    "F0": {"PITCH_FLAT", "PITCH_SPIKE"},
    "PITCH_RANGE": {"PITCH_FLAT", "PITCH_SPIKE"},
    "ENERGY": {"ENERGY_LOW", "ENERGY_HIGH"},
    "ENERGY_DB": {"ENERGY_LOW", "ENERGY_HIGH"},
    "RMS": {"ENERGY_LOW", "ENERGY_HIGH"},
    "CLARITY": {"CLARITY_SHIFT"},
    "SPECTRAL": {"CLARITY_SHIFT"},
    "MFCC": {"CLARITY_SHIFT"},
    "CADENCE": {"CADENCE"},
}


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def _labels_by_sample() -> dict[str, list[dict]]:
    path = DATASET_ROOT / "labels" / "flaw_regions.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        labels: dict[str, list[dict]] = {}
        for row in csv.DictReader(handle):
            labels.setdefault(row["sample_id"], []).append(
                {
                    "type": row["feature_expected"].upper(),
                    "start": float(row["start_sec"]),
                    "end": float(row["end_sec"]),
                    "severity": float(row["severity"]),
                }
            )
        return labels


def evaluate(method: str, split: str | None) -> dict:
    records = load_manifest(DATASET_ROOT / "manifest.csv")
    labels_by_sample = _labels_by_sample()
    selected = [record for record in records if not split or record.split == split]
    if not selected:
        raise ValueError("No records match the requested split.")

    true_positive = false_positive = false_negative = 0
    iou_values: list[float] = []
    timestamp_errors: list[float] = []
    severity_errors: list[float] = []
    repeat_score_deltas: list[float] = []
    ideal_false_positive_cases = 0
    repeated_hash_mismatches = 0

    for record in selected:
        transcript = Path(record.transcript_file).read_text(encoding="utf-8").strip()
        expected = labels_by_sample.get(record.sample_id, [])
        actual = analyze_audio_pair(
            record.ideal_audio,
            record.participant_audio,
            transcript,
            transcript,
            record.sample_id,
            method,
        )
        repeated = analyze_audio_pair(
            record.ideal_audio,
            record.participant_audio,
            transcript,
            transcript,
            record.sample_id,
            method,
        )
        repeat_score_deltas.append(abs(actual["overall_score"] - repeated["overall_score"]))
        repeated_hash_mismatches += int(actual["result_hash"] != repeated["result_hash"])
        ideal_result = analyze_audio_pair(
            record.ideal_audio,
            record.ideal_audio,
            transcript,
            transcript,
            record.sample_id,
            method,
        )
        ideal_false_positive_cases += int(bool(ideal_result["flaws"]))
        unmatched = list(actual["flaws"])

        for label in expected:
            expected_type = label["type"].replace(" ", "_")
            allowed_types = LABEL_TYPES.get(expected_type, {expected_type})
            matches = [
                (
                    interval_iou(
                        (label["start"], label["end"]),
                        (flaw["start"], flaw["end"]),
                    ),
                    index,
                    flaw,
                )
                for index, flaw in enumerate(unmatched)
                if flaw["type"] in allowed_types
            ]
            best = max(matches, default=(0.0, -1, None), key=lambda value: value[0])
            if best[0] >= 0.5 and best[2] is not None:
                true_positive += 1
                iou_values.append(best[0])
                flaw = best[2]
                timestamp_errors.extend(
                    [
                        abs(label["start"] - flaw["start"]),
                        abs(label["end"] - flaw["end"]),
                    ]
                )
                severity_errors.append(abs(label["severity"] / 4 - flaw["severity"]))
                unmatched.pop(best[1])
            else:
                false_negative += 1
        false_positive += len(unmatched)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None
    )
    report = {
        "split": split or "all",
        "sample_count": len(selected),
        "temporal_iou_mean": sum(iou_values) / len(iou_values) if iou_values else None,
        "precision": precision,
        "recall": recall,
        "timestamp_absolute_error_seconds_mean": (
            sum(timestamp_errors) / len(timestamp_errors) if timestamp_errors else None
        ),
        "severity_mae": (
            sum(severity_errors) / len(severity_errors) if severity_errors else None
        ),
        "repeated_score_max_delta": max(repeat_score_deltas, default=0.0),
        "repeated_result_hash_mismatches": repeated_hash_mismatches,
        "ideal_false_positive_rate": ideal_false_positive_cases / len(selected),
        "alignment_method": method,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detections against held-out temporal labels.")
    parser.add_argument("--method", choices=["whisperx", "estimate"], default="whisperx")
    parser.add_argument("--split", default="test")
    parser.add_argument("--all", action="store_true", help="Evaluate all samples, not only the test split.")
    parser.add_argument("--output", type=Path, default=DATASET_ROOT / "benchmark.json")
    args = parser.parse_args()
    result = evaluate(args.method, None if args.all else args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
