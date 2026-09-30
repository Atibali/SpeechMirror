from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf

from backend.config import DATASET_ROOT
from backend.services.dataset_service import load_manifest
from backend.services.alignment import AlignmentError, validate_alignment
from speechmirror.evaluation import normalize_transcript


def validate_dataset() -> int:
    records = load_manifest(DATASET_ROOT / "manifest.csv")
    text_splits: dict[str, set[str]] = {}
    issues: list[str] = []
    for record in records:
        transcript = Path(record.transcript_file).read_text(encoding="utf-8").strip()
        if not normalize_transcript(transcript):
            issues.append(f"{record.sample_id}: transcript is empty.")
        ideal_info = sf.info(record.ideal_audio)
        participant_info = sf.info(record.participant_audio)
        if ideal_info.channels != 1 or ideal_info.samplerate != 16000:
            issues.append(f"{record.sample_id}: ideal WAV is not mono 16 kHz.")
        if participant_info.channels != 1 or participant_info.samplerate != 16000:
            issues.append(f"{record.sample_id}: participant WAV is not mono 16 kHz.")
        if record.start_sec is not None and record.end_sec is not None:
            if record.end_sec > participant_info.duration:
                issues.append(f"{record.sample_id}: flaw label exceeds participant audio duration.")
        if record.split:
            text_splits.setdefault(record.text_id or record.sample_id, set()).add(record.split)
        transcript_words = normalize_transcript(transcript)
        for role, audio_path in (
            ("ideal", record.ideal_audio),
            ("participant", record.participant_audio),
        ):
            alignment_path = (
                DATASET_ROOT / "alignments" / f"{record.sample_id}_{role}.json"
            )
            if alignment_path.is_file():
                try:
                    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
                    if normalize_transcript(alignment["transcript"]) != transcript_words:
                        issues.append(
                            f"{record.sample_id} {role}: alignment transcript does not match manifest."
                        )
                    validate_alignment(
                        alignment["words"],
                        transcript,
                        sf.info(audio_path).duration,
                    )
                except (OSError, KeyError, TypeError, ValueError, AlignmentError) as exc:
                    issues.append(f"{record.sample_id} {role}: invalid alignment: {exc}")

    for text_id, splits in text_splits.items():
        if len(splits) > 1:
            issues.append(
                f"{text_id}: variants of the same text leak across splits: {sorted(splits)}."
            )

    labels_path = DATASET_ROOT / "labels" / "flaw_regions.csv"
    if labels_path.exists():
        valid_ids = {record.sample_id for record in records}
        durations = {
            record.sample_id: sf.info(record.participant_audio).duration
            for record in records
        }
        with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = row.get("sample_id")
                if sample_id not in valid_ids:
                    issues.append(f"Label references unknown sample_id {sample_id!r}.")
                    continue
                try:
                    start = float(row["start_sec"])
                    end = float(row["end_sec"])
                    severity = float(row["severity"])
                    if start < 0 or end <= start or end > durations[sample_id]:
                        issues.append(f"{sample_id}: flaw label has invalid time bounds.")
                    if not 0 <= severity <= 4:
                        issues.append(f"{sample_id}: flaw label severity must be between 0 and 4.")
                    alignment_path = (
                        DATASET_ROOT / "alignments" / f"{sample_id}_participant.json"
                    )
                    if alignment_path.is_file():
                        alignment = json.loads(
                            alignment_path.read_text(encoding="utf-8")
                        )
                        if not any(
                            float(word["start"]) < end and float(word["end"]) > start
                            for word in alignment.get("words", [])
                        ):
                            issues.append(
                                f"{sample_id}: flaw label does not overlap any aligned word."
                            )
                except (KeyError, TypeError, ValueError):
                    issues.append(f"{sample_id}: flaw label has invalid numeric fields.")

    if issues:
        print("Dataset validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"Dataset valid: {len(records)} paired records; label paths and split groups checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate_dataset())
