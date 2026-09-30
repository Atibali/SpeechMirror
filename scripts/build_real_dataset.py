from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dataset_service import load_manifest
from speechmirror.audio import save_audio, synthesize_speech
from speechmirror.features import SAMPLE_RATE, load_audio

DATASET_ROOT = ROOT / "dataset"
TEXTS_ROOT = DATASET_ROOT / "texts"
AUDIO_ROOT = DATASET_ROOT / "audio"
LABELS_ROOT = DATASET_ROOT / "labels"

MANIFEST_FIELDS = [
    "sample_id",
    "baseline_id",
    "text_id",
    "speaker_id",
    "recording_id",
    "transcript_file",
    "ideal_audio",
    "participant_audio",
    "condition",
    "flaw_type",
    "severity",
    "start_sec",
    "end_sec",
    "source_license",
    "source_url",
    "split",
    "recording_conditions",
    "notes",
    "labels_path",
    "ideal_original_sha256",
    "participant_original_sha256",
]
SOURCE_REQUIRED = {
    "sample_id",
    "speaker_id",
    "transcript",
    "ideal_audio",
    "participant_audio",
    "flaw_type",
    "severity",
    "source_license",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else base_dir / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Source file does not exist: {resolved}")
    return resolved


def _normalized_copy(source: Path, target: Path) -> None:
    signal, sample_rate = load_audio(source, sample_rate=SAMPLE_RATE)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, signal, sample_rate, subtype="PCM_16")


def ingest_source_manifest(
    source_manifest: Path, labels_manifest: Path | None = None
) -> None:
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest does not exist: {source_manifest}")

    with source_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Source manifest must include a CSV header.")
        missing = SOURCE_REQUIRED - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
        source_rows = list(reader)

    if not source_rows:
        raise ValueError("Source manifest contains no samples.")

    incoming_ids: set[str] = set()
    for index, source in enumerate(source_rows, start=2):
        sample_id = (source.get("sample_id") or "").strip()
        if not sample_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in sample_id
        ):
            raise ValueError(f"Invalid sample_id in source manifest row {index}.")
        if sample_id in incoming_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        incoming_ids.add(sample_id)

    existing_manifest = DATASET_ROOT / "manifest.csv"
    existing_records = []
    existing_rows = []
    existing_label_rows = []
    if existing_manifest.exists():
        existing_records = load_manifest(existing_manifest)
        existing_ids = {record.sample_id for record in existing_records}
        conflicts = existing_ids & incoming_ids
        if conflicts:
            raise FileExistsError(
                f"Dataset already contains sample_id values: {sorted(conflicts)}. "
                "Choose new IDs rather than silently overwriting existing records."
            )
        with existing_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
        existing_labels_path = LABELS_ROOT / "flaw_regions.csv"
        if existing_labels_path.is_file():
            with existing_labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
                existing_label_rows = list(csv.DictReader(handle))
    for sample_id in incoming_ids:
        existing_outputs = [
            TEXTS_ROOT / f"{sample_id}.txt",
            AUDIO_ROOT / f"{sample_id}_ideal.wav",
            AUDIO_ROOT / f"{sample_id}_participant.wav",
        ]
        if any(path.exists() for path in existing_outputs):
            raise FileExistsError(
                f"Dataset files for {sample_id!r} already exist; choose a new sample ID."
            )

    text_groups: dict[str, list[dict]] = {}
    for source in source_rows:
        transcript_key = " ".join((source.get("transcript") or "").casefold().split())
        if not transcript_key:
            raise ValueError("Every source manifest row must include its exact transcript.")
        text_id = (source.get("text_id") or "").strip() or hashlib.sha256(
            transcript_key.encode("utf-8")
        ).hexdigest()
        source["_resolved_text_id"] = text_id
        text_groups.setdefault(text_id, []).append(source)
    explicit_splits: dict[str, set[str]] = {
        text_id: {row["split"].strip() for row in rows if row.get("split", "").strip()}
        for text_id, rows in text_groups.items()
    }
    for record in existing_records:
        if record.text_id in explicit_splits:
            explicit_splits[record.text_id].add(record.split)
    if any(len(splits) > 1 for splits in explicit_splits.values()):
        raise ValueError("All variants of one text_id must use the same dataset split.")
    unassigned = sorted(
        (text_id for text_id, splits in explicit_splits.items() if not splits),
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    generated_splits: dict[str, str] = {}
    if len(unassigned) >= 3:
        test_count = max(1, round(len(unassigned) * 0.2))
        validation_count = max(1, round(len(unassigned) * 0.1))
        for position, text_id in enumerate(unassigned):
            if position < test_count:
                generated_splits[text_id] = "test"
            elif position < test_count + validation_count:
                generated_splits[text_id] = "validation"
            else:
                generated_splits[text_id] = "train"
    else:
        generated_splits = {text_id: "train" for text_id in unassigned}
    for text_id, rows in text_groups.items():
        assigned = next(iter(explicit_splits[text_id]), generated_splits.get(text_id, "train"))
        for row in rows:
            row["split"] = assigned

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    TEXTS_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    LABELS_ROOT.mkdir(parents=True, exist_ok=True)
    source_base = source_manifest.resolve().parent
    output_rows = existing_rows.copy()
    label_rows = existing_label_rows.copy()
    seen_ids = {record.sample_id for record in existing_records}

    for index, source in enumerate(source_rows):
        sample_id = (source.get("sample_id") or "").strip()
        if not sample_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in sample_id):
            raise ValueError(f"Invalid sample_id in source manifest row {index + 2}.")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        transcript = " ".join((source.get("transcript") or "").split())
        if not transcript:
            raise ValueError(f"Transcript is empty for sample {sample_id}.")
        if not (source.get("source_license") or "").strip():
            raise ValueError(f"source_license/provenance is required for sample {sample_id}.")

        ideal_source = _source_path(source_base, source["ideal_audio"])
        participant_source = _source_path(source_base, source["participant_audio"])
        ideal_hash = _sha256(ideal_source)
        participant_hash = _sha256(participant_source)
        transcript_path = TEXTS_ROOT / f"{sample_id}.txt"
        ideal_path = AUDIO_ROOT / f"{sample_id}_ideal.wav"
        participant_path = AUDIO_ROOT / f"{sample_id}_participant.wav"

        transcript_path.write_text(transcript + "\n", encoding="utf-8")
        _normalized_copy(ideal_source, ideal_path)
        _normalized_copy(participant_source, participant_path)

        start_value = (source.get("start_sec") or "").strip()
        end_value = (source.get("end_sec") or "").strip()
        start = float(start_value) if start_value else None
        end = float(end_value) if end_value else None
        if (start is None) != (end is None) or (
            start is not None and (start < 0 or end is None or end <= start)
        ):
            raise ValueError(f"Invalid flaw time bounds for sample {sample_id}.")
        severity = float(source["severity"])
        if not 0 <= severity <= 4:
            raise ValueError(f"Severity for sample {sample_id} must be between 0 and 4.")
        if end is not None and end > sf.info(participant_path).duration:
            raise ValueError(f"Flaw label exceeds participant audio duration for {sample_id}.")

        text_id = source["_resolved_text_id"]
        output_rows.append(
            {
                "sample_id": sample_id,
                "baseline_id": (source.get("baseline_id") or sample_id).strip(),
                "text_id": text_id,
                "speaker_id": (source.get("speaker_id") or "unknown").strip(),
                "recording_id": (source.get("recording_id") or sample_id).strip(),
                "transcript_file": f"texts/{sample_id}.txt",
                "ideal_audio": f"audio/{sample_id}_ideal.wav",
                "participant_audio": f"audio/{sample_id}_participant.wav",
                "condition": (source.get("condition") or "flawed").strip(),
                "flaw_type": (source.get("flaw_type") or "general").strip(),
                "severity": severity,
                "start_sec": start if start is not None else "",
                "end_sec": end if end is not None else "",
                "source_license": source["source_license"].strip(),
                "source_url": (source.get("source_url") or "").strip(),
                "split": (source.get("split") or "train").strip(),
                "recording_conditions": (source.get("recording_conditions") or "").strip(),
                "notes": (source.get("notes") or "").strip(),
                "labels_path": "labels/flaw_regions.csv",
                "ideal_original_sha256": ideal_hash,
                "participant_original_sha256": participant_hash,
            }
        )
        if start is not None:
            label_rows.append(
                {
                    "sample_id": sample_id,
                    "flaw_id": f"{sample_id}-F01",
                    "start_sec": start,
                    "end_sec": end,
                    "severity": severity,
                    "feature_expected": (source.get("feature_expected") or source["flaw_type"]).strip(),
                    "notes": (source.get("notes") or "").strip(),
                }
            )

    if labels_manifest is not None:
        if not labels_manifest.is_file():
            raise FileNotFoundError(f"Additional label manifest does not exist: {labels_manifest}")
        with labels_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_labels = {
                "sample_id",
                "flaw_id",
                "start_sec",
                "end_sec",
                "severity",
                "feature_expected",
            }
            if not reader.fieldnames or not required_labels.issubset(reader.fieldnames):
                raise ValueError(
                    f"Label manifest must contain columns: {sorted(required_labels)}"
                )
            for row in reader:
                sample_id = row["sample_id"].strip()
                if sample_id not in seen_ids:
                    raise ValueError(f"Label manifest references unknown sample_id {sample_id!r}.")
                start = float(row["start_sec"])
                end = float(row["end_sec"])
                severity = float(row["severity"])
                if start < 0 or end <= start or not 0 <= severity <= 4:
                    raise ValueError(f"Invalid label interval/severity for sample {sample_id}.")
                target_audio = AUDIO_ROOT / f"{sample_id}_participant.wav"
                if end > sf.info(target_audio).duration:
                    raise ValueError(f"Label interval exceeds participant audio for {sample_id}.")
                label_rows.append(
                    {
                        "sample_id": sample_id,
                        "flaw_id": row["flaw_id"].strip(),
                        "start_sec": start,
                        "end_sec": end,
                        "severity": severity,
                        "feature_expected": row["feature_expected"].strip(),
                        "notes": (row.get("notes") or "").strip(),
                    }
                )

    _write_outputs(output_rows, label_rows)
    records = load_manifest(DATASET_ROOT / "manifest.csv")
    print(f"Ingested {len(records)} real paired recordings into {DATASET_ROOT}")
    print("Audio was resampled to mono 16 kHz PCM WAV; original audio hashes were recorded.")


def _write_outputs(rows: list[dict], labels: list[dict]) -> None:
    with (DATASET_ROOT / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with (LABELS_ROOT / "flaw_regions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "flaw_id",
                "start_sec",
                "end_sec",
                "severity",
                "feature_expected",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(labels)


def build_synthetic_demo() -> None:
    manifest_path = DATASET_ROOT / "manifest.csv"
    if manifest_path.exists():
        existing_records = load_manifest(manifest_path)
        if any(
            not record.source_license.startswith("Synthetic signal;")
            for record in existing_records
        ):
            raise FileExistsError(
                "Refusing to overwrite a dataset containing non-synthetic recordings."
            )
    TEXTS_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    LABELS_ROOT.mkdir(parents=True, exist_ok=True)
    entries = [
        (
            "demo_pace",
            "speaker_demo",
            "We can build a better future together with steady effort and clear purpose.",
            "PACE_FAST",
            (2.2, 4.8),
            1.8,
            1.2,
        ),
        (
            "demo_pitch",
            "speaker_demo",
            "The workshop emphasizes calm pacing, crisp articulation, and decisive speaking habits.",
            "PITCH_FLAT",
            (2.4, 5.2),
            1.6,
            1.05,
        ),
    ]
    rows = []
    labels = []
    for sample_id, speaker, transcript, flaw, region, severity, speed in entries:
        (TEXTS_ROOT / f"{sample_id}.txt").write_text(transcript + "\n", encoding="utf-8")
        ideal_path = AUDIO_ROOT / f"{sample_id}_ideal.wav"
        participant_path = AUDIO_ROOT / f"{sample_id}_participant.wav"
        save_audio(
            ideal_path,
            synthesize_speech(transcript, duration_seconds=7, energy=0.62),
            SAMPLE_RATE,
        )
        save_audio(
            participant_path,
            synthesize_speech(
                transcript,
                duration_seconds=7,
                energy=0.85,
                flaw_region=region,
                speed_multiplier=speed,
            ),
            SAMPLE_RATE,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "baseline_id": sample_id,
                "text_id": sample_id,
                "speaker_id": speaker,
                "recording_id": sample_id,
                "transcript_file": f"texts/{sample_id}.txt",
                "ideal_audio": f"audio/{sample_id}_ideal.wav",
                "participant_audio": f"audio/{sample_id}_participant.wav",
                "condition": "synthetic-demo",
                "flaw_type": flaw,
                "severity": severity,
                "start_sec": region[0],
                "end_sec": region[1],
                "source_license": "Synthetic signal; not human speech or a real dataset.",
                "source_url": "",
                "split": "demo",
                "recording_conditions": "Programmatically generated test tone.",
                "notes": "Synthetic pipeline smoke test only; do not use as benchmark data.",
                "labels_path": "labels/flaw_regions.csv",
                "ideal_original_sha256": _sha256(ideal_path),
                "participant_original_sha256": _sha256(participant_path),
            }
        )
        labels.append(
            {
                "sample_id": sample_id,
                "flaw_id": f"{sample_id}-F01",
                "start_sec": region[0],
                "end_sec": region[1],
                "severity": severity,
                "feature_expected": flaw,
                "notes": "Synthetic demonstration label only.",
            }
        )
    _write_outputs(rows, labels)
    print(
        "Generated a synthetic signal demo. It is NOT a real speech corpus; "
        "provide self-recorded/licensed recordings for evaluation."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest paired, licensed/self-recorded speech into the SpeechMirror dataset."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-manifest", type=Path)
    source.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Generate artificial test tones for a UI/pipeline smoke test only.",
    )
    parser.add_argument(
        "--labels-manifest",
        type=Path,
        help="Optional CSV with multiple flaw regions for each sample.",
    )
    args = parser.parse_args()
    if args.synthetic_demo:
        build_synthetic_demo()
    else:
        ingest_source_manifest(
            args.source_manifest.resolve(),
            args.labels_manifest.resolve() if args.labels_manifest else None,
        )


if __name__ == "__main__":
    main()
