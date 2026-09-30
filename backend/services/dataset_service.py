from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.config import DATASET_ROOT

REQUIRED_COLUMNS = {
    "sample_id",
    "text_id",
    "speaker_id",
    "transcript_file",
    "ideal_audio",
    "participant_audio",
    "flaw_type",
    "severity",
    "source_license",
    "split",
}
@dataclass
class DatasetRecord:
    sample_id: str
    speaker_id: str
    transcript_file: str
    ideal_audio: str
    participant_audio: str
    flaw_type: str
    severity: float
    baseline_id: str
    text_id: str = ""
    recording_id: str = ""
    condition: str = "flawed"
    start_sec: float | None = None
    end_sec: float | None = None
    source_license: str = ""
    source_url: str = ""
    split: str = ""
    notes: str = ""
    labels_path: str | None = None
    recording_conditions: str = ""

    def to_dict(self) -> dict:
        values = asdict(self)
        for field in ("transcript_file", "ideal_audio", "participant_audio", "labels_path"):
            value = values[field]
            if value:
                try:
                    values[field] = Path(value).resolve().relative_to(
                        DATASET_ROOT.resolve()
                    ).as_posix()
                except ValueError:
                    values[field] = Path(value).name
        return values


def _resolve_dataset_path(path_value: str, root: Path) -> Path:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Dataset path escapes its root directory: {path_value}") from exc
    return resolved


def load_manifest(
    manifest_path: Path | None = None, dataset_root: Path | None = None
) -> list[DatasetRecord]:
    root = (dataset_root or DATASET_ROOT).resolve()
    target = manifest_path or root / "manifest.csv"
    if not target.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {target}")

    records: list[DatasetRecord] = []
    seen_ids: set[str] = set()
    with target.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if not reader.fieldnames:
            raise ValueError("Dataset manifest is missing a header row.")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset manifest is missing required columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"Manifest row {line_number}: sample_id is required.")
            if any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in sample_id
            ):
                raise ValueError(f"Manifest row {line_number}: invalid sample_id {sample_id!r}.")
            if sample_id in seen_ids:
                raise ValueError(f"Manifest row {line_number}: duplicate sample_id {sample_id!r}.")
            seen_ids.add(sample_id)
            try:
                severity = float(row["severity"])
                start_value = (row.get("start_sec") or "").strip()
                end_value = (row.get("end_sec") or "").strip()
                start = float(start_value) if start_value else None
                end = float(end_value) if end_value else None
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Manifest row {line_number}: severity/times must be numeric.") from exc
            if not 0 <= severity <= 4:
                raise ValueError(f"Manifest row {line_number}: severity must be between 0 and 4.")
            if not (row.get("source_license") or "").strip():
                raise ValueError(f"Manifest row {line_number}: source_license is required.")
            if not (row.get("split") or "").strip():
                raise ValueError(f"Manifest row {line_number}: split is required.")
            if (start is None) != (end is None) or (
                start is not None and (start < 0 or end is None or end <= start)
            ):
                raise ValueError(f"Manifest row {line_number}: invalid flaw time bounds.")

            paths = {}
            for field in ("transcript_file", "ideal_audio", "participant_audio"):
                path_value = (row.get(field) or "").strip()
                if not path_value:
                    raise ValueError(f"Manifest row {line_number}: {field} is required.")
                path = _resolve_dataset_path(path_value, root)
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Manifest row {line_number}: {field} file not found: {path}"
                    )
                if path.suffix.lower() not in {".wav", ".mp3", ".m4a"} and field != "transcript_file":
                    raise ValueError(
                        f"Manifest row {line_number}: unsupported audio extension for {field}."
                    )
                paths[field] = str(path)

            labels = (row.get("labels_path") or "").strip()
            if labels:
                label_path = _resolve_dataset_path(labels, root)
                if not label_path.is_file():
                    raise FileNotFoundError(
                        f"Manifest row {line_number}: labels_path not found: {label_path}"
                    )
                labels = str(label_path)

            records.append(
                DatasetRecord(
                    sample_id=sample_id,
                    baseline_id=(row.get("baseline_id") or sample_id).strip(),
                    speaker_id=(row.get("speaker_id") or "unknown").strip(),
                    transcript_file=paths["transcript_file"],
                    ideal_audio=paths["ideal_audio"],
                    participant_audio=paths["participant_audio"],
                    flaw_type=(row.get("flaw_type") or "general").strip(),
                    severity=severity,
                    text_id=(row.get("text_id") or "").strip(),
                    recording_id=(row.get("recording_id") or "").strip(),
                    condition=(row.get("condition") or "flawed").strip(),
                    start_sec=start,
                    end_sec=end,
                    source_license=(row.get("source_license") or "").strip(),
                    source_url=(row.get("source_url") or "").strip(),
                    split=(row.get("split") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                    labels_path=labels or None,
                    recording_conditions=(row.get("recording_conditions") or "").strip(),
                )
            )
    return records


def get_record_by_id(
    sample_id: str, manifest_path: Path | None = None
) -> DatasetRecord | None:
    for record in load_manifest(manifest_path):
        if record.sample_id == sample_id or record.baseline_id == sample_id:
            return record
    return None
