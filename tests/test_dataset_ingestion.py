import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import backend.services.dataset_service as dataset_service
import scripts.build_real_dataset as dataset_builder


def write_tone(path: Path, frequency: float) -> None:
    rate = 22_050
    time = np.arange(rate // 2, dtype=np.float32) / rate
    samples = (np.sin(2 * np.pi * frequency * time) * 0.2).astype(np.float32)
    sf.write(path, samples, rate)


class DatasetIngestionTests(unittest.TestCase):
    def test_ingestion_normalizes_audio_and_keeps_text_variants_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            write_tone(source_dir / "ideal.wav", 160)
            write_tone(source_dir / "fast.wav", 180)
            manifest_path = source_dir / "pairs.csv"
            fields = [
                "sample_id",
                "text_id",
                "speaker_id",
                "transcript",
                "ideal_audio",
                "participant_audio",
                "flaw_type",
                "severity",
                "start_sec",
                "end_sec",
                "source_license",
            ]
            rows = [
                {
                    "sample_id": sample_id,
                    "text_id": "same-text",
                    "speaker_id": "speaker-1",
                    "transcript": "Same exact words.",
                    "ideal_audio": "ideal.wav",
                    "participant_audio": "fast.wav",
                    "flaw_type": "PACE_FAST",
                    "severity": 2,
                    "start_sec": 0.1,
                    "end_sec": 0.3,
                    "source_license": "Recorded with speaker consent",
                }
                for sample_id in ("pair-a", "pair-b")
            ]
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            old_values = (
                dataset_builder.DATASET_ROOT,
                dataset_builder.TEXTS_ROOT,
                dataset_builder.AUDIO_ROOT,
                dataset_builder.LABELS_ROOT,
                dataset_service.DATASET_ROOT,
            )
            target_root = root / "output"
            try:
                dataset_builder.DATASET_ROOT = target_root
                dataset_builder.TEXTS_ROOT = target_root / "texts"
                dataset_builder.AUDIO_ROOT = target_root / "audio"
                dataset_builder.LABELS_ROOT = target_root / "labels"
                dataset_service.DATASET_ROOT = target_root
                dataset_builder.ingest_source_manifest(manifest_path)
                records = dataset_service.load_manifest(target_root / "manifest.csv")
                self.assertEqual(len(records), 2)
                self.assertEqual(records[0].split, records[1].split)
                self.assertEqual(
                    Path(records[0].transcript_file).read_text(encoding="utf-8").strip(),
                    "Same exact words.",
                )
                self.assertEqual(sf.info(records[0].ideal_audio).samplerate, 16_000)
                self.assertEqual(sf.info(records[0].ideal_audio).channels, 1)
                self.assertEqual(records[0].source_license, "Recorded with speaker consent")

                append_manifest = source_dir / "append.csv"
                with append_manifest.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow({**rows[0], "sample_id": "pair-c"})
                dataset_builder.ingest_source_manifest(append_manifest)
                records = dataset_service.load_manifest(target_root / "manifest.csv")
                self.assertEqual(len(records), 3)
                self.assertEqual({record.split for record in records}, {"train"})
                with self.assertRaises(FileExistsError):
                    dataset_builder.ingest_source_manifest(append_manifest)
            finally:
                (
                    dataset_builder.DATASET_ROOT,
                    dataset_builder.TEXTS_ROOT,
                    dataset_builder.AUDIO_ROOT,
                    dataset_builder.LABELS_ROOT,
                    dataset_service.DATASET_ROOT,
                ) = old_values


if __name__ == "__main__":
    unittest.main()
