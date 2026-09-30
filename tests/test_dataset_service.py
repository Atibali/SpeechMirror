import csv
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from backend.services.dataset_service import load_manifest


def write_silence(path: Path) -> None:
    samples = np.zeros(16_000, dtype=np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(samples.tobytes())


class DatasetServiceTests(unittest.TestCase):
    def test_manifest_resolves_files_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "texts").mkdir()
            (root / "audio").mkdir()
            (root / "labels").mkdir()
            (root / "texts" / "one.txt").write_text("hello world", encoding="utf-8")
            (root / "labels" / "labels.csv").write_text("sample_id\none\n", encoding="utf-8")
            write_silence(root / "audio" / "ideal.wav")
            write_silence(root / "audio" / "participant.wav")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
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
                        "labels_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "one",
                        "text_id": "text-one",
                        "speaker_id": "speaker",
                        "transcript_file": "texts/one.txt",
                        "ideal_audio": "audio/ideal.wav",
                        "participant_audio": "audio/participant.wav",
                        "flaw_type": "PACE_FAST",
                        "severity": "2",
                        "source_license": "test-only",
                        "split": "train",
                        "labels_path": "labels/labels.csv",
                    }
                )
            records = load_manifest(manifest, root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].baseline_id, "one")
            self.assertEqual(Path(records[0].ideal_audio).parent, root / "audio")

    def test_manifest_rejects_paths_outside_dataset_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "texts").mkdir()
            (root / "audio").mkdir()
            (root / "texts" / "one.txt").write_text("hello", encoding="utf-8")
            write_silence(root / "audio" / "participant.wav")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "sample_id,text_id,speaker_id,transcript_file,ideal_audio,participant_audio,flaw_type,severity,source_license,split\n"
                "one,text-one,speaker,texts/one.txt,../secret.wav,audio/participant.wav,PACE_FAST,1,test-only,train\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "escapes its root"):
                load_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
