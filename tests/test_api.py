import csv
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import backend.config as config
import backend.main as api
import backend.services.audio_service as audio_service
import backend.services.dataset_service as dataset_service
from backend.services.alignment import AlignmentUnavailable


def write_tone(path: Path) -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = (np.sin(2 * np.pi * 180 * time) * 0.25 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "texts").mkdir(parents=True)
        (self.dataset / "audio").mkdir()
        (self.dataset / "texts" / "sample.txt").write_text("Speak with purpose.", encoding="utf-8")
        write_tone(self.dataset / "audio" / "ideal.wav")
        write_tone(self.dataset / "audio" / "participant.wav")
        with (self.dataset / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
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
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "sample",
                    "text_id": "text-1",
                    "speaker_id": "speaker",
                    "transcript_file": "texts/sample.txt",
                    "ideal_audio": "audio/ideal.wav",
                    "participant_audio": "audio/participant.wav",
                    "flaw_type": "PACE_FAST",
                    "severity": 1,
                    "source_license": "test-only",
                    "split": "train",
                }
            )

        self.old = {
            "config_dataset": config.DATASET_ROOT,
            "config_temp": config.TEMP_ROOT,
            "config_cache": config.CACHE_ROOT,
            "config_results": config.RESULTS_ROOT,
            "config_db": config.DATABASE_PATH,
            "api_dataset": api.DATASET_ROOT,
            "api_temp": api.TEMP_ROOT,
            "api_cache": audio_service.CACHE_ROOT,
            "api_results": api.RESULTS_ROOT,
            "api_db": api.DATABASE_PATH,
            "service_dataset": dataset_service.DATASET_ROOT,
        }
        config.DATASET_ROOT = self.dataset
        config.TEMP_ROOT = self.root / "tmp"
        config.CACHE_ROOT = config.TEMP_ROOT / "cache"
        config.RESULTS_ROOT = config.TEMP_ROOT / "results"
        config.DATABASE_PATH = config.TEMP_ROOT / "speechmirror.sqlite3"
        config.ensure_directories()
        api.DATASET_ROOT = config.DATASET_ROOT
        api.TEMP_ROOT = config.TEMP_ROOT
        api.CACHE_ROOT = config.CACHE_ROOT
        api.RESULTS_ROOT = config.RESULTS_ROOT
        api.DATABASE_PATH = config.DATABASE_PATH
        audio_service.CACHE_ROOT = config.CACHE_ROOT
        dataset_service.DATASET_ROOT = config.DATASET_ROOT
        api._initialize_database()
        self.client = TestClient(api.app)

    def tearDown(self):
        self.client.close()
        config.DATASET_ROOT = self.old["config_dataset"]
        config.TEMP_ROOT = self.old["config_temp"]
        config.CACHE_ROOT = self.old["config_cache"]
        config.RESULTS_ROOT = self.old["config_results"]
        config.DATABASE_PATH = self.old["config_db"]
        api.DATASET_ROOT = self.old["api_dataset"]
        api.TEMP_ROOT = self.old["api_temp"]
        api.RESULTS_ROOT = self.old["api_results"]
        audio_service.CACHE_ROOT = self.old["api_cache"]
        api.DATABASE_PATH = self.old["api_db"]
        dataset_service.DATASET_ROOT = self.old["service_dataset"]
        self.temporary.cleanup()

    def test_uploaded_analysis_is_saved_without_retaining_uploaded_audio(self):
        participant_bytes = (self.dataset / "audio" / "participant.wav").read_bytes()
        response = self.client.post(
            "/api/analyze",
            data={
                "baseline_id": "sample",
                "transcript": "Speak with purpose.",
                "alignment_mode": "estimate",
            },
            files={"audio": ("participant.wav", participant_bytes, "audio/wav")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["alignment_method"], "uniform_estimate")
        self.assertEqual(result["sample_id"], "sample")
        self.assertEqual(
            self.client.get(f"/api/result/{result['analysis_id']}").status_code, 200
        )
        self.assertEqual(list(config.TEMP_ROOT.glob("speechmirror-*")), [])

    def test_transcript_mismatch_is_rejected_before_processing(self):
        response = self.client.post(
            "/api/analyze",
            data={
                "baseline_id": "sample",
                "transcript": "This is a different sentence.",
                "alignment_mode": "estimate",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_missing_forced_alignment_model_returns_actionable_service_error(self):
        with patch(
            "backend.services.alignment._load_whisperx_model",
            side_effect=AlignmentUnavailable("WhisperX model unavailable for test."),
        ):
            response = self.client.post(
                "/api/analyze",
                data={
                    "baseline_id": "sample",
                    "transcript": "Speak with purpose.",
                    "alignment_mode": "whisperx",
                },
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("WhisperX model unavailable", response.json()["detail"])

    def test_dataset_audio_feature_and_alignment_endpoints(self):
        self.assertEqual(self.client.get("/api/health").json()["dataset_entries"], 1)
        dataset_response = self.client.get("/api/dataset/sample")
        self.assertEqual(dataset_response.status_code, 200)
        self.assertEqual(dataset_response.json()["ideal_audio"], "audio/ideal.wav")
        self.assertNotIn(str(self.root), dataset_response.text)
        self.assertEqual(self.client.get("/api/audio/sample?role=ideal").status_code, 200)
        feature_response = self.client.post(
            "/api/features",
            data={"sample_id": "sample", "role": "ideal"},
        )
        self.assertEqual(feature_response.status_code, 200, feature_response.text)
        self.assertTrue(feature_response.json()["features"])
        alignment_response = self.client.post(
            "/api/align",
            data={"transcript": "Speak with purpose.", "alignment_mode": "estimate"},
            files={
                "audio": (
                    "ideal.wav",
                    (self.dataset / "audio" / "ideal.wav").read_bytes(),
                    "audio/wav",
                )
            },
        )
        self.assertEqual(alignment_response.status_code, 200, alignment_response.text)
        self.assertEqual(len(alignment_response.json()["words"]), 3)


if __name__ == "__main__":
    unittest.main()
