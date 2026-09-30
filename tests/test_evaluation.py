import unittest

import numpy as np
import pandas as pd

from speechmirror.evaluation import compare_aligned_features, load_pipeline_config
import hashlib
import json


def make_features(duration: float, flat_pitch: bool = False) -> pd.DataFrame:
    times = np.arange(0.0, duration, 0.02)
    pitch = np.full_like(times, 120.0)
    if not flat_pitch:
        pitch += 35 * np.sin(times * 9)
    return pd.DataFrame(
        {
            "time": times,
            "rms": np.full_like(times, 0.08),
            "energy_db": np.full_like(times, -22.0),
            "pitch_hz": pitch,
            "voiced": np.ones_like(times, dtype=bool),
            "mfcc_1": np.sin(times),
            "mfcc_2": np.cos(times),
            "centroid": np.full_like(times, 1500.0),
            "flatness": np.full_like(times, 0.02),
            "bandwidth": np.full_like(times, 800.0),
            "zero_crossing_rate": np.full_like(times, 0.1),
            "speech_activity": np.ones_like(times, dtype=bool),
        }
    )


class EvaluationTests(unittest.TestCase):
    def test_scores_are_deterministic_and_report_feature_evidence(self):
        baseline_alignment = [
            {"word": "Speak", "start": 0.1, "end": 0.5},
            {"word": "with", "start": 0.7, "end": 1.1},
            {"word": "purpose.", "start": 1.3, "end": 1.7},
        ]
        participant_alignment = [
            {"word": "Speak", "start": 0.1, "end": 0.25},
            {"word": "with", "start": 0.45, "end": 0.6},
            {"word": "purpose.", "start": 0.8, "end": 0.95},
        ]
        config = load_pipeline_config()
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        arguments = dict(
            ideal_features=make_features(2),
            participant_features=make_features(2, flat_pitch=True),
            ideal_alignment=baseline_alignment,
            participant_alignment=participant_alignment,
            transcript="Speak with purpose.",
            baseline_transcript="Speak with purpose.",
            sample_id="test-1",
            input_hash="fixed-input-hash",
            alignment_method="whisperx_forced",
            configuration_hash=config_hash,
            alignment_model_version="test-aligner-v1",
        )
        result = compare_aligned_features(**arguments)
        repeated = compare_aligned_features(**arguments)
        self.assertEqual(result["result_hash"], repeated["result_hash"])
        self.assertEqual(result["overall_score"], repeated["overall_score"])
        self.assertEqual(len(result["word_evidence"]), 3)
        self.assertTrue(any(item["type"] == "PACE_FAST" for item in result["flaws"]))
        self.assertIn("evidence", result["flaws"][0])
        self.assertTrue(0 <= result["overall_score"] <= 100)

    def test_transcript_mismatch_is_rejected(self):
        alignment = [{"word": "hello", "start": 0.1, "end": 0.5}]
        with self.assertRaisesRegex(ValueError, "exactly match"):
            compare_aligned_features(
                ideal_features=make_features(1),
                participant_features=make_features(1),
                ideal_alignment=alignment,
                participant_alignment=alignment,
                transcript="hello there",
                baseline_transcript="hello",
                sample_id="test-2",
                input_hash="input",
                alignment_method="estimate",
                configuration_hash="config",
            )


if __name__ == "__main__":
    unittest.main()
