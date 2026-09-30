import unittest

import numpy as np

from backend.services.alignment import AlignmentError, align_transcript_to_audio, validate_alignment


class AlignmentTests(unittest.TestCase):
    def test_estimate_is_explicit_and_covers_each_word(self):
        signal = np.zeros(16_000, dtype=np.float32)
        words, method = align_transcript_to_audio(
            signal, 16_000, "Speak with clear purpose.", method="estimate"
        )
        self.assertEqual(method, "uniform_estimate")
        self.assertEqual([word["word"] for word in words], ["Speak", "with", "clear", "purpose."])
        self.assertAlmostEqual(words[0]["start"], 0)
        self.assertAlmostEqual(words[-1]["end"], 1)

    def test_alignment_rejects_transcript_coverage_mismatch(self):
        with self.assertRaisesRegex(AlignmentError, "coverage mismatch"):
            validate_alignment([{"word": "hello", "start": 0, "end": 0.2}], "hello world", 1)

    def test_alignment_rejects_bad_timestamps(self):
        words = [
            {"word": "one", "start": 0.2, "end": 0.4},
            {"word": "two", "start": 0.1, "end": 0.5},
        ]
        with self.assertRaisesRegex(AlignmentError, "monotonically"):
            validate_alignment(words, "one two", 1)


if __name__ == "__main__":
    unittest.main()
