from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import DATASET_ROOT
from backend.services.alignment import align_transcript_to_audio
from backend.services.dataset_service import load_manifest
from speechmirror.features import SAMPLE_RATE, load_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute word-level alignments for a dataset.")
    parser.add_argument("--method", choices=["whisperx", "estimate"], default="whisperx")
    args = parser.parse_args()

    output_dir = DATASET_ROOT / "alignments"
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in load_manifest(DATASET_ROOT / "manifest.csv"):
        transcript = Path(record.transcript_file).read_text(encoding="utf-8").strip()
        for role, path in (
            ("ideal", record.ideal_audio),
            ("participant", record.participant_audio),
        ):
            signal, sr = load_audio(path, sample_rate=SAMPLE_RATE)
            words, actual_method = align_transcript_to_audio(
                signal, sr, transcript, method=args.method
            )
            target = output_dir / f"{record.sample_id}_{role}.json"
            target.write_text(
                json.dumps(
                    {
                        "sample_id": record.sample_id,
                        "role": role,
                        "transcript": transcript,
                        "alignment_method": actual_method,
                        "words": words,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"Wrote {target.relative_to(ROOT)} ({len(words)} aligned words)")


if __name__ == "__main__":
    main()
