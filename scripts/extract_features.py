from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import DATASET_ROOT
from backend.services.audio_service import feature_records
from backend.services.dataset_service import load_manifest


def main() -> None:
    output_dir = DATASET_ROOT / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(DATASET_ROOT / "manifest.csv")
    for record in records:
        for role, path in (
            ("ideal", record.ideal_audio),
            ("participant", record.participant_audio),
        ):
            rows = feature_records(path)
            target = output_dir / f"{record.sample_id}_{role}.parquet"
            import pandas as pd

            pd.DataFrame(rows).to_parquet(target, index=False)
            print(f"Wrote {target.relative_to(ROOT)} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
