from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
TEMP_ROOT = PROJECT_ROOT / "tmp"
RESULTS_ROOT = TEMP_ROOT / "results"
CACHE_ROOT = TEMP_ROOT / "cache"
DATABASE_PATH = TEMP_ROOT / "speechmirror.sqlite3"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def ensure_directories() -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
