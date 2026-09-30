from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speechmirror.audio import generate_demo_dataset


if __name__ == "__main__":
    output_dir = ROOT / "data"
    generate_demo_dataset(output_dir)
    print(f"Demo dataset created in {output_dir}")
