from pathlib import Path

from speechmirror.audio import generate_demo_dataset


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent.parent / "data"
    generate_demo_dataset(output_dir)
    print(f"Demo dataset created in {output_dir}")
