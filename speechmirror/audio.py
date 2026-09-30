from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _signal_envelope(samples: int, attack: float = 0.02, release: float = 0.1) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples, endpoint=False)
    env = np.minimum(t / max(attack, 1e-6), 1.0)
    env *= np.minimum((1.0 - t) / max(release, 1e-6), 1.0)
    return np.clip(env, 0.0, 1.0)


def synthesize_speech(
    transcript: str,
    sample_rate: int = 16_000,
    duration_seconds: float = 7.0,
    base_pitch_hz: float = 150.0,
    energy: float = 0.75,
    flaw_region: tuple[float, float] | None = None,
    speed_multiplier: float = 1.0,
) -> np.ndarray:
    words = [word for word in transcript.split() if word]
    if not words:
        raise ValueError("Transcript must contain at least one word.")

    total_samples = max(1, int(sample_rate * duration_seconds))
    t = np.arange(total_samples, dtype=np.float64) / sample_rate

    base = 1.0 + 0.12 * np.sin(2.0 * np.pi * 0.5 * t)
    pitch_curve = np.full(total_samples, base_pitch_hz, dtype=np.float64)
    if flaw_region is not None:
        start, end = flaw_region
        mask = (t >= start) & (t <= end)
        pitch_curve[mask] *= 0.72
        base[mask] *= 1.25

    if speed_multiplier != 1.0:
        modulation = 1.0 + (speed_multiplier - 1.0) * 0.35
        pitch_curve *= modulation

    phase = np.cumsum(2.0 * np.pi * pitch_curve / sample_rate)
    signal = np.sin(phase)
    signal += 0.35 * np.sin(2.0 * phase)
    signal *= base * energy
    signal *= _signal_envelope(total_samples)

    if flaw_region is not None:
        start, end = flaw_region
        signal[(t >= start) & (t <= end)] *= 0.8
    return signal.astype(np.float32)


def save_audio(path: str | Path, signal: np.ndarray, sample_rate: int = 16_000) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), signal, sample_rate)


def generate_demo_dataset(output_dir: str | Path) -> dict:
    root = Path(output_dir)
    demo_dir = root / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    transcript = (
        "Welcome to the workshop. Speak with steady pacing, clear articulation, and confident delivery."
    )
    ideal_signal = synthesize_speech(
        transcript,
        sample_rate=16_000,
        duration_seconds=7.0,
        base_pitch_hz=150.0,
        energy=0.68,
    )
    flawed_signal = synthesize_speech(
        transcript,
        sample_rate=16_000,
        duration_seconds=7.0,
        base_pitch_hz=150.0,
        energy=0.92,
        flaw_region=(2.2, 4.8),
        speed_multiplier=1.2,
    )

    ideal_path = demo_dir / "ideal.wav"
    flawed_path = demo_dir / "participant.wav"
    save_audio(ideal_path, ideal_signal, sample_rate=16_000)
    save_audio(flawed_path, flawed_signal, sample_rate=16_000)

    manifest = {
        "pair_id": "demo-pair-001",
        "speaker_id": "speaker-01",
        "transcript": transcript,
        "ideal_audio": str(ideal_path),
        "participant_audio": str(flawed_path),
        "labels": [
            {"type": "pitch_drop", "start": 2.2, "end": 4.8, "severity": 0.82},
            {"type": "tempo_spike", "start": 2.6, "end": 4.5, "severity": 0.74},
        ],
    }

    manifest_path = root / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
