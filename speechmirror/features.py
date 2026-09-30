from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

SAMPLE_RATE = 16_000
FRAME_LENGTH = 1_024
FRAME_HOP = 320
PITCH_FRAME_LENGTH = 2_048
N_MFCC = 13


def load_audio(path: str | Path, sample_rate: int | None = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    signal, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    if signal.size == 0:
        raise ValueError(f"Audio file contains no samples: {path}")
    return signal.astype(np.float32), sr


def extract_feature_frame(
    signal: np.ndarray,
    sample_rate: int,
    frame_hop: int = FRAME_HOP,
) -> pd.DataFrame:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("Audio signal must be a non-empty mono array.")
    if sample_rate <= 0 or frame_hop <= 0:
        raise ValueError("Sample rate and frame hop must be positive.")

    frame_length = max(FRAME_LENGTH, frame_hop)
    rms = librosa.feature.rms(
        y=signal, frame_length=frame_length, hop_length=frame_hop
    )[0]
    db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=1.0)
    pitch = librosa.yin(
        signal,
        fmin=60,
        fmax=min(500, sample_rate / 2),
        sr=sample_rate,
        frame_length=max(PITCH_FRAME_LENGTH, frame_length),
        hop_length=frame_hop,
    )
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sample_rate, n_mfcc=N_MFCC, hop_length=frame_hop
    )
    centroid = librosa.feature.spectral_centroid(
        y=signal, sr=sample_rate, hop_length=frame_hop
    )[0]
    flatness = librosa.feature.spectral_flatness(y=signal, hop_length=frame_hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal, sr=sample_rate, hop_length=frame_hop
    )[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=signal, frame_length=frame_length, hop_length=frame_hop
    )[0]

    frame_count = min(
        len(rms),
        len(pitch),
        mfcc.shape[1],
        len(centroid),
        len(flatness),
        len(bandwidth),
        len(zero_crossing_rate),
    )
    times = np.arange(frame_count, dtype=np.float64) * frame_hop / sample_rate
    rms = rms[:frame_count]
    pitch = pitch[:frame_count]
    rms_threshold = max(
        0.008,
        min(float(np.percentile(rms, 20)) * 1.5, float(np.median(rms)) * 0.5),
    )
    speech_activity = rms > rms_threshold
    voiced = speech_activity & (pitch >= 60)
    pitch = np.where(voiced, pitch, np.nan)
    energy_db = db[:frame_count]

    return pd.DataFrame(
        {
            "time": times,
            "rms": rms,
            "energy_db": energy_db,
            "pitch_hz": pitch,
            "voiced": voiced,
            "mfcc_1": mfcc[0, :frame_count],
            "mfcc_2": mfcc[1, :frame_count],
            "mfcc_3": mfcc[2, :frame_count],
            "centroid": centroid[:frame_count],
            "flatness": flatness[:frame_count],
            "bandwidth": bandwidth[:frame_count],
            "zero_crossing_rate": zero_crossing_rate[:frame_count],
            "speech_activity": speech_activity,
        }
    )


def aggregate_word_features(
    features: pd.DataFrame, alignment: list[dict]
) -> list[dict]:
    if features.empty:
        return []

    valid_pitch = features["pitch_hz"].dropna()
    pitch_reference = float(valid_pitch.median()) if not valid_pitch.empty else 0.0
    energy_reference = float(features["energy_db"].median())
    rows: list[dict] = []

    for index, word in enumerate(alignment):
        start = float(word["start"])
        end = float(word["end"])
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            continue
        selected = features[(features["time"] >= start) & (features["time"] <= end)]
        if selected.empty:
            continue
        pitch_values = selected["pitch_hz"].dropna().to_numpy(dtype=np.float64)
        pitch_median = float(np.median(pitch_values)) if pitch_values.size else None
        pitch_relative = (
            float(12 * np.log2(pitch_median / pitch_reference))
            if pitch_median is not None and pitch_reference > 0
            else None
        )
        next_word_start = (
            float(alignment[index + 1]["start"])
            if index + 1 < len(alignment)
            else end
        )
        pause_after = max(0.0, next_word_start - end)
        duration = max(end - start, 1e-6)

        row = {
            "word": str(word["word"]),
            "start": start,
            "end": end,
            "confidence": word.get("confidence"),
            "duration": duration,
            "speech_rate": 1.0 / duration,
            "pause_after": pause_after,
            "energy_db": float(selected["energy_db"].median()),
            "energy_relative_db": float(selected["energy_db"].median() - energy_reference),
            "pitch_hz": pitch_median,
            "pitch_relative_semitones": pitch_relative,
            "pitch_range_semitones": (
                float(
                    12
                    * np.log2(
                        max(float(np.percentile(pitch_values, 90)), 1e-6)
                        / max(float(np.percentile(pitch_values, 10)), 1e-6)
                    )
                )
                if pitch_values.size > 1
                else 0.0
            ),
            "mfcc_1": float(selected["mfcc_1"].median()),
            "mfcc_2": float(selected["mfcc_2"].median()),
            "centroid": float(selected["centroid"].median()),
            "flatness": float(selected["flatness"].median()),
            "bandwidth": float(selected["bandwidth"].median()),
            "voiced_fraction": float(selected["voiced"].mean()),
            "speech_activity_fraction": float(selected["speech_activity"].mean()),
        }
        rows.append(row)
    return rows
