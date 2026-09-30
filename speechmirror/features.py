from __future__ import annotations

import numpy as np
import pandas as pd
import librosa


def load_audio(path: str, sample_rate: int | None = None):
    signal, sr = librosa.load(path, sr=sample_rate, mono=True, duration=None)
    return signal.astype(np.float32), sr


def _safe_median(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.nanmedian(values))


def extract_feature_frame(signal: np.ndarray, sample_rate: int, frame_hop: int = 512) -> pd.DataFrame:
    rms = librosa.feature.rms(y=signal, frame_length=2048, hop_length=frame_hop)[0]
    pitch_raw = librosa.yin(
        y=signal,
        fmin=60,
        fmax=350,
        sr=sample_rate,
        frame_length=2048,
        hop_length=frame_hop,
    )
    pitch = np.nan_to_num(pitch_raw, nan=0.0)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=5, hop_length=frame_hop)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, hop_length=frame_hop)[0]
    flatness = librosa.feature.spectral_flatness(y=signal, hop_length=frame_hop)[0]

    energy = np.clip(rms, 0.0, 1.0)
    activity = energy > 0.04
    speech_rate = np.full_like(energy, 0.0, dtype=np.float64)
    if activity.any():
        for idx in range(len(energy)):
            window = slice(max(0, idx - 4), min(len(energy), idx + 5))
            speech_rate[idx] = activity[window].mean()

    pause_rate = 1.0 - speech_rate
    times = np.arange(len(rms), dtype=np.float64) * frame_hop / sample_rate

    return pd.DataFrame(
        {
            "time": times,
            "energy": energy,
            "pitch": pitch,
            "mfcc_1": mfcc[0],
            "mfcc_2": mfcc[1],
            "centroid": centroid,
            "flatness": flatness,
            "speech_rate": speech_rate,
            "pause_rate": pause_rate,
        }
    )


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in ["energy", "pitch", "mfcc_1", "mfcc_2", "centroid", "flatness", "speech_rate", "pause_rate"]:
        values = normalized[column].to_numpy(dtype=np.float64)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        if std == 0.0:
            normalized[column] = 0.0
        else:
            normalized[column] = (values - mean) / std
    return normalized
