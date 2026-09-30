# SpeechMirror

SpeechMirror is a local, evidence-first speech delivery analysis prototype. It compares a participant recording against an ideal recording of the same transcript, aligns both recordings at word level, extracts acoustic features, detects time-localized delivery changes, and reports rubric scores with numeric evidence.

The repository does **not** contain a genuine human speech corpus. The optional synthetic demo creates test tones, not speech, and must not be presented as real speech, an accuracy benchmark, or evidence of human performance. Ingest self-recorded or openly licensed speech and record the permission/provenance before using the system for real evaluation.

## Implemented pipeline

1. Ingest a manifest of ideal/participant audio pairs and exact transcripts; normalize audio to mono 16 kHz PCM WAV and record source hashes.
2. Validate audio paths, transcript metadata, temporal labels, dataset splits, and source provenance.
3. Align baseline and participant words with WhisperX forced alignment (optional model dependency). An explicitly labelled uniform estimate is provided for UI/pipeline smoke tests only.
4. Extract frame-level RMS/dB, F0, MFCC, spectral centroid, flatness, bandwidth, and zero-crossing features.
5. Aggregate measurements over aligned words; compare pace, pauses, speaker-relative pitch contour, energy, and spectral evidence.
6. Generate temporal flaw cards, dimension scores, causal explanations, recommended actions, and reproducibility hashes.
7. Display audio, word timing, feature overlays, score cards, evidence, and JSON/CSV exports in Streamlit.

## Requirements

- Python 3.10 or 3.11
- Windows, macOS, or Linux
- FFmpeg available on `PATH` for MP3/M4A decoding (WAV does not require FFmpeg)
- WhisperX and its compatible PyTorch/CUDA stack for genuine forced alignment. See [Forced alignment setup](#forced-alignment-setup).

## Windows setup

Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass`, then activate the environment again.

### Build a synthetic smoke-test dataset

This generates artificial tones solely to check that ingestion, UI, and feature code execute:

```powershell
python scripts\build_real_dataset.py --synthetic-demo
python scripts\validate_dataset.py
```

### Ingest genuine paired recordings

1. Record the same exact English transcript as an ideal and a flawed delivery. Prefer the same speaker, microphone, and room; change one main delivery behavior at a time.
2. Keep the recordings and a CSV manifest together. Start from [dataset/manifest.example.csv](dataset/manifest.example.csv). Paths in that source CSV are relative to the CSV file or absolute.
3. Replace the example values, including the exact transcript, audio paths, `source_license` (permission/license), flaw label, severity (0–4), and any known start/end times.
4. Ingest and validate:

```powershell
python scripts\build_real_dataset.py --source-manifest .\my-recordings\manifest.csv
python scripts\validate_dataset.py
```

The ingestion script stores normalized copies in `dataset/audio`, transcripts in `dataset/texts`, labels in `dataset/labels`, and a resolved catalog in `dataset/manifest.csv`. Original source audio SHA-256 hashes and provenance are retained in the catalog. Audio is not uploaded to a third-party service by this local pipeline. Ensure you have the required speaker consent and redistribution rights before storing or sharing source audio.

If one sample has multiple annotated flaw regions, supply a second label CSV with `sample_id,flaw_id,start_sec,end_sec,severity,feature_expected` and pass it using `--labels-manifest .\my-recordings\flaws.csv`. The CSV may contain several rows for the same sample.

Use multiple transcript groups to create meaningful train/validation/test splits. The importer groups all variants of a transcript into one split; with fewer than three transcript groups it assigns them to `train` because a blind test split would not be meaningful. Never tune thresholds on the held-out `test` split.

## Forced alignment setup

Forced alignment requires a transcript and a compatible acoustic alignment model. Install the appropriate PyTorch build for your machine first, then install WhisperX in the activated virtual environment:

```powershell
python -m pip install whisperx
```

WhisperX may download model weights on first use and has additional platform/model requirements. Check the WhisperX installation documentation and configure FFmpeg for non-WAV audio. Model files are not bundled in this repository. The API uses WhisperX by default and returns an explicit error if it is unavailable; it never labels proportional timing estimates as forced alignment.

The aligner uses CPU by default. To run it on a PyTorch-supported CUDA device, set `SPEECHMIRROR_DEVICE` before starting both app processes:

```powershell
$env:SPEECHMIRROR_DEVICE = "cuda"
```

For a quick pipeline/UI smoke test without WhisperX, choose **Uniform estimate (demo only)** in the dashboard or pass `alignment_mode=estimate` to the API. These timestamps split total audio duration uniformly and are not valid alignment or benchmark labels.

## Run the app

In one PowerShell window at the repository root:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another window, activate the same virtual environment and run:

```powershell
python -m streamlit run app.py
```

Open the Streamlit URL printed in the terminal (usually `http://localhost:8501`). Select a dataset reference, retain its exact transcript, optionally upload participant WAV/MP3/M4A, and choose forced or demo-estimate alignment. The dashboard includes a waveform with detected-region overlays, aligned word table, baseline/participant feature plots, timestamp seeking, rubric cards, and JSON/CSV export. Uploaded files are size-limited and deleted after analysis; only a preview in the current Streamlit session and the derived analysis result are retained locally. Uploaded audio is excluded from the disk feature/alignment cache.

## API

- `GET /api/health` — readiness and dataset record count
- `GET /api/config` — active feature thresholds and rubric weights
- `GET /api/dataset` and `GET /api/dataset/{sample_id}` — validated dataset catalog
- `GET /api/manifest` — CSV manifest records
- `POST /api/analyze` — baseline ID + exact transcript + optional audio upload + alignment mode
- `POST /api/align` — word alignment for uploaded audio/transcript
- `POST /api/features` — frame-level feature extraction for a dataset role or uploaded audio
- `POST /api/compare` — contrastive analysis (same input contract as analyze)
- `GET /api/result/{analysis_id}` — saved JSON result
- `GET /api/audio/{sample_id}?role=ideal|participant` — stream dataset audio

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### Example analyze request

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/analyze" `
  -F "baseline_id=speaker01_text01_fast" `
  -F "transcript=Paste the exact spoken transcript here." `
  -F "alignment_mode=whisperx" `
  -F "audio=@C:\path\to\participant.wav"
```

## Dataset / reproducibility tools

```powershell
python scripts\validate_dataset.py
python scripts\align_dataset.py --method whisperx
python scripts\extract_features.py
python scripts\benchmark.py --method whisperx --split test
```

`extract_features.py` writes Parquet frame artifacts, `align_dataset.py` writes word-alignment JSON, and `benchmark.py` reports temporal IoU, detection precision/recall, timestamp error, severity MAE, repeated-score delta, and false-positive rate on the ideal reference against `dataset/labels/flaw_regions.csv`. Meaningful benchmark results require genuine held-out labeled recordings; the synthetic demo is not valid benchmark data. With fewer than three text groups the importer deliberately creates no validation/test split; use `python scripts\benchmark.py --method estimate --all` only for a pipeline smoke test, not a quality claim.

### Run automated tests

```powershell
python -m unittest discover -s tests -v
```

### Docker

After ingesting a dataset into `dataset/`, run:

```powershell
docker compose up --build
```

The API is exposed on port 8000 and the dashboard on port 8501. The local dataset and `tmp` outputs are mounted into both containers. WhisperX model dependencies/weights are optional and are not installed in the base image.

Thresholds, frame parameters, and rubric dimension weights live in [configs/pipeline.json](configs/pipeline.json). Dataset-only feature/alignment cache entries are content/config/model keyed under `tmp/cache`; output analysis JSON and the local SQLite result index are under `tmp/results` and `tmp/speechmirror.sqlite3`. Remove those specific local artifact paths when clearing cached analysis results.

## Privacy, limitations, and interpretation

- Local analysis by default; no hosted transcription call is made.
- Uploaded analysis audio is deleted after each API request and is not written to the disk feature cache. The result database retains the derived transcript, timestamps, scores, and numeric word evidence; delete the named `tmp/results` and `tmp/speechmirror.sqlite3` artifacts to remove those results.
- Energy comparisons depend on comparable recording gain/microphone conditions. Record those conditions and interpret energy changes cautiously.
- Speech-rate comparisons and word-level timestamps depend on accurate forced alignment. Review low-coverage or suspicious alignments; do not use estimates as ground truth.
- Spectral changes are acoustic indicators, not direct intelligibility or speech-quality judgments.
- The rubric is deterministic and evidence-linked, but its example thresholds require calibration on consented, correctly labeled data before consequential use.
- The system does not infer emotion, personality, truthfulness, or speech-content correctness.

## Repository layout

```text
backend/                 FastAPI endpoints, schemas, audio/dataset services
configs/pipeline.json    Versioned feature thresholds and rubric weights
dataset/                 Local manifest, labels, audio, alignments, features
scripts/                 Dataset ingestion, alignment, feature, validation, benchmark
speechmirror/            Acoustic features, comparison, scoring, Streamlit UI
tests/                   Reproducibility, validation, feature/evaluation tests
```
