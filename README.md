# SpeechMirror

SpeechMirror compares a speaker's delivery with a reference recording of the **same words**. It extracts measurable audio features, finds delivery differences, and shows timestamped evidence and feedback in a local dashboard.

> **Know what the demo data is:** the included/generated demo uses artificial test tones. It is not spoken English, a real speech dataset, or a valid accuracy benchmark. To evaluate speech, add recordings you made or are licensed to use.

## Quick start (Windows / PowerShell)

These steps run the dashboard and API in separate terminals. Run commands from the repository folder:

```powershell
cd C:\Users\minha\Documents\atib\Charusat\sem7\SpeechMirror
```

### 1. Create a Python environment and install requirements

Do this once. Python 3.10 or 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell says script execution is disabled, run this in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

When you open a new terminal later, activate the environment in that terminal too:

```powershell
cd C:\Users\minha\Documents\atib\Charusat\sem7\SpeechMirror
.\.venv\Scripts\Activate.ps1
```

### 2. Create demo data (first run only)

This is just to confirm that the app and audio pipeline start. Do **not** run it over a dataset of real recordings; the script refuses to replace an existing non-synthetic manifest.

```powershell
python scripts\build_real_dataset.py --synthetic-demo
python scripts\validate_dataset.py
```

Expected validation output begins with `Dataset valid`.

### 3. Start the API (Terminal 1)

Use `python -m uvicorn`, not the `uvicorn` command. This avoids Windows PATH issues. For the port you are already using:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8002
```

Leave this terminal open. Check that the API is ready:

- Health: http://127.0.0.1:8002/api/health
- Interactive API docs: http://127.0.0.1:8002/docs

### 4. Start the dashboard (Terminal 2)

Open another PowerShell window, go to the project folder, activate `.venv`, then run:

```powershell
cd C:\Users\minha\Documents\atib\Charusat\sem7\SpeechMirror
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Open the URL Streamlit prints, usually http://localhost:8501. If that port is busy, Streamlit will print a different port; open the URL it gives you.

The dashboard currently analyzes the selected dataset directly in its own process. The API is a separate service for API clients and API testing; **the dashboard does not send its analysis requests to the API port**. Therefore, changing the API port to 8002 does not change the dashboard URL or fix missing alignment packages.

### 5. Try the demo analysis

1. In the dashboard sidebar, keep the provided demo reference selected.
2. For the generated test-tone demo, choose **Uniform estimate (demo only)** under Alignment.
3. Click **Analyze delivery**.
4. Review the score cards, waveform, word-level table, feature chart, and flaw evidence.

Uniform estimate divides the recording duration evenly among transcript words. It is **not forced alignment** and should not be used to judge speech or report alignment accuracy.

## Analyze real speech recordings

The project does not supply a genuine speech corpus. Prepare a matched pair of recordings:

- **Ideal/reference:** a clear delivery of the transcript.
- **Participant/flawed:** the same exact transcript, delivered with a known change such as fast pacing or a long pause.

For a useful comparison, record both takes with the same speaker, microphone, and room when possible. Only vary one main delivery behavior at a time. Keep the original recordings and a source CSV together, and use the template [dataset/manifest.example.csv](dataset/manifest.example.csv).

In that CSV, update the example row and paths:

- `sample_id`: unique ID using letters, numbers, `_` or `-`.
- `transcript`: exact words spoken in both recordings.
- `ideal_audio`, `participant_audio`: paths to the two source files, relative to the CSV file or absolute.
- `flaw_type`, `severity`: label and severity from 0 to 4.
- `source_license`: permission or license/provenance; do not leave this blank.
- `start_sec`, `end_sec`: optional known flaw interval in seconds.
- `text_id` and `split`: keep every variant of the same text in one split.

Then, from the project root:

```powershell
python scripts\build_real_dataset.py --source-manifest C:\path\to\your-recordings\manifest.csv
python scripts\validate_dataset.py
```

The importer creates normalized mono 16 kHz WAV copies, transcript files, labels, and the dataset catalog. It keeps source audio hashes and provenance in the catalog. It adds new sample IDs to the existing manifest; it rejects duplicate IDs rather than overwriting them.

For multiple labeled regions per sample, make a second CSV with columns `sample_id,flaw_id,start_sec,end_sec,severity,feature_expected` and add `--labels-manifest C:\path\to\your-recordings\flaws.csv`.

Use at least three different transcript groups if you want the importer to make train/validation/test splits. Variants of one transcript stay in the same split to reduce data leakage. Small datasets are assigned to `train`; they do not provide a meaningful blind test.

## Forced alignment: WhisperX

The dashboard/API defaults to **WhisperX forced alignment**. If WhisperX is missing, analysis reports `No module named 'whisperx'`. Port 8002 does not affect this dependency.

WhisperX and PyTorch have platform-specific requirements. In the activated `.venv`, install a PyTorch build that matches your machine using the official PyTorch installation instructions, then install WhisperX:

```powershell
python -m pip install whisperx
python -c "import whisperx; print('WhisperX is installed')"
```

WhisperX may download alignment-model weights on first use and can require FFmpeg. The model is not bundled with SpeechMirror. If installation is not available, use **Uniform estimate (demo only)** to check the UI; it does not replace forced alignment. WAV is recommended. MP3/M4A decoding may require FFmpeg on `PATH`.

WhisperX uses CPU by default. To select CUDA (only if your PyTorch installation supports it), set this in **both** app terminals before starting them:

```powershell
$env:SPEECHMIRROR_DEVICE = "cuda"
```

## Run the dataset pipeline tools

Run these from the project root, with `.venv` activated:

```powershell
python scripts\validate_dataset.py
python scripts\align_dataset.py --method whisperx
python scripts\extract_features.py
python scripts\benchmark.py --method whisperx --split test
```

- `validate_dataset.py` checks manifest files, audio format, labels, splits, and saved alignments.
- `align_dataset.py` writes per-recording word-alignment JSON files.
- `extract_features.py` writes per-recording frame features as Parquet.
- `benchmark.py` compares detections against labeled regions and reports temporal IoU, precision/recall, timestamp error, severity error, repeated-run consistency, and ideal-recording false positives.

Benchmark figures only mean something with genuine, correctly annotated held-out speech. Do not present synthetic-demo output as accuracy.

## API on port 8002

With the backend running in Terminal 1:

- Health check: http://127.0.0.1:8002/api/health
- Swagger/API explorer: http://127.0.0.1:8002/docs
- Dataset records: http://127.0.0.1:8002/api/dataset

Example API analysis using PowerShell's `curl.exe`:

```powershell
curl.exe -X POST "http://127.0.0.1:8002/api/analyze" `
  -F "baseline_id=YOUR_SAMPLE_ID" `
  -F "transcript=THE EXACT TRANSCRIPT FROM YOUR DATASET" `
  -F "alignment_mode=whisperx" `
  -F "audio=@C:\path\to\participant.wav"
```

For a smoke test without WhisperX, set `alignment_mode=estimate`. The API also provides `/api/align`, `/api/features`, `/api/compare`, `/api/result/{analysis_id}`, and `/api/audio/{sample_id}`.

## Common problems

| Message / symptom | What to do |
|---|---|
| `uvicorn is not recognized` | Activate `.venv` and run `python -m uvicorn ...` instead. |
| `No module named 'whisperx'` | Install WhisperX in the same activated `.venv`, or select **Uniform estimate (demo only)** for a smoke test. |
| `No module named ...` for another dependency | Activate `.venv`, then run `python -m pip install -r requirements.txt`. |
| Port 8002 is already in use | Stop the other backend with Ctrl+C, or choose another API port. Update the API URL in your request too. |
| Streamlit uses another port | Open the URL Streamlit prints; it runs separately from the API. |
| `Transcript must exactly match...` | Use the exact transcript in the selected dataset record. |
| MP3/M4A cannot be decoded | Install FFmpeg and add it to `PATH`, or use WAV. |
| No dataset manifest / records | Run dataset ingestion or the synthetic smoke-test setup above. |

To stop either server, focus its terminal and press **Ctrl+C**.

## What the project measures

The pipeline extracts RMS/energy, pitch/F0, MFCC, spectral centroid, spectral flatness, bandwidth, and zero-crossing rate. It compares aligned words and produces pace, pause, pitch, energy, clarity-shift, and cadence evidence with deterministic rubric scores. Configuration and weights are in [configs/pipeline.json](configs/pipeline.json).

This is a local research/hackathon prototype, not a validated production speech judge. Energy is sensitive to microphone/gain differences; alignment errors affect word-level results; spectral shifts are not direct intelligibility judgments. Thresholds need calibration against consented, human-labeled recordings. The project does not judge meaning, correctness, emotion, or personality.

## Privacy and generated files

Uploaded audio is size-limited and removed by the API after processing. Analysis results and dataset-only caches are stored locally under `tmp/`; results include derived transcript, timestamps, scores, and evidence, but not the uploaded audio. The Streamlit dashboard keeps its uploaded audio preview only in the active session. Remove the local `tmp/results/`, `tmp/cache/`, or `tmp/speechmirror.sqlite3` artifacts if you want to clear those outputs.

## Tests and optional Docker

Run tests:

```powershell
python -m unittest discover -s tests -v
```

After ingesting your dataset, optional Docker startup is:

```powershell
docker compose up --build
```

Docker exposes the API on port 8000 and the dashboard on port 8501. The base image does not install WhisperX or its model weights.

## Project layout

```text
app.py                         Streamlit entry point
backend/                       FastAPI API and dataset/audio services
configs/pipeline.json          Versioned feature settings, thresholds, rubric weights
dataset/manifest.example.csv   Source dataset CSV template
scripts/                       Ingestion, validation, alignment, features, benchmark
speechmirror/                   Features, scoring, and dashboard implementation
tests/                         API, dataset, alignment, scoring, and repeatability tests
```
