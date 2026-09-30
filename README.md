# SpeechMirror

SpeechMirror is a reproducible contrastive speech analytics prototype designed around the requirements in the Track C SRS and Hackathon Build Guide. It compares an ideal delivery against a participant delivery of the same transcript, extracts acoustic evidence, detects temporal flaw windows, and presents rubric-based feedback through a Streamlit dashboard.

## Project goals

- Compare reference and participant audio using aligned signal features.
- Detect temporal flaw regions such as pitch drops, pace spikes, and energy anomalies.
- Provide explainable, rubric-style feedback grounded in acoustic evidence.
- Offer a reproducible demo pipeline with a local dataset and dashboard.

## Architecture

- Data layer: speech pair metadata and synthetic demo dataset generation
- Signal processing: audio synthesis and acoustic feature extraction
- Evaluation engine: normalization, contrastive comparison, and flaw detection
- Dashboard: interactive report with audio playback, feature timelines, and JSON export

## Quick start

1. Create a virtual environment and install dependencies.
2. Generate the demo dataset.
3. Run the Streamlit dashboard.

### Install dependencies

```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Generate demo data

```bash
python scripts/generate_demo_data.py
```

### Start the dashboard

```bash
streamlit run app.py
```

## Repository structure

```text
SpeechMirror/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── demo_manifest.json
│   └── demo/
│       ├── ideal.wav
│       └── participant.wav
├── scripts/
│   └── generate_demo_data.py
└── speechmirror/
    ├── __init__.py
    ├── audio.py
    ├── dashboard.py
    ├── evaluation.py
    └── features.py
```

## Functional coverage

This implementation addresses the key requirements from the guide:

- FR-01: dataset metadata and labels
- FR-02, FR-03: audio and transcript ingestion
- FR-05: acoustic feature extraction via RMS, pitch, MFCC, centroid, and flatness
- FR-06: normalization using z-score comparison
- FR-07, FR-08: contrastive comparison and temporal flaw detection
- FR-09: structured explanations for detected deviations
- FR-10: reproducible scoring from evidence windows
- FR-11: dashboard with playback, score cards, and visualization
- FR-12: JSON export support
- NFR-01, NFR-02, NFR-04: reproducibility, explainability, and performance-friendly demo mode

## Demo behavior

The default demo evaluates a synthetic ideal speech against a deliberately flawed version. The system highlights the main acoustic deltas and produces a report that can be exported as JSON.

## Notes

- The initial version focuses on a compact, deterministic demo rather than large-scale multilingual processing.
- The pipeline is deliberately modular so teams can swap in real forced alignment, ASR, or dataset engineering components in later iterations.
- The project is designed to be hackathon-ready and easy to run locally or in a Docker environment.
