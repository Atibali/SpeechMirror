TRACK C Contrastive Speech Analytics & Temporal Flaw Grounding Software
Requirements Specification (SRS) + Hackathon Implementation & Data
Engineering Guide

Multimodal AI Hackathon 2026 --- Track C Goal: Build a reproducible
speech-evaluation system that compares an ideal delivery with flawed
delivery of the same text, detects the exact time region of delivery
problems, explains the acoustic deviation, and presents rubric-based
feedback in an interactive dashboard.

# 1. Executive Summary

This project is a contrastive speech-evaluation platform. Instead of
judging a speech in isolation, the system uses a paired baseline: an
ideal delivery of a fixed transcript and one or more deliberately flawed
deliveries of that same transcript. The system aligns the transcript to
audio, extracts time-series acoustic features, normalizes
speaker-dependent characteristics, compares the participant against the
baseline, detects temporal deviation regions, maps those deviations to a
delivery rubric, and generates human-readable feedback. The challenge
specifically asks teams to engineer their own paired good/bad spectrum,
perform forced alignment, extract acoustic features such as
FFT/MFCC/pitch/speech rate/pause intervals/vocal clarity, temporally
ground flaw regions, explain the mathematical rationale, and demonstrate
the result through an interactive dashboard. \## 1.1 Success Definition
for the Hackathon - A judge can upload a test speech and transcript and
receive a reproducible score plus timestamped flaws. - Every detected
flaw has an evidence window such as 00:42.3--00:47.8 and at least one
measurable acoustic delta. - The same audio produces the same score and
flaw timestamps when processed repeatedly with the same configuration. -
The dashboard visually overlays baseline vs. participant features and
highlights the detected flaw regions. - The GitHub repository contains
the code, dataset or public dataset link, documentation, reproducible
setup, and demo assets. \## 1.2 Evaluation Priorities

# 2. Scope

## 2.1 In Scope

-   English speech evaluation for the initial hackathon prototype.
-   Single-speaker speech recordings with a fixed transcript per
    evaluation case.
-   Contrastive baseline/flawed dataset creation.
-   Forced alignment to obtain word/segment timestamps.
-   Acoustic feature extraction: RMS/energy, F0/pitch, MFCC, spectral
    features, speech rate and pauses.
-   Speaker-agnostic normalization using relative changes and
    per-speaker baseline statistics.
-   Temporal flaw detection and severity estimation.
-   Rubric-based scoring and actionable feedback.
-   Interactive dashboard with audio playback, timeline,
    waveform/features and flaw cards.
-   Offline/precomputed demo mode plus optional live inference mode. \##
    2.2 Out of Scope for the First Version
-   Universal multilingual judging.
-   Emotion/personality/mental-state inference.
-   Judging the truthfulness or political correctness of speech content.
-   Semantic fact-checking of speech content.
-   Large-scale model training from scratch.
-   Automatic ranking of people against each other.
-   Perfect human-judge replacement. \# 3. Functional Requirements

# 4. Non-Functional Requirements

# 5. Users and Use Cases

## 5.1 Primary User Flow

Upload Audio + Transcript ↓ Audio Validation / Normalization ↓
Transcript Validation ↓ ASR (optional if transcript is absent) ↓ Forced
Alignment ↓ Voice Activity / Pause Detection ↓ Frame-Level Acoustic
Features ↓ Word / Phrase / Window Aggregation ↓ Speaker Normalization ↓
Baseline vs Participant Comparison ↓ Temporal Flaw Detector ↓ Rubric
Scoring + Causal Explanation ↓ Interactive Dashboard + Report \# 6. Data
Strategy --- What Data to Use and Where The most important hackathon
asset is not the number of recordings; it is the quality of the
contrastive pairs. The challenge explicitly prioritizes a clean, aligned
contrastive spectrum over a large noisy corpus. For the first version,
build a compact but carefully controlled dataset. \## 6.1 Recommended
Dataset Composition

## 6.2 Best Data Collection Strategy

For a hackathon, the safest and most reproducible approach is to use
short public-domain/openly licensed text or original text and record
your own ideal delivery and all flawed mirrors. This removes ambiguity
about whether the 'bad' recording truly represents the same text and
makes the contrastive labels exact. - Use the same microphone/room when
creating a controlled pair whenever possible. - Keep the transcript
identical across ideal and flawed versions. - Change one primary
delivery variable at a time for clean causal labels. - Then create
compound flaws for stress testing. - Keep a manifest containing source,
license/permission, speaker, recording conditions and transformation
history. \## 6.3 Flaw Spectrum

## 6.4 Primary Flaw Labels

# 7. Dataset Schema

Recommended files: dataset/ manifest.csv texts/ text_001.txt audio/
speaker_01/ text_001/ ideal.wav pace_fast_g2.wav pause_long_g2.wav
pitch_flat_g2.wav alignments/ speaker_01_text_001_ideal.json
speaker_01_text_001_pace_fast_g2.json labels/ flaw_regions.csv features/
\*.parquet splits/ train.csv validation.csv test.csv \## 7.1
manifest.csv

## 7.2 flaw_regions.csv

sample_id,flaw_id,start_sec,end_sec,severity,feature_expected,notes
S01_T001_G2_PACEFAST,F001,42.3,47.8,2,speech_rate,Fast delivery of
phrase S01_T001_G2_PAUSE,F002,58.1,60.0,2,pause_duration,Injected long
pause \## 7.3 Alignment JSON { "sample_id": "S01_T001_G2_PACEFAST",
"transcript": "We can build a better future together.", "words": \[
{"word":"We","start":42.30,"end":42.48},
{"word":"can","start":42.49,"end":42.71},
{"word":"build","start":42.72,"end":43.03} \] } \# 8. System
Architecture ┌───────────────────────┐ │ React Dashboard │ │ Upload /
Timeline / UI │ └───────────┬─────────────┘ │ REST / JSON
┌───────────▼─────────────┐ │ FastAPI Backend │ │ session +
orchestration │ └───────────┬─────────────┘ │
┌────────────────────────┼─────────────────────────┐ │ │ │
┌──────▼──────┐ ┌────────▼────────┐ ┌────────▼─────────┐ │ Audio/VAD │ │
WhisperX Align │ │ Feature Engine │ │ FFmpeg │ │ ASR + alignment │ │
librosa + custom │ └──────┬──────┘ └────────┬────────┘
└────────┬─────────┘ │ │ │
└────────────────────────┼─────────────────────────┘ ▼
┌──────────────────────┐ │ Contrastive Engine │ │ normalize + compare │
└──────────┬───────────┘ ▼ ┌──────────────────────┐ │ Flaw + Rubric
Engine │ │ timestamps + reason │ └──────────┬───────────┘ ▼
┌──────────────────────┐ │ JSON / SQLite / Files│
└──────────────────────┘ \## 8.1 Recommended Stack

# 9. Acoustic Feature Engineering

Use short frames (for example 20--40 ms with overlap) for acoustic
features, then aggregate them into word, phrase and sliding-window
statistics. Do not compare raw feature values blindly across speakers.

## 9.1 Why MFCC + RMS + F0 + Rate + Pauses

The challenge names FFT/MFCCs, pitch/F0, speech rate, pause intervals
and vocal clarity as examples of the acoustic footprint. A practical MVP
should therefore prioritize interpretable features that can directly
support a causal explanation rather than adding many opaque embeddings.
\# 10. Forced Alignment Design If the transcript is known, treat it as
ground truth and align it to the audio. This is especially important
because the challenge requires temporal grounding and exact flaw
regions. WhisperX is suitable for the MVP because its pipeline uses
phoneme-based forced alignment to produce word-level timing.
citeturn0search5 - Input: normalized audio + exact transcript +
language. - Output: word-level start/end timestamps, segment timestamps
and alignment metadata. - Store the alignment JSON as a versioned
artifact; do not recompute it every dashboard refresh. - Flag
low-confidence/un-alignable words rather than silently treating them as
perfect alignment. - For known transcripts, use the known transcript for
the final alignment instead of relying on an ASR transcript that may
contain recognition errors. \## 10.1 Alignment Validation

# 11. Contrastive Comparison Algorithm

The core design should compare equivalent semantic/time regions rather
than simply comparing two entire recordings. 1. Align both baseline and
participant recordings to the same transcript. 1. Map words into
phrases/windows, e.g. 5--10 words or 1--3 seconds. 1. Extract feature
statistics for each window. 1. Normalize speaker-dependent features. 1.
Calculate baseline distribution and participant deviation. 1. Smooth
noisy frame-level deviations with a short median/rolling filter. 1.
Apply thresholds and minimum-duration rules to create candidate flaw
regions. 1. Merge nearby candidate regions of the same flaw type. 1. Map
each region to rubric rules and generate evidence-based feedback. 1.
Store every intermediate value so the result is auditable. \## 11.1
Example Deviation baseline_rate = 3.8 words/sec participant_rate = 5.4
words/sec

relative_delta = (5.4 - 3.8) / 3.8 = 0.421 = +42.1%

if relative_delta \> 0.30 for \>= 1.5 sec: candidate = PACE_FAST \##
11.2 Avoid Hard-Coding One Global Threshold A single threshold such as
'5 words/sec is bad' is not speaker-agnostic. Prefer baseline-relative
thresholds and robust statistics. For example, compare the participant
to the same speaker's ideal delivery for the same phrase, then use a
robust deviation score. z = (participant_feature - baseline_median) /
(baseline_MAD + epsilon)

candidate if: abs(z) \>= threshold AND region_duration \>=
minimum_duration AND confidence \>= minimum_confidence \# 12. Rubric and
Scoring Model The scoring model should be transparent and deterministic.
The system should never output a score without being able to list the
feature evidence that contributed to it.

## 12.1 Example Score Calculation

dimension_score = 100 - penalty

penalty = 0.40 \* pacing_penalty + 0.25 \* pitch_penalty + 0.20 \*
energy_penalty + 0.15 \* pause_penalty

overall = 0.25 \* pacing + 0.20 \* pitch + 0.20 \* energy + 0.15 \*
clarity + 0.10 \* emphasis + 0.10 \* consistency The exact weights
should be treated as configuration, not hidden constants. During the
hackathon, document how they were chosen and keep the configuration in a
versioned YAML/JSON file. \# 13. Causal Explainability Engine Use rule
templates driven by measured deltas. Do not generate generic feedback
disconnected from the signal.

## 13.1 Feedback Object

{ "flaw_id": "F001", "type": "PACE_FAST", "severity": 0.82, "start":
42.3, "end": 47.8, "transcript": "We can build a better future
together.", "evidence": { "baseline_rate": 3.8, "participant_rate": 5.4,
"relative_delta": 0.421 }, "explanation": "Delivery was 42.1% faster
than baseline.", "action": "Reduce rate and preserve a short pause
before the key phrase." } \# 14. Dashboard Specification \## 14.1 Main
Screen ┌──────────────────────────────────────────────────────────┐ │
SpeechMirror Overall: 82 / 100 │
├───────────────┬──────────────────────────────────────────┤ │ SCORE
CARDS │ Waveform / Audio Player │ │ Pacing 76 │ ────────████ flaw region
███────── │ │ Pitch 88 │ 0:00 0:45 1:30│ │ Energy 84
├──────────────────────────────────────────┤ │ Pauses 79 │ Feature
Overlay │ │ Clarity 86 │ Baseline ───────────── │ │ │ Test ────╱╲──────
│ ├───────────────┴──────────────────────────────────────────┤ │
Detected Flaws │ │ 00:42.3--00:47.8 HIGH Fast Pacing │ │ Baseline: 3.8
w/s \| Test: 5.4 w/s \| +42.1% │ │ Why: delivery is faster than the
reference │ │ Fix: slow down around the key phrase │
└──────────────────────────────────────────────────────────┘ \## 14.2
Dashboard Features - Audio player with clickable flaw timestamps. -
Waveform with colored/annotated flaw regions. - Transcript synchronized
to playback. - Baseline vs participant line charts for selected
feature. - Score cards by rubric dimension. - Flaw list sorted by
severity or timeline. - Evidence panel showing raw values, baseline
values and deltas. - Toggle: RMS / F0 / speech rate / pause /
MFCC-derived metric. - Download JSON/CSV report. - Demo mode with
precomputed examples for reliable judging. \# 15. Backend API

## 15.1 Suggested Analyze Request

POST /api/analyze multipart/form-data

audio=\<participant.wav\> transcript=\<speech.txt\>
baseline_id=`<SPK01_T001_IDEAL>`{=html} mode=live \## 15.2 Suggested
Analyze Response { "analysis_id": "A1024", "overall_score": 82,
"dimensions": { "pacing": 76, "pitch": 88, "energy": 84, "clarity": 86
}, "flaws": \[ { "start": 42.3, "end": 47.8, "type": "PACE_FAST",
"severity": "high", "evidence": {"baseline": 3.8, "actual": 5.4},
"explanation": "..." } \] } \# 16. Recommended Repository Structure
speechmirror/ ├── README.md ├── LICENSE ├── docker-compose.yml ├──
.env.example ├── requirements.txt ├── package.json │ ├── backend/ │ ├──
app/ │ │ ├── main.py │ │ ├── api/ │ │ ├── schemas/ │ │ ├── services/ │ │
│ ├── alignment.py │ │ │ ├── audio.py │ │ │ ├── features.py │ │ │ ├──
comparison.py │ │ │ ├── flaw_detector.py │ │ │ ├── scoring.py │ │ │ └──
explanation.py │ │ └── config.py │ └── tests/ │ ├── frontend/ │ ├── src/
│ │ ├── components/ │ │ ├── pages/ │ │ ├── charts/ │ │ ├── api/ │ │ └──
types/ │ ├── dataset/ │ ├── manifest.csv │ ├── texts/ │ ├── audio/ │ ├──
alignments/ │ ├── labels/ │ ├── features/ │ └── splits/ │ ├── configs/ │
├── features.yaml │ ├── rubric.yaml │ └── thresholds.yaml │ ├── scripts/
│ ├── normalize_audio.py │ ├── align_dataset.py │ ├──
extract_features.py │ ├── validate_dataset.py │ └── build_demo.py │ ├──
docs/ │ ├── SRS.md │ ├── DATASET.md │ ├── ARCHITECTURE.md │ └──
METHODOLOGY.md │ └── demo/ ├── examples/ └── screenshots/ \# 17.
Hackathon Build Plan

## 17.1 Minimum Viable Product

If time is limited, implement these in this order: - 1) 10 clean
transcripts + 10 ideal recordings. - 2) At least 3 controlled flaws:
fast pacing, long pause, flat pitch. - 3) Forced alignment with word
timestamps. - 4) RMS + F0 + speech rate + pause detection. - 5)
Baseline-relative comparison. - 6) Timestamped flaw cards with numerical
evidence. - 7) Dashboard overlay. - 8) Reproducibility test and demo
video. \# 18. Stress Testing Plan

## 18.1 Ground-Truth Evaluation Metrics

-   Temporal IoU: overlap between predicted flaw interval and labeled
    flaw interval.
-   Detection precision/recall: whether the intended flaw was detected.
-   Timestamp error: absolute difference between predicted and labeled
    start/end.
-   Severity agreement: predicted severity vs controlled dataset
    severity.
-   Score consistency: variance across repeated runs.
-   False-positive rate on ideal recordings. \# 19. Dataset Split and
    Leakage Prevention Do not randomly split individual flawed variants
    from the same transcript across train and test. That makes the test
    artificially easy because the model/rules can see the same
    transcript, speaker and baseline conditions.
-   Prefer speaker-disjoint or text-disjoint test cases where practical.
-   Keep all variants of one base recording/text together when
    evaluating generalization.
-   Do not tune thresholds using the final blind test set.
-   Record the dataset version and configuration hash with every
    benchmark. \# 20. Reproducibility Specification

## 20.1 Result Hash

result_id = SHA256( audio_hash + transcript_hash + baseline_id +
model_version + feature_config_hash + rubric_config_hash ) \# 21. Data
Licensing, Privacy and Provenance - Prefer original/self-recorded audio
for the paired flawed dataset. - For external speeches, record the
source URL, speaker, date accessed, license/usage basis and transcript
provenance. - Do not redistribute copyrighted recordings in the
repository unless the license/permission allows redistribution. - If a
source is only used as a reference, keep metadata and instructions
rather than bundling the restricted audio. - Keep participant uploads
temporary unless consent is obtained for retention. - Document
microphone/room/recording settings for every dataset batch. \# 22.
Implementation Details \## 22.1 Audio Preprocessing 1. Decode with
FFmpeg 2. Convert to mono 3. Resample to a fixed sample rate (e.g. 16
kHz) 4. Trim only leading/trailing non-speech when configured 5.
Preserve the original file hash 6. Store processed WAV separately 7.
Record preprocessing configuration \## 22.2 Feature Pipeline audio -\>
frames -\> VAD -\> RMS / dB -\> F0 -\> STFT / spectral features -\> MFCC
-\> word-aligned aggregation -\> phrase/window aggregation -\>
normalized features -\> baseline deltas -\> flaw detector librosa
provides standard feature extraction functions including MFCC, RMS, mel
spectrogram, spectral centroid and related spectral features, making it
appropriate for the deterministic feature-engineering layer.
citeturn0search1turn0search13 \## 22.3 Optional ML Layer Do not make
a black-box classifier the core of the hackathon system. If time
permits, add a lightweight classifier or anomaly detector after the
interpretable rule-based baseline is working. The rule engine provides
the causal evidence needed by the challenge; ML can improve ranking of
candidate regions. - Option A: Isolation Forest on normalized feature
windows. - Option B: Gradient Boosting classifier for known flaw
labels. - Option C: Small temporal model over feature sequences. -
Always retain the raw feature deltas used to explain the final
prediction. \# 23. Recommended 3--10 Minute Demo Flow 1. Show the
problem: two deliveries of the exact same text can sound dramatically
different. 1. Show the dataset: one ideal recording plus several
controlled flawed mirrors. 1. Open the dashboard and load a prepared
test recording. 1. Play the speech and show the synchronized
transcript. 1. Jump to a detected timestamp, e.g. 00:42.3--00:47.8. 1.
Show the baseline vs participant speech-rate graph. 1. Show the
numerical delta and explain the rule that fired. 1. Show the rubric
score change caused by the flaw. 1. Test another flaw such as a long
pause or flat pitch. 1. Finish with dataset size, reproducibility and
GitHub architecture. \# 24. Final Submission Checklist - ☐ GitHub
repository with clean README and setup instructions. - ☐ Custom paired
dataset with ideal/flawed audio, transcripts and alignment labels. - ☐
Dataset manifest and provenance/license information. - ☐ Functional
interactive dashboard. - ☐ Forced alignment implementation. - ☐ Acoustic
feature extraction implementation. - ☐ Temporal flaw detector. - ☐
Mathematical evidence for each flaw. - ☐ Rubric scoring configuration. -
☐ Actionable feedback generation. - ☐ Blind test set and stress-test
results. - ☐ Reproducibility instructions. - ☐ Technical documentation
≤6 pages for the official technical submission; this SRS can be kept as
the engineering master document. - ☐ 3--10 minute YouTube demonstration.
\# 25. README Structure \# SpeechMirror --- Contrastive Speech Analytics

## 1. Problem

## 2. Solution

## 3. Architecture

## 4. Dataset

## 5. Contrastive Flaw Spectrum

## 6. Feature Engineering

## 7. Forced Alignment

## 8. Temporal Grounding

## 9. Rubric & Scoring

## 10. Dashboard

## 11. Quick Start

## 12. Dataset Access / License

## 13. Reproducibility

## 14. Evaluation Results

## 15. Demo Video

## 16. Team / Credits

# 26. Risks and Mitigations

# 27. Acceptance Criteria

-   A valid audio + transcript can be processed end-to-end.
-   At least one known flawed recording is detected in the correct
    temporal region.
-   The dashboard shows the corresponding transcript phrase and audio
    timestamp.
-   The system displays at least two numerical pieces of evidence for a
    detected flaw.
-   The same input produces the same result across repeated runs.
-   An ideal recording does not generate a large number of severe false
    positives.
-   A different speaker can be evaluated without using absolute pitch as
    a direct quality score.
-   Dataset labels can be traced from manifest → audio → alignment →
    feature windows → flaw result.
-   The demo can run using precomputed artifacts even if live ML
    inference is unavailable. \# 28. Source Notes Challenge-specific
    requirements in this document are derived from the uploaded Track C
    brief: custom ideal-vs-flawed contrastive dataset, forced alignment,
    acoustic feature extraction, temporal flaw grounding, causal
    explanations, dashboard, dataset/code/documentation/video
    deliverables, and the stated evaluation weights.
    fileciteturn0file0L23-L30 The brief also explicitly calls for
    forced alignment and features such as FFT, MFCC, pitch/F0, speech
    rate, pause intervals and vocal clarity.
    fileciteturn0file0L32-L42 Implementation references consulted:
    WhisperX project documentation/repository for word-level forced
    alignment and related speech processing capabilities; librosa
    documentation for MFCC/RMS/spectral feature extraction; FastAPI
    documentation for file upload APIs.
    citeturn0search5turn0search1turn0search15 Reference links:
-   https://github.com/m-bain/whisperX
-   https://librosa.org/doc/0.11.0/feature.html
-   https://fastapi.tiangolo.com/tutorial/request-files/

  -----------------------------------------------------------------------
  Criterion               Weight                  Engineering Priority
  ----------------------- ----------------------- -----------------------
  Data Engineering &      30%                     Highest --- clean
  Stress Testing                                  paired spectrum and
                                                  temporal labels

  Causal Explainability & 25%                     Highest --- precise
  Temporal Grounding                              timestamps +
                                                  mathematical rationale

  Feature Extraction      20%                     High --- reliable
                                                  acoustic features and
                                                  normalization

  Visualization &         15%                     High --- clear,
  Dashboard                                       interactive evidence

  Reproducibility & Code  10%                     Required ---
  Quality                                         deterministic pipeline
                                                  and setup
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  ID                      Requirement             Specification
  ----------------------- ----------------------- -----------------------
  FR-01                   Dataset Management      System shall store
                                                  speech-pair metadata,
                                                  transcript, speaker ID,
                                                  flaw type, severity,
                                                  and temporal labels.

  FR-02                   Audio Ingestion         System shall accept
                                                  WAV/MP3/M4A or a
                                                  normalized WAV
                                                  representation.

  FR-03                   Transcript Ingestion    System shall accept the
                                                  exact transcript used
                                                  for the recording.

  FR-04                   Alignment               System shall produce
                                                  word/segment timestamps
                                                  for the transcript.

  FR-05                   Feature Extraction      System shall compute
                                                  time-series acoustic
                                                  features for baseline
                                                  and test audio.

  FR-06                   Normalization           System shall normalize
                                                  speaker-dependent pitch
                                                  and energy before
                                                  comparison.

  FR-07                   Contrastive Comparison  System shall compare
                                                  participant features
                                                  with a baseline using
                                                  aligned
                                                  temporal/semantic
                                                  regions.

  FR-08                   Flaw Detection          System shall identify
                                                  candidate temporal
                                                  regions where
                                                  deviations exceed
                                                  configured thresholds.

  FR-09                   Causal Explanation      System shall convert
                                                  detected deviations
                                                  into a structured
                                                  explanation tied to a
                                                  rubric dimension.

  FR-10                   Scoring                 System shall calculate
                                                  reproducible rubric
                                                  scores from feature
                                                  evidence.

  FR-11                   Dashboard               System shall display
                                                  audio playback,
                                                  transcript, timestamps,
                                                  feature overlays, score
                                                  cards and feedback.

  FR-12                   Export                  System should allow
                                                  JSON/CSV report export
                                                  and optionally a PDF
                                                  summary.

  FR-13                   Reproducibility         System shall record
                                                  pipeline
                                                  version/configuration
                                                  and use fixed
                                                  preprocessing
                                                  parameters.
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  ID                      Area                    Requirement
  ----------------------- ----------------------- -----------------------
  NFR-01                  Reproducibility         Same input + same
                                                  version/configuration
                                                  should produce the same
                                                  feature/scoring result.

  NFR-02                  Explainability          Every major score
                                                  should have measurable
                                                  evidence; avoid opaque
                                                  'AI says so' feedback.

  NFR-03                  Speaker Agnostic        Pitch/energy comparison
                                                  must use normalized
                                                  values rather than raw
                                                  biological differences.

  NFR-04                  Performance             A 1--3 minute demo clip
                                                  should process fast
                                                  enough for a live
                                                  hackathon
                                                  demonstration; cache
                                                  artifacts where needed.

  NFR-05                  Usability               A first-time judge
                                                  should understand the
                                                  main score, flaws and
                                                  timestamps without
                                                  reading documentation.

  NFR-06                  Data Quality            Paired recordings must
                                                  use the exact same
                                                  transcript and clean
                                                  alignment labels.

  NFR-07                  Privacy                 Uploaded participant
                                                  audio should be
                                                  processed locally or
                                                  deleted after analysis
                                                  unless explicitly
                                                  retained.

  NFR-08                  Portability             Provide Docker/local
                                                  setup and pinned
                                                  dependency versions.
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  Actor                   Use Case                Outcome
  ----------------------- ----------------------- -----------------------
  Participant             Upload speech +         Receives score and
                          transcript              actionable delivery
                                                  feedback.

  Judge/Demo User         Open prepared example   Sees a known flaw
                                                  caught at a known
                                                  timestamp.

  Dataset Engineer        Add good/bad pair       Creates a validated
                                                  contrastive sample with
                                                  labels.

  Developer               Run evaluation pipeline Gets reproducible
                                                  JSON/feature artifacts.
  -----------------------------------------------------------------------

  ------------------------------------------------------------------------------
  Layer             Recommended Size  Purpose           Used By
  ----------------- ----------------- ----------------- ------------------------
  Text bank         10--20 speeches   Fixed transcripts Dataset + alignment +
                                      representing      evaluation
                                      rhetorical styles 

  Ideal recordings  2--3 speakers ×   Clean reference   Baseline feature
                    10--20 texts      delivery          profiles

  Flawed mirrors    4--6 variants per Controlled stress Training/calibration +
                    ideal             spectrum          stress tests

  Blind test set    20--30 recordings Never used to     Final demo/evaluation
                                      tune thresholds   

  Edge cases        10--20 clips      Noise, low        Robustness testing
                                      volume, long      
                                      pause, fast       
                                      speech            
  ------------------------------------------------------------------------------

  -----------------------------------------------------------------------
  Level             Example           How to Create     Expected Acoustic
                                                        Evidence
  ----------------- ----------------- ----------------- -----------------
  G0 Ideal          Natural           Normal            Reference
                    persuasive        performance       distribution
                    delivery                            

  G1 Near-perfect   Small pacing/tone Minor controlled  Small delta
                    deviation         change            

  G2 Mild           Noticeably fast   Moderate          Moderate delta
                    or flat           controlled change 

  G3 Strong         Very fast, long   Large controlled  Large localized
                    pauses, monotone  change            delta

  G4 Severe         Clearly botched   Extreme           Large and obvious
                    delivery          controlled change delta
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  Flaw ID           Delivery          Primary Features  Feedback Example
                    Dimension                           
  ----------------- ----------------- ----------------- -----------------
  PACE_FAST         Too fast          Speech rate,      Slow down in this
                                      duration/word,    phrase; rate is
                                      pause ratio       substantially
                                                        above baseline.

  PACE_SLOW         Too slow          Speech rate,      Delivery is
                                      pause ratio       slower than the
                                                        reference and
                                                        loses momentum.

  PAUSE_LONG        Excessive pause   VAD, silence      Pause at 00:43.2
                                      duration          is much longer
                                                        than the
                                                        reference.

  PAUSE_MISSING     Insufficient      Pause interval    Expected
                    pause                               rhetorical pause
                                                        is compressed or
                                                        absent.

  PITCH_FLAT        Monotone          F0 range, F0      Pitch variation
                                      variance          is below the
                                                        reference in this
                                                        emphasis region.

  PITCH_SPIKE       Unnatural pitch   F0 delta/z-score  Pitch changes
                    spike                               sharply relative
                                                        to local
                                                        baseline.

  ENERGY_LOW        Weak projection   RMS/energy        Energy is below
                                      percentile        the normalized
                                                        baseline.

  ENERGY_HIGH       Over-projection   RMS/energy        Energy is
                                      percentile        substantially
                                                        above baseline.

  CLARITY           Reduced vocal     Spectral/MFCC     Acoustic clarity
                    clarity           stability,        changed sharply
                                      optional ASR      in this region.
                                      confidence        

  CADENCE           Rhythmic          Rate + pause +    Cadence differs
                    inconsistency     energy contour    from the
                                                        reference
                                                        pattern.
  -----------------------------------------------------------------------

  Field             Example                  Why It Exists
  ----------------- ------------------------ ------------------------------------------
  sample_id         S01_T001_G2_PACEFAST     Unique reproducible identifier
  text_id           T001                     Pairs exact transcript across recordings
  speaker_id        SPK01                    Speaker normalization/grouping
  recording_id      R008                     Audio-level identity
  audio_path        audio/SPK01/T001/...     Audio location
  transcript_path   texts/T001.txt           Ground-truth text
  condition         flawed                   ideal/flawed
  flaw_type         PACE_FAST                Primary intended flaw
  severity          2                        Controlled stress level
  start_sec         42.3                     Ground-truth flaw start
  end_sec           47.8                     Ground-truth flaw end
  source_license    Original/self-recorded   Provenance
  split             test                     Prevents leakage

  ----------------------------------------------------------------------------
  Layer                   Technology              Reason
  ----------------------- ----------------------- ----------------------------
  Frontend                React + Vite +          Fast dashboard development
                          TypeScript              and charting

  Charts                  Plotly.js / Recharts    Interactive time-series
                                                  overlays

  Backend                 Python + FastAPI        Natural fit for audio/ML
                                                  pipeline

  Audio                   FFmpeg +                Normalization and feature
                          soundfile/librosa       extraction

  ASR/Alignment           WhisperX                Word-level timestamps using
                                                  forced alignment

  Acoustic Features       librosa + NumPy/SciPy   MFCC, RMS, spectral and
                                                  signal processing

  Storage                 JSON + Parquet; SQLite  Simple, reproducible
                          for sessions            hackathon storage

  Packaging               Docker + requirements   Deployment/reproducibility
                          lock                    
  ----------------------------------------------------------------------------

  ---------------------------------------------------------------------------------
  Feature           Formula /         Role                      Normalization
                    Measurement                                 
  ----------------- ----------------- ------------------------- -------------------
  RMS Energy        sqrt(mean(x²))    Volume/projection         Per-speaker z-score
                                                                or percentile

  dB Energy         20 log10(RMS + ε) Readable energy scale     Relative to speaker
                                                                baseline

  F0 / Pitch        Fundamental       Pitch movement            Semitone or z-score
                    frequency                                   relative to
                                                                baseline

  F0 Range          P95(F0) − P5(F0)  Expressiveness/flatness   Relative range

  MFCC              DCT(log Mel       Vocal spectral footprint  Standardize per
                    spectrum)                                   speaker

  Spectral Centroid Σ f·m(f) / Σm(f)  Brightness/spectral shift Relative delta

  Speech Rate       words or          Pacing                    Compare within
                    syllables /                                 aligned phrase
                    second                                      

  Pause Duration    silence interval  Cadence                   Relative to nearby
                                                                baseline

  Pause Ratio       silence time /    Overall rhythm            Baseline-relative
                    window time                                 

  Feature Variance  window variance   Stability                 Normalized per
                                                                feature
  ---------------------------------------------------------------------------------

  -----------------------------------------------------------------------
  Check                   Rule                    Action
  ----------------------- ----------------------- -----------------------
  Word coverage           Most transcript words   Reject/review if
                          have timestamps         coverage is too low

  Monotonicity            start \<= end and word  Repair/reject invalid
                          order is increasing     alignment

  Audio bounds            timestamps inside audio Clamp only with logged
                          duration                warning

  Transcript match        Exact normalized text   Fail evaluation if
                          matches dataset text    mismatch

  Flaw overlap            Ground-truth flaw       Required for label
                          overlaps aligned words  quality
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  Dimension               Suggested Weight        Evidence
  ----------------------- ----------------------- -----------------------
  Pacing & Cadence        25                      Speech rate, pause
                                                  duration, pause ratio,
                                                  rate stability

  Pitch & Intonation      20                      F0 range, contour
                                                  variation, local pitch
                                                  deviations

  Projection & Energy     20                      RMS/dB energy, energy
                                                  stability, normalized
                                                  dynamic range

  Vocal Clarity           15                      Spectral/MFCC
                                                  stability, optional
                                                  ASR/alignment quality

  Pausing & Emphasis      10                      Strategic pauses +
                                                  energy/pitch emphasis
                                                  around key phrases

  Consistency             10                      Temporal stability and
                                                  absence of repeated
                                                  severe deviations
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  Detected Evidence                   Causal Template
  ----------------------------------- -----------------------------------
  Rate +35% vs baseline               Pacing: this phrase was delivered
                                      substantially faster than the
                                      reference, reducing pause
                                      separation and emphasis.

  Pause +1.2 sec vs baseline          Pausing: the silence interval is
                                      longer than the reference and
                                      interrupts the phrase cadence.

  F0 range −50%                       Intonation: pitch variation is
                                      substantially lower than the
                                      reference, producing a flatter
                                      contour.

  Energy −30%                         Projection: normalized energy is
                                      below the reference in this region,
                                      indicating weaker vocal projection.

  Multiple correlated deltas          Cadence: rate, pause and energy
                                      changed together, producing a
                                      compound delivery deviation.
  -----------------------------------------------------------------------

  Method   Endpoint            Purpose
  -------- ------------------- -----------------------------------------------
  POST     /api/analyze        Upload audio + transcript and start analysis
  POST     /api/align          Run or retrieve forced alignment
  POST     /api/features       Extract acoustic features
  POST     /api/compare        Compare participant against selected baseline
  GET      /api/result/{id}    Return score, flaws and evidence
  GET      /api/audio/{id}     Stream processed audio
  GET      /api/dataset/{id}   Return dataset metadata for demo
  GET      /api/health         Health/readiness check

  ------------------------------------------------------------------------
  Phase                   Tasks                    Deliverable
  ----------------------- ------------------------ -----------------------
  Phase 1                 Choose 10--20 texts;     Text bank + ideal
                          define rubric; record    dataset
                          2--3 ideal speakers      

  Phase 2                 Create 4--6 controlled   Contrastive spectrum
                          flaw variants per sample 

  Phase 3                 Normalize audio; run     Alignment JSON + label
                          alignment; validate      CSV
                          labels                   

  Phase 4                 Implement                Feature Parquet/JSON
                          RMS/F0/rate/pause/MFCC   
                          features                 

  Phase 5                 Implement                Flaw detection engine
                          normalization + delta +  
                          temporal detector        

  Phase 6                 Implement deterministic  Evaluation engine
                          scoring + explanation    
                          rules                    

  Phase 7                 Build React dashboard +  Interactive prototype
                          audio timeline           

  Phase 8                 Create blind tests +     Stress-test suite
                          edge cases + demo mode   

  Phase 9                 Documentation + video +  Final submission
                          README + deployment      
  ------------------------------------------------------------------------

  -----------------------------------------------------------------------
  Test                    Input Condition         Expected Behavior
  ----------------------- ----------------------- -----------------------
  T01                     Ideal recording         Few/no severe flaws;
                                                  high score

  T02                     Fast pacing             PACE_FAST localized to
                                                  changed phrase

  T03                     Long pause              PAUSE_LONG at inserted
                                                  silence

  T04                     Flat pitch              PITCH_FLAT around
                                                  modified region

  T05                     Low energy              ENERGY_LOW after
                                                  normalization

  T06                     Compound flaw           Multiple evidence
                                                  signals; avoid
                                                  duplicate alerts

  T07                     Different speaker       Speaker normalization
                                                  prevents raw-pitch
                                                  penalty

  T08                     Background noise        Confidence reduced /
                                                  warning rather than
                                                  hallucinated flaw

  T09                     Transcript mismatch     Validation error

  T10                     Repeated run            Same result
                                                  hash/timestamps
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  Artifact                            Must Be Versioned
  ----------------------------------- -----------------------------------
  Python dependencies                 requirements.txt / lock file

  Model versions                      ASR/alignment model identifiers

  Feature parameters                  sample rate, frame length, hop
                                      length, n_mfcc etc.

  Thresholds                          thresholds.yaml

  Rubric weights                      rubric.yaml

  Dataset manifest                    manifest.csv + dataset version

  Preprocessing                       normalization and resampling
                                      configuration

  Code                                Git commit SHA

  Result                              input hash + configuration hash
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  Risk                    Impact                  Mitigation
  ----------------------- ----------------------- -----------------------
  Alignment errors        Wrong timestamps        Validate alignment; use
                                                  known transcript and
                                                  forced alignment

  Speaker differences     False flaws             Per-speaker baseline
                                                  normalization

  Room/microphone         Energy/spectral drift   Controlled recording +
  differences                                     normalization

  Over-sensitive          Too many flaws          Minimum duration +
  thresholds                                      robust statistics +
                                                  blind validation

  Under-sensitive         Missed flaws            Controlled G1--G4
  thresholds                                      stress spectrum

  Copyright issues        Submission risk         Self-recorded/openly
                                                  licensed dataset

  Live inference too slow Demo failure            Precompute demo
                                                  examples and cache
                                                  features

  Black-box feedback      Poor explainability     Rule/evidence-first
                                                  explanations
  -----------------------------------------------------------------------
