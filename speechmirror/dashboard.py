from __future__ import annotations

import json
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from backend.services.audio_service import analyze_audio_pair
from backend.services.dataset_service import load_manifest
from speechmirror.features import SAMPLE_RATE, load_audio


def _load_records() -> list:
    try:
        return load_manifest(Path(__file__).resolve().parent.parent / "dataset" / "manifest.csv")
    except FileNotFoundError:
        return []


def _waveform_row(audio_source, role: str) -> pd.DataFrame:
    if isinstance(audio_source, dict):
        suffix = Path(audio_source["name"]).suffix.lower() or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
            handle.write(audio_source["data"])
            handle.flush()
            signal, sample_rate = load_audio(handle.name, sample_rate=SAMPLE_RATE)
    else:
        signal, sample_rate = load_audio(audio_source, sample_rate=SAMPLE_RATE)
    stride = max(1, len(signal) // 6000)
    indices = np.arange(0, len(signal), stride)
    return pd.DataFrame(
        {
            "time": indices / sample_rate,
            "amplitude": signal[indices],
            "role": role,
        }
    )


def _render_result(result: dict, record, participant_playback) -> None:
    dimensions = result["dimensions"]
    score_columns = st.columns(len(dimensions) + 1)
    score_columns[0].metric("Overall score", f"{result['overall_score']:.1f}/100")
    for column, (dimension, score) in zip(score_columns[1:], dimensions.items()):
        column.metric(dimension.title(), f"{score:.1f}/100")

    if result["alignment_method"] != "whisperx_forced":
        st.warning(
            "These are uniform estimate timestamps, not forced-alignment output. "
            "Install/configure WhisperX before using these timestamps as ground truth."
        )
    for warning in result.get("quality_warnings", []):
        st.warning(warning)
    st.caption(
        f"Analysis {result['analysis_id']} · {result['alignment_method']} · "
        f"Pipeline {result['reproducibility']['pipeline_version']} · "
        f"input {result['result_hash'][:12]}"
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Reference delivery")
        st.audio(record.ideal_audio)
    with right:
        st.subheader("Participant delivery")
        playback_data = (
            participant_playback["data"]
            if isinstance(participant_playback, dict)
            else participant_playback
        )
        playback_format = (
            participant_playback["format"]
            if isinstance(participant_playback, dict)
            else "audio/wav"
        )
        st.audio(playback_data, format=playback_format)

    st.subheader("Waveform and detected regions")
    waveform = pd.concat(
        [
            _waveform_row(record.ideal_audio, "Baseline"),
            _waveform_row(participant_playback, "Participant"),
        ],
        ignore_index=True,
    )
    flaw_regions = pd.DataFrame(result["flaws"])
    waveform_chart = alt.Chart(waveform).mark_line(strokeWidth=0.7).encode(
        x=alt.X("time:Q", title="Time (seconds)"),
        y=alt.Y("amplitude:Q", title="Amplitude"),
        color=alt.Color("role:N", title="Recording"),
        tooltip=["role:N", "time:Q", "amplitude:Q"],
    ).properties(height=300)
    if not flaw_regions.empty:
        region_chart = alt.Chart(flaw_regions).mark_rect(
            color="#e63946", opacity=0.16
        ).encode(
            x=alt.X("start:Q"),
            x2=alt.X2("end:Q"),
            y=alt.value(0),
            y2=alt.value(300),
            tooltip=["type:N", "start:Q", "end:Q"],
        )
        waveform_chart = region_chart + waveform_chart
    st.altair_chart(waveform_chart.interactive(), use_container_width=True)

    st.subheader("Transcript with aligned word timestamps")
    aligned = result["alignment"]
    if aligned:
        st.dataframe(
            pd.DataFrame(aligned)[["word", "start", "end", "confidence"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Feature comparison")
    evidence = pd.DataFrame(result["word_evidence"])
    if evidence.empty:
        st.info("There are no aligned word-level feature records to plot.")
    else:
        plot_options = {
            "Pacing (words / second)": ("baseline_rate", "participant_rate"),
            "Pitch contour range (semitones)": (
                "baseline_pitch_range",
                "participant_pitch_range",
            ),
            "Energy (dBFS)": ("baseline_energy_dbfs", "participant_energy_dbfs"),
            "Pause duration (seconds)": ("baseline_pause", "participant_pause"),
            "Spectral centroid (Hz)": ("baseline_centroid", "participant_centroid"),
            "MFCC coefficient 1": ("baseline_mfcc_1", "participant_mfcc_1"),
        }
        selected = st.selectbox("Feature", list(plot_options))
        baseline_column, participant_column = plot_options[selected]
        chart_data = evidence.set_index("start")[[baseline_column, participant_column]]
        chart_data.columns = ["Baseline", "Participant"]
        st.line_chart(chart_data)

        st.caption("Each point is a transcript word. Values are aggregated over aligned word boundaries.")
        st.subheader("Word-level evidence")
        st.dataframe(evidence, use_container_width=True, hide_index=True)

    st.subheader("Detected delivery flaws")
    flaws = result["flaws"]
    if not flaws:
        st.success("No configured delivery thresholds were exceeded.")
    else:
        selected_flaw = st.selectbox(
            "Jump to detected region",
            flaws,
            format_func=lambda item: (
                f"{item['type']} · {item['start']:.2f}–{item['end']:.2f}s · "
                f"{item['transcript']}"
            ),
        )
        st.audio(
            playback_data,
            format=playback_format,
            start_time=int(selected_flaw["start"]),
        )
        for flaw in sorted(flaws, key=lambda item: (-item["severity"], item["start"])):
            with st.expander(
                f"{flaw['type']} · {flaw['start']:.2f}–{flaw['end']:.2f}s · "
                f"severity {flaw['severity']:.2f}",
                expanded=True,
            ):
                st.write(f"**Transcript:** {flaw['transcript']}")
                st.write(f"**Why:** {flaw['explanation']}")
                st.write(f"**Try:** {flaw['action']}")
                st.json(flaw["evidence"])

    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button(
            "Download JSON report",
            json.dumps(result, indent=2, ensure_ascii=False),
            file_name=f"{record.sample_id}-{result['analysis_id']}.json",
            mime="application/json",
        )
    with export_right:
        st.download_button(
            "Download CSV evidence",
            evidence.to_csv(index=False),
            file_name=f"{record.sample_id}-{result['analysis_id']}-evidence.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(page_title="SpeechMirror", page_icon="🎙️", layout="wide")
    st.title("SpeechMirror")
    st.caption(
        "Evidence-led comparison of a delivery against a transcript-matched reference. "
        "The tool measures speech features; it does not judge speech content."
    )

    records = _load_records()
    if not records:
        st.error("No validated dataset records found. Build the dataset manifest first.")
        return

    selected_id = st.sidebar.selectbox(
        "Reference speech", [record.sample_id for record in records]
    )
    record = next(item for item in records if item.sample_id == selected_id)
    baseline_transcript = Path(record.transcript_file).read_text(encoding="utf-8").strip()
    st.sidebar.caption(
        f"Speaker: {record.speaker_id} · flaw label: {record.flaw_type} · "
        f"license: {record.source_license or 'not recorded'}"
    )
    transcript = st.sidebar.text_area(
        "Exact transcript", baseline_transcript, height=140
    )
    audio_upload = st.sidebar.file_uploader(
        "Participant audio (optional: otherwise use dataset participant)",
        type=["wav", "mp3", "m4a"],
    )
    alignment_method = st.sidebar.selectbox(
        "Alignment",
        options=["whisperx", "estimate"],
        format_func=lambda value: (
            "WhisperX forced alignment"
            if value == "whisperx"
            else "Uniform estimate (demo only)"
        ),
    )
    run_analysis = st.sidebar.button("Analyze delivery", type="primary")

    result_key = f"analysis:{record.sample_id}"
    playback_key = f"audio:{record.sample_id}"
    if run_analysis:
        st.session_state.pop(result_key, None)
        st.session_state.pop(playback_key, None)
        if " ".join(transcript.split()) != " ".join(baseline_transcript.split()):
            st.sidebar.error("Transcript must exactly match the selected reference transcript.")
        else:
            participant_path = Path(record.participant_audio)
            with tempfile.TemporaryDirectory(prefix="speechmirror-ui-") as temp_dir:
                if audio_upload is not None:
                    participant_path = Path(temp_dir) / f"participant{Path(audio_upload.name).suffix.lower()}"
                    upload_bytes = audio_upload.getvalue()
                    participant_path.write_bytes(upload_bytes)
                    st.session_state[playback_key] = {
                        "data": upload_bytes,
                        "format": audio_upload.type or "audio/wav",
                        "name": audio_upload.name,
                    }
                else:
                    st.session_state[playback_key] = record.participant_audio
                try:
                    with st.spinner("Extracting features, aligning words, and scoring evidence..."):
                        result = analyze_audio_pair(
                            ideal_audio_path=record.ideal_audio,
                            participant_audio_path=participant_path,
                            transcript=transcript,
                            baseline_transcript=baseline_transcript,
                            baseline_id=record.sample_id,
                            alignment_method=alignment_method,
                        )
                    st.session_state[result_key] = result
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    result = st.session_state.get(result_key)
    if result:
        _render_result(result, record, st.session_state.get(playback_key, record.participant_audio))
    else:
        st.info(
            "Choose a reference speech, optionally upload a participant recording, "
            "and select **Analyze delivery** to view the evidence."
        )


if __name__ == "__main__":
    main()
