from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .audio import generate_demo_dataset
from .evaluation import compare_pair, export_results
from .features import extract_feature_frame, load_audio


def _load_manifest(base_dir: str | Path) -> dict:
    manifest_path = Path(base_dir) / "demo_manifest.json"
    if not manifest_path.exists():
        return generate_demo_dataset(base_dir)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _evaluate_pair(pair: dict) -> dict:
    ideal_signal, ideal_sr = load_audio(pair["ideal_audio"], sample_rate=16_000)
    participant_signal, participant_sr = load_audio(pair["participant_audio"], sample_rate=16_000)

    ideal_frame = extract_feature_frame(ideal_signal, ideal_sr)
    participant_frame = extract_feature_frame(participant_signal, participant_sr)
    result = compare_pair(ideal_frame, participant_frame)
    result["pair_id"] = pair["pair_id"]
    result["transcript"] = pair["transcript"]
    result["labels"] = pair.get("labels", [])
    return result


def main() -> None:
    st.set_page_config(page_title="SpeechMirror", page_icon="🎙️", layout="wide")
    st.title("SpeechMirror")
    st.caption("Contrastive speech analytics system for temporal flaw grounding and rubric-based feedback.")

    project_root = Path(__file__).resolve().parent.parent
    demo_manifest = _load_manifest(project_root / "data")
    pair_options = [demo_manifest["pair_id"]]

    st.sidebar.header("Demo controls")
    selected_pair = st.sidebar.selectbox("Select demo pair", pair_options)
    uploaded_audio = st.sidebar.file_uploader("Upload participant audio", type=["wav", "mp3", "m4a"])
    transcript_text = st.sidebar.text_area("Transcript", demo_manifest["transcript"], height=100)

    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.read()
        audio_path = project_root / "data" / "uploaded_audio.wav"
        audio_path.write_bytes(audio_bytes)
        demo_manifest["participant_audio"] = str(audio_path)
        demo_manifest["transcript"] = transcript_text
        demo_manifest["pair_id"] = "uploaded-demo"

    pair = demo_manifest
    result = _evaluate_pair(pair)

    col_a, col_b = st.columns([1.4, 1.2])
    with col_a:
        st.subheader("Evaluation summary")
        st.metric("Overall score", f"{result['score']:.2f}/100")
        st.metric("Detected flaw regions", result["summary"]["detected_regions"])
        st.metric("Average severity", f"{result['summary']['average_severity']:.2f}")
        st.write("Transcript")
        st.code(result["transcript"], language="text")

    with col_b:
        st.subheader("Audio playback")
        participant_audio = demo_manifest["participant_audio"]
        if participant_audio:
            st.audio(participant_audio)
        st.write("Evaluation evidence")
        export_results(result, project_root / "data" / "report.json")
        st.download_button(
            label="Download JSON report",
            data=json.dumps(result, indent=2),
            file_name="speechmirror_report.json",
            mime="application/json",
        )

    delta_df = pd.DataFrame(result["deltas"])
    st.subheader("Feature delta timeline")
    chart_df = delta_df[["time", "energy", "pitch", "speech_rate", "pause_rate"]].copy()
    chart_df = chart_df.rename(columns={"energy": "rms_delta", "pitch": "pitch_delta", "speech_rate": "tempo_delta", "pause_rate": "pause_delta"})
    st.line_chart(chart_df.set_index("time"))

    st.subheader("Detected flaws")
    if result["flaws"]:
        flaws_df = pd.DataFrame(result["flaws"])
        st.dataframe(flaws_df[["type", "feature", "start", "end", "severity", "explanation"]])
    else:
        st.info("No threshold-exceeding flaw windows were detected in this sample.")

    st.subheader("Rubric feedback")
    for flaw in result["flaws"][:5]:
        st.markdown(
            "- **{type}** ({start:.2f}s–{end:.2f}s): {explanation}".format(
                type=flaw["type"].replace("_", " ").title(),
                start=flaw["start"],
                end=flaw["end"],
                explanation=flaw["explanation"],
            )
        )


if __name__ == "__main__":
    main()
