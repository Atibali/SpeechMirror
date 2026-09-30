from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from backend.config import (
    DATABASE_PATH,
    DATASET_ROOT,
    MAX_UPLOAD_BYTES,
    RESULTS_ROOT,
    TEMP_ROOT,
    ensure_directories,
)
from backend.services.alignment import AlignmentError, AlignmentUnavailable, align_transcript_to_audio
from backend.services.audio_service import analyze_audio_pair, feature_records
from backend.services.dataset_service import get_record_by_id, load_manifest
from speechmirror.features import SAMPLE_RATE, load_audio

ensure_directories()

app = FastAPI(title="SpeechMirror API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _initialize_database() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS results ("
            "analysis_id TEXT PRIMARY KEY, result_hash TEXT NOT NULL UNIQUE, "
            "sample_id TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "payload TEXT NOT NULL)"
        )


_initialize_database()


def _save_result(result: dict) -> Path:
    output_path = RESULTS_ROOT / f"{result['analysis_id']}.json"
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO results "
            "(analysis_id, result_hash, sample_id, payload) VALUES (?, ?, ?, ?)",
            (
                result["analysis_id"],
                result["result_hash"],
                result["sample_id"],
                json.dumps(result, allow_nan=False),
            ),
        )
    return output_path


def _get_result(analysis_id: str) -> dict | None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        row = connection.execute(
            "SELECT payload FROM results WHERE analysis_id = ?", (analysis_id,)
        ).fetchone()
    return json.loads(row[0]) if row else None


def _record(sample_id: str):
    try:
        record = get_record_by_id(sample_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Dataset manifest is invalid: {exc}") from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset sample_id: {sample_id}")
    return record


async def _save_upload(audio: UploadFile | None) -> Path | None:
    if audio is None:
        return None
    if audio.size is not None and audio.size > MAX_UPLOAD_BYTES:
        await audio.close()
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a"}:
        await audio.close()
        raise HTTPException(status_code=415, detail="Audio must be WAV, MP3, or M4A.")
    if audio.content_type and audio.content_type not in {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/x-m4a",
        "application/octet-stream",
    }:
        await audio.close()
        raise HTTPException(status_code=415, detail="Unsupported uploaded audio content type.")

    temporary = tempfile.NamedTemporaryFile(
        mode="wb", suffix=suffix, prefix="speechmirror-", dir=TEMP_ROOT, delete=False
    )
    size = 0
    try:
        while chunk := await audio.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                )
            temporary.write(chunk)
    except Exception:
        temporary.close()
        Path(temporary.name).unlink(missing_ok=True)
        await audio.close()
        raise
    temporary.close()
    if size == 0:
        Path(temporary.name).unlink(missing_ok=True)
        await audio.close()
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    return Path(temporary.name)


def _baseline_transcript(record) -> str:
    try:
        return Path(record.transcript_file).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read dataset transcript: {exc}") from exc


def _map_pipeline_error(exc: Exception) -> HTTPException:
    if isinstance(exc, AlignmentUnavailable):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, AlignmentError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (OSError, RuntimeError)):
        return HTTPException(status_code=422, detail=f"Audio processing failed: {exc}")
    return HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


async def _analyze_request(
    sample_id: str, transcript: str, alignment_mode: str, audio: UploadFile | None
) -> dict:
    temporary_path = None
    try:
        record = _record(sample_id)
        expected_transcript = _baseline_transcript(record)
        input_transcript = " ".join((transcript or expected_transcript).split())
        if input_transcript != " ".join(expected_transcript.split()):
            raise HTTPException(
                status_code=422,
                detail="Transcript must exactly match the selected dataset baseline transcript.",
            )
        temporary_path = await _save_upload(audio)
        participant_path = temporary_path or Path(record.participant_audio)
        result = await run_in_threadpool(
            analyze_audio_pair,
            record.ideal_audio,
            participant_path,
            input_transcript,
            expected_transcript,
            record.sample_id,
            alignment_mode,
        )
        _save_result(result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise _map_pipeline_error(exc) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if audio is not None:
            await audio.close()


@app.get("/api/health")
def health() -> dict:
    try:
        records = load_manifest(DATASET_ROOT / "manifest.csv")
    except FileNotFoundError:
        records = []
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Dataset manifest is invalid: {exc}") from exc
    return {"status": "ok", "dataset_entries": len(records), "api_version": app.version}


@app.get("/api/config")
def get_config() -> dict:
    from speechmirror.evaluation import load_pipeline_config

    return load_pipeline_config()


@app.get("/api/dataset")
def get_dataset() -> dict:
    try:
        records = load_manifest(DATASET_ROOT / "manifest.csv")
        return {"records": [record.to_dict() for record in records]}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Dataset manifest is invalid: {exc}") from exc


@app.get("/api/dataset/{sample_id}")
def get_dataset_record(sample_id: str) -> dict:
    return _record(sample_id).to_dict()


@app.post("/api/analyze")
async def analyze(
    baseline_id: Annotated[str, Form()],
    transcript: Annotated[str, Form()],
    alignment_mode: Annotated[str, Form()] = "whisperx",
    audio: Annotated[UploadFile | None, File()] = None,
) -> dict:
    return await _analyze_request(baseline_id, transcript, alignment_mode, audio)


@app.post("/api/compare")
async def compare(
    baseline_id: Annotated[str, Form()],
    transcript: Annotated[str, Form()],
    alignment_mode: Annotated[str, Form()] = "whisperx",
    audio: Annotated[UploadFile | None, File()] = None,
) -> dict:
    return await _analyze_request(baseline_id, transcript, alignment_mode, audio)


@app.post("/api/align")
async def align(
    transcript: Annotated[str, Form()],
    audio: Annotated[UploadFile, File()],
    alignment_mode: Annotated[str, Form()] = "whisperx",
) -> dict:
    temporary_path = await _save_upload(audio)
    if temporary_path is None:
        raise HTTPException(status_code=400, detail="Audio upload is required.")
    try:
        signal, sr = await run_in_threadpool(load_audio, temporary_path, SAMPLE_RATE)
        words, method = await run_in_threadpool(
            align_transcript_to_audio,
            signal,
            sr,
            transcript,
            method=alignment_mode,
        )
        return {"alignment_method": method, "words": words}
    except Exception as exc:
        raise _map_pipeline_error(exc) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
        await audio.close()


@app.post("/api/features")
async def features(
    sample_id: Annotated[str, Form()],
    role: Annotated[str, Form()] = "participant",
    audio: Annotated[UploadFile | None, File()] = None,
) -> dict:
    if role not in {"ideal", "participant"}:
        raise HTTPException(status_code=422, detail="role must be 'ideal' or 'participant'.")
    record = _record(sample_id)
    temporary_path = await _save_upload(audio)
    path = temporary_path or Path(
        record.ideal_audio if role == "ideal" else record.participant_audio
    )
    try:
        rows = await run_in_threadpool(feature_records, path)
        return {"sample_id": sample_id, "role": role, "features": rows}
    except Exception as exc:
        raise _map_pipeline_error(exc) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if audio is not None:
            await audio.close()


@app.get("/api/result/{analysis_id}")
def get_result(analysis_id: str) -> dict:
    result = _get_result(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown analysis_id: {analysis_id}")
    return result


@app.get("/api/audio/{sample_id}")
def get_audio(
    sample_id: str,
    role: Annotated[str, Query()] = "participant",
) -> FileResponse:
    if role not in {"ideal", "participant"}:
        raise HTTPException(status_code=422, detail="role must be 'ideal' or 'participant'.")
    record = _record(sample_id)
    path = Path(record.ideal_audio if role == "ideal" else record.participant_audio)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file was not found.")
    media_types = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}
    return FileResponse(
        path,
        media_type=media_types[path.suffix.lower()],
        filename=f"{sample_id}-{role}{path.suffix}",
    )


@app.get("/api/manifest")
def get_manifest() -> dict:
    manifest_path = DATASET_ROOT / "manifest.csv"
    if not manifest_path.exists():
        return {"entries": []}
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {"entries": list(csv.DictReader(handle))}
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read dataset manifest: {exc}") from exc
