from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetRecordSchema(BaseModel):
    sample_id: str
    baseline_id: str
    speaker_id: str
    transcript_file: str
    ideal_audio: str
    participant_audio: str
    flaw_type: str
    severity: float = Field(ge=0, le=4)
    notes: str = ""
    labels_path: str | None = None
    source_license: str = ""
    split: str = ""


class FlawSchema(BaseModel):
    flaw_id: str
    feature: str
    type: str
    dimension: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    severity: float = Field(ge=0, le=1)
    evidence: dict[str, float]
    explanation: str
    action: str
    transcript: str


class AnalyzeResponse(BaseModel):
    analysis_id: str
    result_hash: str
    sample_id: str
    overall_score: float = Field(ge=0, le=100)
    transcript: str
    dimensions: dict[str, float]
    flaws: list[FlawSchema]
    alignment: list[dict]
    alignment_method: str
    word_evidence: list[dict]
    summary: dict
    reproducibility: dict
