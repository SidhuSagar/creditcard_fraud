from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

ScoreType = str  # keep flexible for generic

class ClassifiedRow(BaseModel):
    row: int
    amount: float
    prediction: int
    label: str
    score: float
    actual_class: int | None = None


class RejectedRow(BaseModel):
    row: int
    reason: str


class AnalyzeResponse(BaseModel):
    filename: str
    model: str
    model_display_name: str
    mode: str = "supervised"
    mode_label: str | None = None
    mode_note: str | None = None
    score_type: str = "probability"
    threshold: float = 0.0

    total_rows: int
    processed_rows: int
    rejected_rows: int
    fraud_rows: int
    legitimate_rows: int

    results: list[ClassifiedRow]
    results_truncated: bool = False
    rejected: list[RejectedRow] = []
    rejected_truncated: bool = False

    evaluation: dict[str, Any] | None = None
    disclaimer: str = (
        "This is an estimate for educational purposes only; it is not proof of fraud, "
        "and no real transaction is blocked."
    )
