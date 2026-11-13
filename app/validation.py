"""CSV parsing and validation for follow-up risk modelling.

Required columns:
    subject_id  repeated-visit identifier; all rows of one subject stay together
    outcome     exactly 0 or 1

Every other column is treated as a numeric feature. Empty cells and the usual
missing-value tokens are accepted and passed to the fold-local imputer.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SUBJECT_COLUMN = "subject_id"
OUTCOME_COLUMN = "outcome"
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", ".", "?"}
MAX_ROWS = 50_000
MAX_COLUMNS = 200


class CsvValidationError(ValueError):
    """A user-facing CSV validation problem."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "INVALID_CSV",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class FollowupDataset:
    frame: pd.DataFrame  # subject_id (str), outcome (int), numeric feature columns
    subject_ids: np.ndarray
    y: np.ndarray
    X: pd.DataFrame
    feature_names: list[str]
    original_filename: str

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    @property
    def n_subjects(self) -> int:
        return int(pd.unique(self.subject_ids).size)

    @property
    def n_missing_feature_values(self) -> int:
        return int(self.X.isna().sum().sum())

    def summary(self) -> dict[str, Any]:
        values, counts = np.unique(self.y, return_counts=True)
        return {
            "n_rows": self.n_rows,
            "n_subjects": self.n_subjects,
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "n_missing_feature_values": self.n_missing_feature_values,
            "outcome_counts": {
                "0": int(counts[list(values).index(0)]) if 0 in values else 0,
                "1": int(counts[list(values).index(1)]) if 1 in values else 0,
            },
        }


def _normalise_header(raw_header: list[str]) -> list[str]:
    headers = [str(h).strip() for h in raw_header]
    if not headers or all(not h for h in headers):
        raise CsvValidationError("CSV 缺少表头。", code="EMPTY_CSV")
    if any(not h for h in headers):
        raise CsvValidationError(
            "CSV 表头中存在空列名。",
            code="EMPTY_COLUMN_NAME",
            details={"headers": headers},
        )
    duplicates = sorted({h for h in headers if headers.count(h) > 1})
    if duplicates:
        raise CsvValidationError(
            "CSV 表头存在重复列名。",
            code="DUPLICATE_COLUMN",
            details={"duplicate_columns": duplicates},
        )
    missing_required = [
        name for name in (SUBJECT_COLUMN, OUTCOME_COLUMN) if name not in headers
    ]
    if missing_required:
        raise CsvValidationError(
            "CSV 必须包含 subject_id 与 outcome 两列。",
            code="MISSING_REQUIRED_COLUMN",
            details={
                "missing_columns": missing_required,
                "received_columns": headers,
            },
        )
    return headers


def _row_examples(mask: pd.Series, limit: int = 5) -> list[int]:
    """Return one-based CSV data row numbers (header is line 1)."""
    return [int(i) + 2 for i in np.flatnonzero(mask.to_numpy())[:limit]]


def parse_followup_csv(raw: bytes, original_filename: str) -> FollowupDataset:
    if not raw:
        raise CsvValidationError("上传的 CSV 为空。", code="EMPTY_CSV")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CsvValidationError(
            "CSV 必须是 UTF-8 编码。",
            code="INVALID_ENCODING",
        ) from exc
    if not text.strip():
        raise CsvValidationError("上传的 CSV 只有空白字符。", code="EMPTY_CSV")

    # Use the csv module to validate the header and catch duplicate names that
    # pandas would silently rename (for example outcome.1).
    try:
        header = next(csv.reader(io.StringIO(text)))
    except (csv.Error, StopIteration) as exc:
        raise CsvValidationError(
            "无法解析 CSV 表头。",
            code="MALFORMED_CSV",
            details={"reason": str(exc)},
        ) from exc
    headers = _normalise_header(header)

    feature_names = [h for h in headers if h not in (SUBJECT_COLUMN, OUTCOME_COLUMN)]
    if not feature_names:
        raise CsvValidationError(
            "除 subject_id 与 outcome 外，至少需要一个数值特征列。",
            code="NO_FEATURE_COLUMN",
            details={"received_columns": headers},
        )
    if len(feature_names) > MAX_COLUMNS - 2:
        raise CsvValidationError(
            f"特征列过多（上限 {MAX_COLUMNS - 2} 个）。",
            code="TOO_MANY_COLUMNS",
            details={"n_features": len(feature_names)},
        )

    # Validate field counts before pandas: otherwise a ragged row silently
    # shifts values (or becomes all-NaN) and corrupts the outcome column.
    try:
        parsed_rows = [
            row
            for row in csv.reader(io.StringIO(text))
            if any(cell.strip() for cell in row)
        ]
    except csv.Error as exc:
        raise CsvValidationError(
            "CSV 数据区格式错误。",
            code="MALFORMED_CSV",
            details={"reason": str(exc)},
        ) from exc
    ragged = [
        {"line": line_number, "fields": len(row), "expected": len(headers)}
        for line_number, row in enumerate(parsed_rows[1:], start=2)
        if len(row) != len(headers)
    ]
    if ragged:
        raise CsvValidationError(
            "CSV 存在列数与表头不一致的行。",
            code="RAGGED_ROW",
            details={"expected_fields": len(headers), "rows": ragged[:5]},
        )

    try:
        frame = pd.read_csv(
            io.StringIO(text),
            dtype=str,
            keep_default_na=False,
            header=0,
            skip_blank_lines=True,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise CsvValidationError(
            "CSV 数据区格式错误。",
            code="MALFORMED_CSV",
            details={"reason": str(exc)},
        ) from exc

    frame.columns = headers
    if frame.empty:
        raise CsvValidationError("CSV 没有数据行。", code="NO_DATA_ROWS")
    if len(frame) > MAX_ROWS:
        raise CsvValidationError(
            f"数据行数超过上限（{MAX_ROWS} 行）。",
            code="TOO_MANY_ROWS",
            details={"n_rows": int(len(frame))},
        )
    frame = frame.reset_index(drop=True)

    # --- subject identifier -------------------------------------------------
    raw_subject = frame[SUBJECT_COLUMN]
    subject_missing = raw_subject.isna() | (
        raw_subject.astype("string").str.strip().fillna("") == ""
    )
    if subject_missing.any():
        raise CsvValidationError(
            "subject_id 不能为空。",
            code="MISSING_SUBJECT_ID",
            details={"rows": _row_examples(subject_missing)},
        )
    subjects = raw_subject.astype("string").str.strip().astype(str)
    if subjects.str.len().gt(256).any():
        raise CsvValidationError(
            "subject_id 过长（上限 256 字符）。",
            code="SUBJECT_ID_TOO_LONG",
        )

    # --- binary outcome -----------------------------------------------------
    raw_outcome = frame[OUTCOME_COLUMN].astype("string").str.strip()
    outcome_missing = raw_outcome.isna() | (raw_outcome.fillna("") == "")
    numeric_outcome = pd.to_numeric(raw_outcome, errors="coerce")
    outcome_invalid = ~outcome_missing & (
        numeric_outcome.isna()
        | ~np.isfinite(numeric_outcome)
        | ~numeric_outcome.isin([0, 1])
    )
    if outcome_missing.any() or outcome_invalid.any():
        raise CsvValidationError(
            "outcome 必须逐行为 0 或 1。",
            code="INVALID_OUTCOME",
            details={
                "missing_rows": _row_examples(outcome_missing),
                "invalid_rows": _row_examples(outcome_invalid),
            },
        )
    y_series = numeric_outcome.astype(int)

    # --- numeric features (missing cells are allowed) -----------------------
    feature_frame = pd.DataFrame(index=frame.index)
    for name in feature_names:
        raw_values = frame[name].astype("string").str.strip()
        missing = raw_values.isna() | raw_values.str.lower().isin(MISSING_TOKENS).fillna(False)
        numeric = pd.to_numeric(raw_values, errors="coerce")
        non_finite = (~missing) & (
            numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        )
        if non_finite.any():
            bad_rows = _row_examples(non_finite)
            examples = [
                str(raw_values.iloc[i - 2])
                for i in bad_rows
                if 0 <= i - 2 < len(raw_values)
            ][:3]
            raise CsvValidationError(
                f"特征列 {name!r} 含有非数值内容。",
                code="NON_NUMERIC_FEATURE",
                details={
                    "column": name,
                    "rows": bad_rows,
                    "examples": examples,
                },
            )
        feature_frame[name] = numeric.where(~missing, np.nan).astype(float)

    frame = pd.concat([subjects.rename(SUBJECT_COLUMN).to_frame(), y_series.rename(OUTCOME_COLUMN).to_frame(), feature_frame], axis=1)

    if len(frame) < 3:
        raise CsvValidationError(
            "三折验证至少需要 3 行数据。",
            code="TOO_FEW_ROWS",
            details={"n_rows": int(len(frame))},
        )
    grouped_counts = frame.groupby(SUBJECT_COLUMN, sort=False).size()
    if len(grouped_counts) < 3:
        raise CsvValidationError(
            "三折分组验证至少需要 3 个不同受试者。",
            code="TOO_FEW_SUBJECTS",
            details={"n_subjects": int(len(grouped_counts))},
        )
    outcome_values = sorted(int(v) for v in np.unique(y_series))
    if len(outcome_values) < 2:
        raise CsvValidationError(
            "outcome 必须同时包含 0 和 1，无法进行风险模型验证。",
            code="SINGLE_CLASS_DATASET",
            details={"outcome_counts": {str(int(v)): int((y_series == v).sum()) for v in outcome_values}},
        )

    return FollowupDataset(
        frame=frame,
        subject_ids=subjects.to_numpy(dtype=object),
        y=y_series.to_numpy(dtype=int),
        X=feature_frame,
        feature_names=feature_names,
        original_filename=original_filename,
    )
