"""Grouped three-fold logistic-regression validation.

The split keeps every subject in exactly one fold.  For each fold the
imputer, the scaler and the model are fitted on that fold's *training* rows
only; the fitted objects then transform/predict the validation rows.  This
module contains no I/O and is called from a worker thread.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .validation import FollowupDataset

N_SPLITS = 3
DEFAULT_SEED = 42
POSITIVE_CLASS = 1
MAX_DEMO_DELAY_SECONDS = 20.0


class JobCancelled(Exception):
    """Raised at a cancellation checkpoint; the worker then stops."""


class AnalysisFailure(Exception):
    """A domain failure that fails the whole batch (no report is produced)."""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _check_cancel(cancel_check: Callable[[], bool]) -> None:
    if cancel_check():
        raise JobCancelled()


def _delay_with_cancel(
    seconds: float, cancel_check: Callable[[], bool]
) -> None:
    """Synthetic demo workload that can be interrupted quickly.

    The sleep is sliced so a cancel request is observed promptly; the cancel
    endpoint still waits until this worker has actually left the loop.
    """
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        _check_cancel(cancel_check)
        time.sleep(min(0.1, remaining))


def _outcome_counts(y: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    values = y[indices]
    return {
        "0": int(np.sum(values == 0)),
        "1": int(np.sum(values == 1)),
    }


def _make_grouped_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    progress: Callable[..., None],
    cancel_check: Callable[[], bool],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic stratified group splits.

    A training fold containing only one outcome class makes the whole batch
    fail, as required.  All folds are checked before any fitting starts.
    """
    progress(stage="splitting")
    class_counts = {
        str(value): int(np.sum(y == value)) for value in np.unique(y)
    }
    if np.unique(groups).size < N_SPLITS:
        raise AnalysisFailure(
            "SPLIT_NOT_FEASIBLE",
            "受试者数量不足以分成三折。",
            {"n_subjects": int(np.unique(groups).size), "n_splits": N_SPLITS},
        )
    if min(class_counts.values()) < N_SPLITS:
        raise AnalysisFailure(
            "SPLIT_NOT_FEASIBLE",
            "某一结局类别的行数少于三折数，无法做分层分组划分。",
            {
                "n_splits": N_SPLITS,
                "outcome_row_counts": class_counts,
                "hint": "每个结局类别至少需要 3 行，且需分布在足够多的受试者中。",
            },
        )
    try:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=seed
        )
        splits = list(splitter.split(X, y, groups))
    except ValueError as exc:
        raise AnalysisFailure(
            "SPLIT_NOT_FEASIBLE",
            "无法在保证同一受试者不跨折的前提下生成三折。",
            {"reason": str(exc), "n_subjects": int(np.unique(groups).size)},
        ) from exc
    if len(splits) != N_SPLITS:
        raise AnalysisFailure(
            "SPLIT_NOT_FEASIBLE",
            "未能生成三折分组划分。",
            {"n_splits_created": len(splits)},
        )

    invalid_folds: list[dict[str, Any]] = []
    for fold_number, (train_idx, validation_idx) in enumerate(splits, start=1):
        _check_cancel(cancel_check)
        train_counts = _outcome_counts(y, train_idx)
        if train_counts["0"] == 0 or train_counts["1"] == 0:
            invalid_folds.append(
                {
                    "fold": fold_number,
                    "train_rows": int(len(train_idx)),
                    "train_outcome_counts": train_counts,
                    "validation_rows": int(len(validation_idx)),
                }
            )
    if invalid_folds:
        raise AnalysisFailure(
            "TRAIN_FOLD_SINGLE_CLASS",
            "至少一个训练折仅含单一结局，整批验证失败。",
            {
                "message": "训练数据必须同时包含 0 和 1；本批未产生任何折外概率或 AUC。",
                "invalid_folds": invalid_folds,
            },
        )
    return splits


def _fit_and_evaluate_fold(
    fold_number: int,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    X_values: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seed: int,
    demo_delay_seconds: float,
    progress: Callable[..., None],
    cancel_check: Callable[[], bool],
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit the fold pipeline on training rows, then score validation rows."""

    progress(stage="fold_delay", fold=fold_number)
    _delay_with_cancel(demo_delay_seconds, cancel_check)

    _check_cancel(cancel_check)
    progress(stage="fold_preprocessing", fold=fold_number)
    X_train = X_values[train_idx]
    X_validation = X_values[validation_idx]

    # --- fit on training rows only -------------------------------------
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_imputed = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    _check_cancel(cancel_check)
    progress(stage="fold_fit", fold=fold_number)
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    fold_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train_scaled, y[train_idx])
    for warning in caught:
        fold_warnings.append(f"{warning.category.__name__}: {warning.message}")
    empty_training_columns = [
        name
        for name, column in zip(feature_names, X_train.T)
        if np.isnan(column).all()
    ]
    if empty_training_columns:
        fold_warnings.append(
            "训练折中以下特征全部缺失，按 0 填补且不携带模型信息: "
            + ", ".join(empty_training_columns)
        )

    _check_cancel(cancel_check)
    progress(stage="fold_predict", fold=fold_number)
    # --- apply the same training-fitted transforms to validation rows ---
    X_validation_imputed = imputer.transform(X_validation)
    X_validation_scaled = scaler.transform(X_validation_imputed)
    probabilities = model.predict_proba(X_validation_scaled)
    positive_column = int(np.flatnonzero(model.classes_ == POSITIVE_CLASS)[0])
    positive_probability = probabilities[:, positive_column]

    progress(stage="fold_evaluate", fold=fold_number)
    validation_values = y[validation_idx]
    validation_classes = np.unique(validation_values)
    if validation_classes.size < 2:
        only_value = int(validation_classes[0])
        auc: float | None = None
        auc_status = "not_defined_single_class"
        auc_reason = (
            f"验证折仅含单一结局（全部为 {only_value}），"
            "阳性概率的排序无法形成 ROC 曲线，因此 AUC 留空。"
        )
    else:
        auc = float(roc_auc_score(validation_values, positive_probability))
        auc_status = "computed"
        auc_reason = None

    record: dict[str, Any] = {
        "fold": fold_number,
        "train": {
            "row_count": int(len(train_idx)),
            "subject_count": None,  # filled by the caller from the shared split
            "outcome_counts": _outcome_counts(y, train_idx),
        },
        "validation": {
            "row_count": int(len(validation_idx)),
            "outcome_counts": _outcome_counts(y, validation_idx),
        },
        "auc": auc,
        "auc_status": auc_status,
        "auc_reason": auc_reason,
        "preprocessing": {
            "fit_scope": "training rows only",
            "imputer_strategy": "median",
            "imputer_statistics": {
                name: float(value)
                for name, value in zip(feature_names, imputer.statistics_)
            },
            "scaler_mean": {
                name: float(value)
                for name, value in zip(feature_names, scaler.mean_)
            },
            "scaler_scale": {
                name: float(value)
                for name, value in zip(feature_names, scaler.scale_)
            },
        },
        "model": {
            "type": "LogisticRegression",
            "solver": "lbfgs",
            "intercept": float(model.intercept_[0]),
            "coefficients": {
                name: float(value)
                for name, value in zip(feature_names, model.coef_[0])
            },
        },
        "warnings": fold_warnings,
        "row_indices": {
            "train": [int(i) for i in train_idx],
            "validation": [int(i) for i in validation_idx],
        },
    }
    return record, positive_probability


def run_grouped_validation(
    dataset: FollowupDataset,
    progress: Callable[..., None],
    cancel_check: Callable[[], bool],
    seed: int = DEFAULT_SEED,
    demo_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    """Run the full validation and return a JSON-serialisable report body."""

    if demo_delay_seconds < 0 or demo_delay_seconds > MAX_DEMO_DELAY_SECONDS:
        raise AnalysisFailure(
            "INVALID_DEMO_DELAY",
            f"demo_delay_seconds 必须在 0 到 {MAX_DEMO_DELAY_SECONDS} 之间。",
        )

    X_values = dataset.X.to_numpy(dtype=float)
    y = dataset.y
    groups = dataset.subject_ids
    if np.isnan(X_values).all():
        raise AnalysisFailure(
            "NO_OBSERVED_FEATURES",
            "所有特征值均为缺失，无法拟合风险模型。",
        )

    splits = _make_grouped_splits(
        X_values, y, groups, seed, progress, cancel_check
    )

    fold_records: list[dict[str, Any]] = []
    oof_probabilities = np.full(len(y), np.nan, dtype=float)
    oof_fold_assignment = np.full(len(y), -1, dtype=int)
    warnings_all: list[str] = []

    for fold_number, (train_idx, validation_idx) in enumerate(splits, start=1):
        _check_cancel(cancel_check)
        progress(
            stage="fold_start",
            fold=fold_number,
            folds_total=N_SPLITS,
            folds_completed=fold_number - 1,
        )
        record, probabilities = _fit_and_evaluate_fold(
            fold_number,
            train_idx,
            validation_idx,
            X_values,
            y,
            dataset.feature_names,
            seed,
            demo_delay_seconds,
            progress,
            cancel_check,
        )
        # subject counts are derived directly from the shared split arrays
        record["train"]["subject_count"] = int(
            np.unique(groups[train_idx]).size
        )
        record["validation"]["subject_count"] = int(
            np.unique(groups[validation_idx]).size
        )
        # invariant: same subject never spans train and validation
        record["group_leakage"] = bool(
            np.intersect1d(groups[train_idx], groups[validation_idx]).size > 0
        )
        fold_records.append(record)
        warnings_all.extend(record["warnings"])

        oof_probabilities[validation_idx] = probabilities
        oof_fold_assignment[validation_idx] = fold_number
        progress(
            stage="fold_completed",
            fold=fold_number,
            folds_total=N_SPLITS,
            folds_completed=fold_number,
        )
        _check_cancel(cancel_check)

    _check_cancel(cancel_check)
    progress(stage="finalizing", folds_total=N_SPLITS, folds_completed=N_SPLITS)

    if np.isnan(oof_probabilities).any() or (oof_fold_assignment < 0).any():
        raise AnalysisFailure(
            "INCOMPLETE_OOF_PREDICTIONS",
            "并非每一行都恰好属于一个验证折，无法生成完整折外概率。",
            {
                "rows_without_probability": int(
                    np.isnan(oof_probabilities).sum()
                )
            },
        )

    folds_without_auc = [
        fold["fold"]
        for fold in fold_records
        if fold["auc_status"] != "computed"
    ]
    # Supplementary only: the task asks for a per-fold AUC.
    pooled_auc: float | None = None
    if np.unique(y).size == 2:
        pooled_auc = float(roc_auc_score(y, oof_probabilities))

    oof_predictions = [
        {
            "row_index": row_index,
            "subject_id": str(groups[row_index]),
            "outcome": int(y[row_index]),
            "fold": int(oof_fold_assignment[row_index]),
            "positive_probability": float(oof_probabilities[row_index]),
        }
        for row_index in range(len(y))
    ]

    result: dict[str, Any] = {
        "dataset": dataset.summary(),
        "config": {
            "n_splits": N_SPLITS,
            "seed": seed,
            "group_column": "subject_id",
            "outcome_column": "outcome",
            "positive_class": POSITIVE_CLASS,
            "preprocessing_fit_scope": "per-fold training rows only",
            "pipeline": [
                {"step": "SimpleImputer", "strategy": "median"},
                {"step": "StandardScaler"},
                {
                    "step": "LogisticRegression",
                    "solver": "lbfgs",
                    "max_iter": 1000,
                },
            ],
            "demo_delay_seconds": demo_delay_seconds,
        },
        "folds": fold_records,
        "oof_predictions": oof_predictions,
        "supplementary_pooled_oof_auc": pooled_auc,
        "folds_with_undefined_auc": folds_without_auc,
        "warnings": warnings_all[:50],
        "auc_note": (
            "每折 AUC 使用该折验证行的折外阳性概率；"
            "验证折单一结局时 AUC 为 null 并附 auc_reason。"
        ),
    }
    return result
