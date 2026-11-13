#!/usr/bin/env python3
"""Generate fictional follow-up data for the validation API demo.

All subject identifiers and values are synthetic; no real patient data is
involved.  Files produced (in ``data/``):

    followup_demo.csv               all folds mixed: 3 AUCs are computed
    followup_validation_single.csv  one validation fold is all-negative:
                                    its AUC is null while the batch succeeds
    followup_train_single.csv       two training folds are all-positive:
                                    the whole batch fails
    followup_slow_demo.csv          small dataset for the slow-training demo

The script searches a deterministic seed space and verifies every intended
split outcome with the same splitter used by the service, so the demo files
stay correct even if scikit-learn changes.

Run from the repository root:

    python3 scripts/generate_demo_data.py
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "ldl",
    "followup_gap_days",
    "medication_count",
]
HEADER = ["subject_id", "outcome", *FEATURES]
N_SPLITS = 3
SPLIT_SEED = 42  # must match the API default


def _feature_row(rng: random.Random, latent: float) -> list[float]:
    age = rng.gauss(61, 11)
    bmi = rng.gauss(26 + 0.4 * latent, 3.4)
    sbp = rng.gauss(132 + 3.5 * latent, 14)
    ldl = rng.gauss(2.9 + 0.25 * latent, 0.7)
    gap = max(7, int(rng.gauss(180 - 18 * latent, 55)))
    meds = max(0, int(round(rng.gauss(2.2 + 0.8 * latent, 1.0))))
    return [round(age, 1), round(bmi, 1), round(sbp, 1), round(ldl, 2), gap, meds]


def _render_row(
    rng: random.Random, row: list[float], *, missing_rate: float = 0.07
) -> list[str]:
    return ["" if rng.random() < missing_rate else str(v) for v in row]


def _draw_outcome(rng: random.Random, latent: float) -> int:
    logit = -1.35 + 1.6 * latent + rng.gauss(0, 0.3)
    return 1 if rng.random() < 1.0 / (1.0 + math.exp(-logit)) else 0


def _split(labels: list[int], groups: list[str]):
    X = np.zeros((len(labels), 1))
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED
    )
    return list(splitter.split(X, np.asarray(labels), np.asarray(groups)))


def _classes(y: np.ndarray) -> set[int]:
    return set(np.unique(y).tolist())


def build_mixed(rng: random.Random, n_subjects: int = 120):
    rows: list[list[str]] = []
    for index in range(1, n_subjects + 1):
        subject_id = f"SUBJ-{index:04d}"
        latent = rng.gauss(0, 1)
        for visit in range(1, rng.choice((1, 2, 3, 4)) + 1):
            feature = _feature_row(rng, latent + rng.gauss(0, 0.25))
            rows.append(
                [subject_id, str(_draw_outcome(rng, latent)), *_render_row(rng, feature)]
            )
    return rows


def build_rare_positive(rng: random.Random, n_negative_subjects: int = 60):
    """Exactly two positive subjects; all other subjects are negative.

    If the splitter puts the two positive subjects in different folds, the
    remaining fold's validation set is single-class while every training fold
    still contains both classes — precisely the validation-AUC-null case.
    """
    rows: list[list[str]] = []
    positive_subjects = {"POS-0001", "POS-0002"}
    for index in range(1, n_negative_subjects + 1):
        subject_id = f"NEG-{index:04d}"
        latent = rng.gauss(0, 1)
        for visit in range(1, rng.choice((1, 2, 3)) + 1):
            feature = _feature_row(rng, latent)
            rows.append([subject_id, "0", *_render_row(rng, feature)])
    for subject_id in sorted(positive_subjects):
        latent = rng.gauss(1.5, 0.2)
        for visit in range(1, 3):
            feature = _feature_row(rng, latent)
            rows.append([subject_id, "1", *_render_row(rng, feature)])
    rng.shuffle(rows)
    return rows


def build_train_single(rng: random.Random, n_subjects: int = 40):
    """All negatives live in one subject, so two training folds lack class 0."""
    rows: list[list[str]] = []
    for index in range(1, n_subjects + 1):
        subject_id = f"TRN-{index:04d}"
        positive = index != 1
        latent = rng.gauss(0.8 if positive else -0.8, 0.5)
        for visit in range(1, 4):
            feature = _feature_row(rng, latent)
            rows.append(
                [subject_id, "1" if positive else "0", *_render_row(rng, feature)]
            )
    return rows


def summarise(rows: list[list[str]], folds) -> dict:
    y = np.array([int(row[1]) for row in rows])
    groups = np.array([row[0] for row in rows])
    return {
        "rows": len(rows),
        "subjects": int(np.unique(groups).size),
        "positives": int(y.sum()),
        "train_classes": [_classes(y[train]) for train, _ in folds],
        "validation_classes": [_classes(y[validation]) for _, validation in folds],
    }


def validate_mixed(rows: list[list[str]]) -> bool:
    y = np.array([int(row[1]) for row in rows])
    groups = np.array([row[0] for row in rows])
    folds = _split(y.tolist(), groups.tolist())
    summary = summarise(rows, folds)
    return all(c == {0, 1} for c in summary["train_classes"]) and all(
        c == {0, 1} for c in summary["validation_classes"]
    )


def validate_rare_positive(rows: list[list[str]]) -> bool:
    y = np.array([int(row[1]) for row in rows])
    groups = np.array([row[0] for row in rows])
    folds = _split(y.tolist(), groups.tolist())
    summary = summarise(rows, folds)
    return (
        all(c == {0, 1} for c in summary["train_classes"])
        and summary["validation_classes"].count({0}) == 1
        and summary["validation_classes"].count({0, 1}) == 2
    )


def validate_train_single(rows: list[list[str]]) -> bool:
    y = np.array([int(row[1]) for row in rows])
    groups = np.array([row[0] for row in rows])
    folds = _split(y.tolist(), groups.tolist())
    summary = summarise(rows, folds)
    return summary["train_classes"].count({1}) >= 1 and any(
        c == {0, 1} for c in summary["train_classes"]
    )


def find_generation_seed(builder, predicate, start: int, attempts: int = 300):
    for seed in range(start, start + attempts):
        rng = random.Random(seed)
        rows = builder(rng)
        if predicate(rows):
            return seed, rows
    raise RuntimeError(f"no dataset satisfying the demo condition (from {start})")


def write_csv(path: Path, rows: list[list[str]], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    summary = summarise(
        rows,
        _split(
            [int(row[1]) for row in rows],
            [row[0] for row in rows],
        ),
    )
    print(
        f"{path.name}: {summary['rows']} rows, {summary['subjects']} subjects, "
        f"{summary['positives']} positive; "
        f"train={summary['train_classes']} validation={summary['validation_classes']}"
        f" — {note}"
    )


def generate(output_dir: Path) -> None:
    seed, mixed_rows = find_generation_seed(build_mixed, validate_mixed, 20261113)
    print(f"mixed dataset generation seed: {seed}")
    write_csv(output_dir / "followup_demo.csv", mixed_rows,
              "all folds mixed; three AUCs computed")

    seed, rare_rows = find_generation_seed(
        build_rare_positive, validate_rare_positive, 20261114
    )
    print(f"rare-positive dataset generation seed: {seed}")
    write_csv(output_dir / "followup_validation_single.csv", rare_rows,
              "one validation fold all-negative; its AUC is null")

    seed, single_rows = find_generation_seed(
        build_train_single, validate_train_single, 20261115
    )
    print(f"train-single dataset generation seed: {seed}")
    write_csv(output_dir / "followup_train_single.csv", single_rows,
              "training folds missing a class; the whole batch fails")

    rng = random.Random(20261116)
    slow_rows = build_mixed(rng, n_subjects=60)
    write_csv(output_dir / "followup_slow_demo.csv", slow_rows,
              "use demo_delay_seconds=8 to observe parallel runs and cancel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI feedback
        print(f"error: {exc}", file=sys.stderr)
        raise
