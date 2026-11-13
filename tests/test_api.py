from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.manager import JobManager

DATA = Path(__file__).resolve().parent.parent / "data"


def csv_bytes(rows: list[list[object]], header: list[str]) -> bytes:
    buffer = io.StringIO()
    buffer.write(",".join(header) + "\n")
    for row in rows:
        buffer.write(",".join(str(value) for value in row) + "\n")
    return buffer.getvalue().encode("utf-8")


def simple_dataset(n_subjects: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(1, n_subjects + 1):
        latent = rng.normal()
        for visit in range(1, 3):
            rows.append(
                [
                    f"S{subject:03d}",
                    int(rng.random() < 1 / (1 + np.exp(-latent))),
                    round(float(rng.normal() + latent), 3),
                    round(float(rng.normal(0.5 * latent)), 3),
                ]
            )
    return csv_bytes(rows, ["subject_id", "outcome", "x1", "x2"])


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def submit(client, content: bytes, delay: float = 0.0, seed: int = 42, name: str = "demo.csv"):
    return client.post(
        "/api/v1/validations",
        files={"file": (name, content, "text/csv")},
        data={"demo_delay_seconds": str(delay), "seed": str(seed)},
    )


def wait_for_state(client, job_id: str, states: set[str], timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/validations/{job_id}")
        payload = response.json()
        if payload["state"] in states:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach {states}: {payload}")


# ------------------------------------------------------------------ CSV rules
@pytest.mark.parametrize(
    "content, code",
    [
        (b"", "EMPTY_CSV"),
        (b"subject_id,outcome,x\n", "NO_DATA_ROWS"),
        (b"subject_id,x\nA,1\nB,2\nC,3\n", "MISSING_REQUIRED_COLUMN"),
        (b"subject_id,outcome,x\nA,2,1\nB,1,2\nC,0,3\n", "INVALID_OUTCOME"),
        (b"subject_id,outcome,x\nA,0,1\nA,1,oops\nB,1,2\nC,0,3\n", "NON_NUMERIC_FEATURE"),
        (b"subject_id,outcome,x\nA,0,1\nA,1,2\nB,1,3\n", "TOO_FEW_SUBJECTS"),
        (b"subject_id,outcome,x\nA,0,1\nB,0,2\nC,0,3\n", "SINGLE_CLASS_DATASET"),
        (b"subject_id,subject_id,outcome\nA,A,0\n", "DUPLICATE_COLUMN"),
        (b"subject_id,outcome,x\n,0,1\nB,1,2\nC,0,3\n", "MISSING_SUBJECT_ID"),
        (b"subject_id,outcome,x\nA,0,1,99\nB,1,2\nC,0,3\n", "RAGGED_ROW"),
    ],
)
def test_invalid_csv_rejected(client, content, code):
    response = submit(client, content)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == code


def test_missing_feature_cells_are_accepted(client):
    content = csv_bytes(
        [
            ["S1", 0, "", 1.0],
            ["S2", 1, 2.0, ""],
            ["S3", 0, 3.0, 2.0],
            ["S4", 1, 4.0, 2.5],
            ["S5", 0, 5.0, 3.0],
            ["S6", 1, 6.0, 3.5],
            ["S7", 0, 7.0, 4.0],
            ["S8", 1, 8.0, 4.5],
        ],
        ["subject_id", "outcome", "x1", "x2"],
    )
    response = submit(client, content)
    assert response.status_code == 202, response.text
    job = wait_for_state(client, response.json()["job_id"], {"completed", "failed"})
    assert job["state"] == "completed", job


# ------------------------------------------------------------------- success
def test_demo_report_has_ordered_oof_and_fold_aucs(client):
    content = (DATA / "followup_demo.csv").read_bytes()
    response = submit(client, content)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    status = wait_for_state(client, job_id, {"completed", "failed"})
    assert status["state"] == "completed", status
    assert status["report_available"] is True

    report_response = client.get(f"/api/v1/validations/{job_id}/report")
    assert report_response.status_code == 200
    assert report_response.headers["content-type"].startswith("application/json")
    report = report_response.json()

    rows = content.decode().strip().splitlines()[1:]
    assert len(report["oof_predictions"]) == len(rows)
    assert [p["row_index"] for p in report["oof_predictions"]] == list(range(len(rows)))
    assert [p["subject_id"] for p in report["oof_predictions"]] == [
        line.split(",")[0] for line in rows
    ]
    for prediction in report["oof_predictions"]:
        assert 0.0 <= prediction["positive_probability"] <= 1.0
        assert prediction["fold"] in (1, 2, 3)

    assert len(report["folds"]) == 3
    for fold in report["folds"]:
        assert fold["auc_status"] == "computed"
        assert 0.0 <= fold["auc"] <= 1.0
        assert fold["group_leakage"] is False
        assert fold["train"]["outcome_counts"]["0"] > 0
        assert fold["train"]["outcome_counts"]["1"] > 0
        assert fold["preprocessing"]["fit_scope"] == "training rows only"

    # subject groups never straddle folds
    by_subject: dict[str, set[int]] = {}
    for prediction in report["oof_predictions"]:
        by_subject.setdefault(prediction["subject_id"], set()).add(prediction["fold"])
    assert all(len(folds) == 1 for folds in by_subject.values())

    oof_response = client.get(f"/api/v1/validations/{job_id}/oof.csv")
    assert oof_response.status_code == 200
    assert len(oof_response.text.strip().splitlines()) == len(rows) + 1


def test_validation_single_class_fold_has_null_auc_but_succeeds(client):
    content = (DATA / "followup_validation_single.csv").read_bytes()
    response = submit(client, content)
    job_id = response.json()["job_id"]
    status = wait_for_state(client, job_id, {"completed", "failed"})
    assert status["state"] == "completed", status
    summary = status["result_summary"]
    statuses = {fold["fold"]: fold for fold in summary["fold_aucs"]}
    assert statuses[2]["auc"] is None
    assert statuses[2]["auc_status"] == "not_defined_single_class"
    assert "单一结局" in statuses[2]["auc_reason"]
    assert summary["folds_with_undefined_auc"] == [2]
    assert statuses[1]["auc"] is not None and statuses[3]["auc"] is not None


def test_train_single_class_fails_whole_batch_without_report(client):
    content = (DATA / "followup_train_single.csv").read_bytes()
    response = submit(client, content)
    job_id = response.json()["job_id"]
    status = wait_for_state(client, job_id, {"completed", "failed"})
    assert status["state"] == "failed"
    assert status["failure"]["code"] == "TRAIN_FOLD_SINGLE_CLASS"
    assert status["report_available"] is False
    assert client.get(f"/api/v1/validations/{job_id}/report").status_code == 409
    assert client.get(f"/api/v1/validations/{job_id}/oof.csv").status_code == 409


# ---------------------------------------------------------- preprocessing fit
def test_preprocessing_is_fitted_on_training_rows_only(client):
    content = (DATA / "followup_demo.csv").read_bytes()
    response = submit(client, content)
    job_id = response.json()["job_id"]
    wait_for_state(client, job_id, {"completed", "failed"})
    report = client.get(f"/api/v1/validations/{job_id}/report").json()

    import pandas as pd

    frame = pd.read_csv(io.BytesIO(content))
    feature_names = report["dataset"]["feature_names"]
    for fold in report["folds"]:
        train_rows = fold["row_indices"]["train"]
        train = frame.iloc[train_rows][feature_names]
        expected_medians = train.median()
        expected_means = train.fillna(expected_medians).mean()
        for name in feature_names:
            assert fold["preprocessing"]["imputer_statistics"][name] == pytest.approx(
                float(expected_medians[name])
            )
            assert fold["preprocessing"]["scaler_mean"][name] == pytest.approx(
                float(expected_means[name])
            )


# ------------------------------------------------------------ capacity rules
def test_third_batch_rejected_and_capacity_reported(client):
    content = (DATA / "followup_slow_demo.csv").read_bytes()
    first = submit(client, content, delay=5)
    second = submit(client, content, delay=5)
    try:
        assert first.status_code == 202, first.text
        assert second.status_code == 202, second.text
        third = submit(client, content, delay=5)
        assert third.status_code == 429
        assert third.json()["error"]["code"] == "CAPACITY_EXCEEDED"
        assert third.headers["retry-after"] == "5"

        health = client.get("/api/v1/health").json()
        assert health["capacity"]["active_slots"] == 2
        assert health["capacity"]["available_slots"] == 0
    finally:
        client.post(f"/api/v1/validations/{first.json()['job_id']}/cancel")
        client.post(f"/api/v1/validations/{second.json()['job_id']}/cancel")


# --------------------------------------------------------------- cancel rules
def test_cancel_waits_until_stopped_and_frees_slot_while_other_continues(client):
    content = (DATA / "followup_slow_demo.csv").read_bytes()
    doomed = submit(client, content, delay=15).json()["job_id"]
    survivor = submit(client, content, delay=1).json()["job_id"]
    time.sleep(0.4)

    started = time.monotonic()
    cancel = client.post(f"/api/v1/validations/{doomed}/cancel")
    elapsed = time.monotonic() - started
    assert cancel.status_code == 200, cancel.text
    body = cancel.json()
    assert body["state"] == "cancelled"
    assert body["capacity"]["active_slots"] == 1
    # The request must not return before the worker actually left the loop.
    assert elapsed >= 0.05

    doomed_status = client.get(f"/api/v1/validations/{doomed}").json()
    assert doomed_status["state"] == "cancelled"
    assert doomed_status["cancel_confirmed_at"]
    assert doomed_status["report_available"] is False

    # the other batch keeps running and finishes normally
    time.sleep(0.3)
    survivor_status = client.get(f"/api/v1/validations/{survivor}").json()
    assert survivor_status["state"] in {"running", "completed"}
    final = wait_for_state(client, survivor, {"completed", "failed", "cancelled"})
    assert final["state"] == "completed"

    # the freed slot accepts a new batch
    again = submit(client, content)
    assert again.status_code == 202


def test_cancel_after_completion_keeps_completed_state(client):
    job = submit(client, simple_dataset()).json()["job_id"]
    wait_for_state(client, job, {"completed", "failed"})
    response = client.post(f"/api/v1/validations/{job}/cancel")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CANCEL_NOT_ALLOWED"
    status = client.get(f"/api/v1/validations/{job}").json()
    assert status["state"] == "completed"


def test_cancelled_job_never_exposes_report(client):
    content = (DATA / "followup_slow_demo.csv").read_bytes()
    job = submit(client, content, delay=10).json()["job_id"]
    time.sleep(0.3)
    assert client.post(f"/api/v1/validations/{job}/cancel").status_code == 200
    time.sleep(0.8)  # allow any late worker result to (not) appear
    status = client.get(f"/api/v1/validations/{job}").json()
    assert status["state"] == "cancelled"
    assert status["report_available"] is False
    assert client.get(f"/api/v1/validations/{job}/report").status_code == 409


def test_cancel_and_completion_race_has_single_terminal_state():
    """Drive many jobs with a tiny delay straight at the finish line."""
    from app.validation import parse_followup_csv

    dataset = parse_followup_csv(simple_dataset(60), "race.csv")
    manager = JobManager(max_concurrent=2)
    outcomes = []
    try:
        for _ in range(8):
            while manager.capacity()["available_slots"] == 0:
                time.sleep(0.005)
            job = manager.submit(dataset, seed=42, demo_delay_seconds=0.02)
            time.sleep(0.05)
            try:
                manager.cancel(job.job_id)
                outcomes.append("cancelled")
            except Exception:
                with manager._lock:
                    outcomes.append(job.state)
        time.sleep(0.5)
        for job in manager.list_jobs():
            assert job.state in {"cancelled", "completed", "failed"}
            if job.state == "cancelled":
                assert job.result is None and job.report_bytes is None
            if job.state == "completed":
                assert job.result is not None and job.report_bytes is not None
        assert "completed" in outcomes or "cancelled" in outcomes
    finally:
        manager.shutdown()


def test_cancel_wins_race_against_late_result(monkeypatch):
    """A cancel requested before publication discards the finished result."""
    import app.manager as manager_module
    from app.validation import parse_followup_csv

    entered = threading.Event()
    release = threading.Event()

    def blocked_validation(dataset, progress, cancel_check, seed, demo_delay_seconds):
        entered.set()
        release.wait(timeout=10)
        return {
            "dataset": dataset.summary(),
            "folds": [],
            "oof_predictions": [],
            "supplementary_pooled_oof_auc": None,
            "folds_with_undefined_auc": [],
            "warnings": [],
            "config": {},
        }

    monkeypatch.setattr(manager_module, "run_grouped_validation", blocked_validation)
    dataset = parse_followup_csv(simple_dataset(20), "race.csv")
    manager = JobManager(max_concurrent=2)
    try:
        job = manager.submit(dataset, seed=42, demo_delay_seconds=0.0)
        assert entered.wait(timeout=10)

        cancel_result: dict = {}

        def do_cancel():
            try:
                cancel_result["job"] = manager.cancel(job.job_id)
            except Exception as exc:  # pragma: no cover - asserted below
                cancel_result["error"] = exc

        thread = threading.Thread(target=do_cancel)
        thread.start()
        deadline = time.monotonic() + 5
        while not job.cancel_requested and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.cancel_requested

        release.set()  # worker now returns a full result and tries to publish
        thread.join(timeout=10)
        assert "error" not in cancel_result
        assert cancel_result["job"].state == "cancelled"
        assert job.state == "cancelled"
        assert job.result is None and job.report_bytes is None
        assert job.cancel_confirmed_at is not None
    finally:
        manager.shutdown()


def test_completion_wins_race_against_cancel(monkeypatch):
    """If publication happens first, cancellation is refused and result stays."""
    import app.manager as manager_module
    from app.validation import parse_followup_csv

    def instant_validation(dataset, progress, cancel_check, seed, demo_delay_seconds):
        return {
            "dataset": dataset.summary(),
            "folds": [],
            "oof_predictions": [],
            "supplementary_pooled_oof_auc": None,
            "folds_with_undefined_auc": [],
            "warnings": [],
            "config": {},
        }

    monkeypatch.setattr(manager_module, "run_grouped_validation", instant_validation)
    dataset = parse_followup_csv(simple_dataset(20), "race.csv")
    manager = JobManager(max_concurrent=2)
    try:
        job = manager.submit(dataset, seed=42, demo_delay_seconds=0.0)
        assert job.done_event.wait(timeout=10)
        with pytest.raises(Exception) as excinfo:
            manager.cancel(job.job_id)
        assert type(excinfo.value).__name__ == "CancelNotAllowed"
        assert job.state == "completed"
        assert job.report_bytes is not None
    finally:
        manager.shutdown()


def test_cancel_request_blocks_until_worker_leaves_computation(monkeypatch):
    """Cancel confirmation must wait for a long, uninterruptible step to end."""
    import app.manager as manager_module
    from app.validation import parse_followup_csv

    def slow_step(dataset, progress, cancel_check, seed, demo_delay_seconds):
        # Simulate a single long compute step that cannot observe the cancel
        # flag until it returns; the endpoint must wait for it.
        time.sleep(0.8)
        return {
            "dataset": dataset.summary(),
            "folds": [],
            "oof_predictions": [],
            "supplementary_pooled_oof_auc": None,
            "folds_with_undefined_auc": [],
            "warnings": [],
            "config": {},
        }

    monkeypatch.setattr(manager_module, "run_grouped_validation", slow_step)
    dataset = parse_followup_csv(simple_dataset(20), "barrier.csv")
    manager = JobManager(max_concurrent=2)
    try:
        job = manager.submit(dataset, seed=42, demo_delay_seconds=0.0)
        time.sleep(0.1)  # worker is inside the long step
        started = time.monotonic()
        cancelled = manager.cancel(job.job_id)
        elapsed = time.monotonic() - started
        assert cancelled.state == "cancelled"
        assert elapsed >= 0.5, f"cancel returned too early: {elapsed:.3f}s"
        assert manager.capacity()["active_slots"] == 0
        assert job.result is None and job.report_bytes is None
    finally:
        manager.shutdown()


def test_unknown_job_returns_404(client):
    assert client.get("/api/v1/validations/nope").status_code == 404
    assert client.post("/api/v1/validations/nope/cancel").status_code == 404
    assert client.get("/api/v1/validations/nope/report").status_code == 404


def test_report_is_not_a_file_on_disk(client, tmp_path):
    content = (DATA / "followup_demo.csv").read_bytes()
    job = submit(client, content).json()["job_id"]
    wait_for_state(client, job, {"completed", "failed"})
    # proved structurally: the manager only holds bytes in memory
    assert not list(Path.cwd().glob(f"**/*{job}*"))


def test_openapi_and_swagger_available(client):
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert "/api/v1/validations" in paths
    assert "/api/v1/validations/{job_id}/cancel" in paths
    assert "/api/v1/validations/{job_id}/report" in paths
    assert client.get("/docs").status_code == 200
