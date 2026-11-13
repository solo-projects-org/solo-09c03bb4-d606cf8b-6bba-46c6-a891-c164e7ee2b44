"""In-memory job manager for validation batches.

Design constraints implemented here:

* At most two batches train concurrently anywhere in the process.  A third
  submission is rejected immediately; there is no queue.
* A slot stays occupied while a job is being cancelled, until its worker
  thread has actually left the computation.
* A job has exactly one terminal state.  Results are published under the same
  lock that finalises the state, so a cancellation that wins the race can
  never be followed by a late result.
* Nothing is persisted: jobs and reports live only in this process.
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .analysis import (
    N_SPLITS,
    AnalysisFailure,
    JobCancelled,
    run_grouped_validation,
)
from .validation import FollowupDataset

MAX_CONCURRENT_JOBS = 2
MAX_RETAINED_TERMINAL_JOBS = 200
CANCEL_CONFIRM_TIMEOUT_SECONDS = 60.0

RUNNING = "running"
CANCELLING = "cancelling"
CANCELLED = "cancelled"
COMPLETED = "completed"
FAILED = "failed"

TERMINAL_STATES = frozenset({CANCELLED, COMPLETED, FAILED})
ACTIVE_STATES = frozenset({RUNNING, CANCELLING})

STAGE_LABELS = {
    "accepted": "已受理，等待计算线程",
    "splitting": "按受试者分组划分三折",
    "fold_start": "开始折",
    "fold_delay": "演示用慢计算",
    "fold_preprocessing": "拟合训练折的缺值填补与标准化",
    "fold_fit": "拟合逻辑回归",
    "fold_predict": "预测验证折阳性概率",
    "fold_evaluate": "计算折 AUC",
    "fold_completed": "折完成",
    "finalizing": "汇总折外结果",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class CapacityExceeded(Exception):
    def __init__(self, active_jobs: int, max_jobs: int) -> None:
        super().__init__("training capacity is full")
        self.active_jobs = active_jobs
        self.max_jobs = max_jobs


class JobNotFound(Exception):
    pass


class CancelNotAllowed(Exception):
    def __init__(self, state: str) -> None:
        super().__init__(f"job is already {state}")
        self.state = state


class CancelConfirmationTimeout(Exception):
    pass


@dataclass
class JobRecord:
    job_id: str
    dataset: FollowupDataset
    seed: int
    demo_delay_seconds: float
    created_at: str = field(default_factory=_utc_now)

    state: str = RUNNING
    cancel_requested: bool = False
    cancel_requested_at: str | None = None
    cancel_confirmed_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    progress: dict[str, Any] = field(
        default_factory=lambda: {
            "stage": "accepted",
            "stage_label": STAGE_LABELS["accepted"],
            "fold": None,
            "folds_completed": 0,
            "folds_total": N_SPLITS,
            "percent": 0,
        }
    )
    result: dict[str, Any] | None = None
    report_bytes: bytes | None = None
    failure: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    # runtime-only synchronisation primitives
    cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    done_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )


class JobManager:
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_JOBS) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._active_slots = 0
        self._max_concurrent = max_concurrent
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_concurrent,
            thread_name_prefix="validation",
        )

    # ------------------------------------------------------------------ API
    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    def capacity(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_concurrent": self._max_concurrent,
                "active_slots": self._active_slots,
                "available_slots": self._max_concurrent - self._active_slots,
            }

    def submit(
        self,
        dataset: FollowupDataset,
        *,
        seed: int,
        demo_delay_seconds: float,
    ) -> JobRecord:
        with self._lock:
            if self._closed:
                raise RuntimeError("job manager is shutting down")
            if self._active_slots >= self._max_concurrent:
                raise CapacityExceeded(self._active_slots, self._max_concurrent)
            self._active_slots += 1
            job = JobRecord(
                job_id=uuid.uuid4().hex,
                dataset=dataset,
                seed=seed,
                demo_delay_seconds=demo_delay_seconds,
            )
            job.started_at = _utc_now()
            self._jobs[job.job_id] = job
            self._prune_terminal_jobs_locked()
        try:
            self._executor.submit(self._run_job, job)
        except Exception as exc:  # pragma: no cover - defensive
            self._finalise(
                job,
                state=FAILED,
                failure={
                    "code": "WORKER_START_FAILED",
                    "message": "无法启动计算线程。",
                    "details": {"reason": str(exc)},
                },
            )
        return job

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def cancel(self, job_id: str) -> JobRecord:
        """Request cancellation and wait for the computation to stop.

        Returns only after the worker has released its slot.  If completion or
        failure already won the race, the final state is left untouched and
        CancelNotAllowed is raised by the caller-facing wrapper.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(job_id)
            if job.state in TERMINAL_STATES:
                raise CancelNotAllowed(job.state)
            if not job.cancel_requested:
                job.cancel_requested = True
                job.cancel_requested_at = _utc_now()
                job.state = CANCELLING
                job.progress = {
                    **job.progress,
                    "stage": "cancelling",
                    "stage_label": "取消请求已受理，等待计算停止",
                }
                job.cancel_event.set()

        # Wait outside the lock: the worker needs the lock to publish its
        # terminal state and release the slot.  Other jobs are unaffected.
        if not job.done_event.wait(timeout=CANCEL_CONFIRM_TIMEOUT_SECONDS):
            raise CancelConfirmationTimeout(job_id)

        with self._lock:
            # Cancelled is the only state this call can observe here, because
            # the worker checks cancel_event under the same lock before
            # publishing a result.  The guard keeps the invariant explicit.
            if job.state not in TERMINAL_STATES:
                raise CancelConfirmationTimeout(job_id)
            if job.state != CANCELLED:
                raise CancelNotAllowed(job.state)
            return job

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            for job in self._jobs.values():
                if job.state in ACTIVE_STATES and not job.cancel_requested:
                    job.cancel_requested = True
                    job.state = CANCELLING
                    job.cancel_event.set()
        for job in list(self._jobs.values()):
            if job.state in ACTIVE_STATES:
                job.done_event.wait(timeout=CANCEL_CONFIRM_TIMEOUT_SECONDS)
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------- internal
    def _prune_terminal_jobs_locked(self) -> None:
        terminal = [
            job for job in self._jobs.values() if job.state in TERMINAL_STATES
        ]
        if len(terminal) <= MAX_RETAINED_TERMINAL_JOBS:
            return
        terminal.sort(key=lambda job: job.finished_at or job.created_at)
        for job in terminal[: len(terminal) - MAX_RETAINED_TERMINAL_JOBS]:
            self._jobs.pop(job.job_id, None)

    def _update_progress(self, job: JobRecord, **update: Any) -> None:
        with self._lock:
            if job.state in TERMINAL_STATES or job.cancel_requested:
                # Keep the visible stage but never publish progress after a
                # cancellation request that could race with completion.
                return
            job.progress = {
                **job.progress,
                **update,
                "stage_label": STAGE_LABELS.get(
                    str(update.get("stage", job.progress["stage"])),
                    job.progress["stage_label"],
                ),
            }
            folds_completed = int(job.progress.get("folds_completed") or 0)
            job.progress["percent"] = round(
                100.0 * folds_completed / N_SPLITS, 1
            )

    def _run_job(self, job: JobRecord) -> None:
        try:
            if job.cancel_event.is_set():
                raise JobCancelled()
            result = run_grouped_validation(
                job.dataset,
                progress=lambda **update: self._update_progress(job, **update),
                cancel_check=job.cancel_event.is_set,
                seed=job.seed,
                demo_delay_seconds=job.demo_delay_seconds,
            )
            report = {
                "schema_version": "1.0",
                "job_id": job.job_id,
                "state": COMPLETED,
                "created_at": job.created_at,
                "completed_at": _utc_now(),
                "source_filename": job.dataset.original_filename,
                "requested": {
                    "seed": job.seed,
                    "demo_delay_seconds": job.demo_delay_seconds,
                },
                **result,
            }
            report_bytes = json.dumps(
                report, ensure_ascii=False, indent=2, allow_nan=False
            ).encode("utf-8")
            self._finalise(
                job,
                state=COMPLETED,
                result=report,
                report_bytes=report_bytes,
            )
        except JobCancelled:
            self._finalise(job, state=CANCELLED)
        except AnalysisFailure as exc:
            self._finalise(
                job,
                state=FAILED,
                failure={
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._finalise(
                job,
                state=FAILED,
                failure={
                    "code": "INTERNAL_ERROR",
                    "message": "验证过程中发生未预期错误。",
                    "details": {"type": type(exc).__name__, "reason": str(exc)},
                },
            )

    def _finalise(
        self,
        job: JobRecord,
        *,
        state: str,
        result: dict[str, Any] | None = None,
        report_bytes: bytes | None = None,
        failure: dict[str, Any] | None = None,
    ) -> None:
        """Atomically choose the single terminal state and release the slot."""
        with self._lock:
            if job.state in TERMINAL_STATES:
                return
            if job.cancel_requested:
                # Cancellation won the race: discard any late result.  Payload
                # fields are cleared before the state flips so readers that
                # observe the terminal state never see a half-published result.
                job.result = None
                job.report_bytes = None
                job.failure = None
                job.cancel_confirmed_at = _utc_now()
                job.state = CANCELLED
                job.progress = {
                    **job.progress,
                    "stage": "cancelled",
                    "stage_label": "已取消，计算已停止",
                    "percent": job.progress.get("percent", 0.0),
                }
            else:
                job.result = result
                job.report_bytes = report_bytes
                job.failure = failure
                job.state = state
                job.progress = {
                    **job.progress,
                    "stage": state,
                    "stage_label": {
                        COMPLETED: "验证完成",
                        FAILED: "验证失败",
                    }.get(state, job.progress["stage_label"]),
                    "percent": 100.0 if state == COMPLETED else job.progress.get("percent", 0.0),
                }
                if state == COMPLETED:
                    job.progress["folds_completed"] = N_SPLITS
            job.finished_at = _utc_now()
            self._active_slots = max(0, self._active_slots - 1)
            job.done_event.set()


# ---------------------------------------------------------------- serialise
def _progress_view(job: JobRecord) -> dict[str, Any]:
    return dict(job.progress)


def _fold_auc_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "fold": fold["fold"],
            "auc": fold["auc"],
            "auc_status": fold["auc_status"],
            "auc_reason": fold["auc_reason"],
            "validation_rows": fold["validation"]["row_count"],
            "validation_outcome_counts": fold["validation"]["outcome_counts"],
        }
        for fold in result.get("folds", [])
    ]


def job_status_payload(job: JobRecord, manager: JobManager) -> dict[str, Any]:
    capacity = manager.capacity()
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "state": job.state,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "source_filename": job.dataset.original_filename,
        "requested": {
            "seed": job.seed,
            "demo_delay_seconds": job.demo_delay_seconds,
        },
        "progress": _progress_view(job),
        "cancel_requested": job.cancel_requested,
        "cancel_requested_at": job.cancel_requested_at,
        "cancel_confirmed_at": job.cancel_confirmed_at,
        "capacity": capacity,
        "report_available": job.state == COMPLETED and job.report_bytes is not None,
        "report_url": (
            f"/api/v1/validations/{job.job_id}/report"
            if job.state == COMPLETED and job.report_bytes is not None
            else None
        ),
    }
    if job.state == COMPLETED and job.result is not None:
        payload["result_summary"] = {
            "dataset": job.result.get("dataset"),
            "config": job.result.get("config"),
            "fold_aucs": _fold_auc_summary(job.result),
            "folds_with_undefined_auc": job.result.get(
                "folds_with_undefined_auc", []
            ),
            "supplementary_pooled_oof_auc": job.result.get(
                "supplementary_pooled_oof_auc"
            ),
            "warnings": job.result.get("warnings", []),
        }
    if job.state == FAILED:
        payload["failure"] = job.failure
    return payload
