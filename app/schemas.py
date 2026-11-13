"""Pydantic models used by the OpenAPI schema (Swagger UI)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JobState = Literal["running", "cancelling", "cancelled", "completed", "failed"]


class CapacityInfo(BaseModel):
    max_concurrent: int = Field(description="全站同时训练批次上限")
    active_slots: int = Field(description="当前占用名额（含取消中的批次）")
    available_slots: int = Field(description="剩余可提交名额")


class ProgressInfo(BaseModel):
    stage: str = Field(description="当前计算阶段标识")
    stage_label: str = Field(description="当前阶段说明")
    fold: int | None = Field(default=None, description="正在处理的折号（1-3）")
    folds_completed: int = Field(description="已完成折数")
    folds_total: int = Field(description="总折数，固定为 3")
    percent: float = Field(description="按完成折数估算的进度百分比")


class FoldAucSummary(BaseModel):
    fold: int
    auc: float | None = Field(description="折外 AUC；验证折单一结局时为 null")
    auc_status: Literal["computed", "not_defined_single_class"]
    auc_reason: str | None = Field(description="AUC 留空的原因说明")
    validation_rows: int
    validation_outcome_counts: dict[str, int]


class DatasetSummary(BaseModel):
    n_rows: int
    n_subjects: int
    n_features: int
    feature_names: list[str]
    n_missing_feature_values: int
    outcome_counts: dict[str, int]


class ResultSummary(BaseModel):
    dataset: dict[str, Any]
    config: dict[str, Any]
    fold_aucs: list[FoldAucSummary]
    folds_with_undefined_auc: list[int]
    supplementary_pooled_oof_auc: float | None = None
    warnings: list[str] = Field(default_factory=list)


class FailureInfo(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RequestInfo(BaseModel):
    seed: int
    demo_delay_seconds: float


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    created_at: str
    started_at: str | None
    finished_at: str | None
    source_filename: str
    requested: RequestInfo
    progress: ProgressInfo
    cancel_requested: bool
    cancel_requested_at: str | None
    cancel_confirmed_at: str | None
    capacity: CapacityInfo
    report_available: bool
    report_url: str | None
    result_summary: ResultSummary | None = None
    failure: FailureInfo | None = None


class JobSubmissionResponse(BaseModel):
    job_id: str
    state: JobState
    status_url: str
    report_url: str | None
    cancel_url: str
    capacity: CapacityInfo
    message: str


class JobListResponse(BaseModel):
    jobs: list[JobStatus]
    capacity: CapacityInfo


class CancelResponse(BaseModel):
    job_id: str
    state: JobState
    message: str = Field(description="仅在本批计算线程实际停止后才返回")
    cancel_confirmed_at: str
    capacity: CapacityInfo


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    capacity: CapacityInfo
