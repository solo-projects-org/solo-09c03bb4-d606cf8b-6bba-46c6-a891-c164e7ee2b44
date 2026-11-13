"""FastAPI application: grouped three-fold follow-up risk validation."""

from __future__ import annotations

import csv
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .analysis import DEFAULT_SEED, MAX_DEMO_DELAY_SECONDS
from .manager import (
    ACTIVE_STATES,
    COMPLETED,
    MAX_CONCURRENT_JOBS,
    CancelConfirmationTimeout,
    CancelNotAllowed,
    CapacityExceeded,
    JobManager,
    JobNotFound,
    job_status_payload,
)
from .schemas import (
    CancelResponse,
    ErrorResponse,
    HealthResponse,
    JobListResponse,
    JobStatus,
    JobSubmissionResponse,
)
from .validation import CsvValidationError, parse_followup_csv

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
CSV_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",
    "application/octet-stream",
    "",
}


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details or {}}},
        headers=headers,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.manager = JobManager(max_concurrent=MAX_CONCURRENT_JOBS)
    yield
    app.state.manager.shutdown()


app = FastAPI(
    title="随访风险模型 · 三折分组验证 API",
    version=__version__,
    description=(
        "上传含受试者号（subject_id）、0/1 结局（outcome）和数值特征的 CSV，"
        "执行按受试者分组的 3 折逻辑回归验证。\n\n"
        "**关键约定**\n"
        "- 同一个受试者的历次随访必定分在同一折。\n"
        "- 每折的缺值填补与标准化只用该折训练行拟合，再变换验证行。\n"
        "- 按原始行序返回折外阳性概率；每折给出 AUC。\n"
        "- 任一训练折结局单一：整批失败，不产出报告。\n"
        "- 验证折结局单一：该折 AUC 为 null，并附原因。\n"
        "- 全站最多同时训练 2 批，超额返回 429；任务仅存内存。\n"
        "- 取消请求会等待该批计算真正停止后才返回并释放名额。\n\n"
        "演示请在 Swagger 的 `POST /api/v1/validations` 上传 `data/` 下的虚构随访 CSV。"
    ),
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
            headers=getattr(exc, "headers", None),
        )
    return error_response(
        exc.status_code,
        "HTTP_ERROR",
        str(exc.detail),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return error_response(
        422,
        "REQUEST_VALIDATION_ERROR",
        "请求参数不合法。",
        {"errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return error_response(
        500,
        "INTERNAL_ERROR",
        "服务器内部错误。",
        {"type": type(exc).__name__},
    )


def manager_of(request: Request) -> JobManager:
    return request.app.state.manager


def _capacity_dict(manager: JobManager) -> dict[str, int]:
    return manager.capacity()


# ------------------------------------------------------------------ routes
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["系统"],
    summary="健康检查与名额状态",
)
def health(request: Request):
    manager = manager_of(request)
    return {
        "status": "ok",
        "version": __version__,
        "capacity": _capacity_dict(manager),
    }


@app.post(
    "/api/v1/validations",
    response_model=JobSubmissionResponse,
    status_code=202,
    tags=["验证任务"],
    summary="上传随访 CSV 并启动三折分组验证",
    description=(
        "CSV 第一行必须是表头，必须含 `subject_id` 与 `outcome` 两列，"
        "其余列视为数值特征；特征单元格可留空（按训练折中位数填补）。\n\n"
        "`demo_delay_seconds` 为演示用每折额外等待秒数（0-20），"
        "便于观察并行与取消；真实分析会忽略规模很小的该延迟。"
    ),
    responses={
        202: {"model": JobSubmissionResponse},
        400: {"model": ErrorResponse, "description": "CSV 校验失败"},
        429: {"model": ErrorResponse, "description": "全站已有 2 批在训练"},
    },
)
async def create_validation(
    request: Request,
    file: UploadFile = File(..., description="UTF-8 编码的随访 CSV"),
    seed: int = Form(DEFAULT_SEED, ge=0, le=2**31 - 1),
    demo_delay_seconds: float = Form(
        0.0, ge=0.0, le=MAX_DEMO_DELAY_SECONDS
    ),
):
    manager = manager_of(request)

    if file.content_type not in CSV_CONTENT_TYPES:
        return error_response(
            400,
            "INVALID_CONTENT_TYPE",
            "请上传 CSV 文件（text/csv）。",
            {"received": file.content_type},
        )

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(raw) > MAX_UPLOAD_BYTES:
        return error_response(
            413,
            "UPLOAD_TOO_LARGE",
            f"文件超过上限（{MAX_UPLOAD_BYTES // (1024 * 1024)} MiB）。",
        )

    filename = file.filename or "upload.csv"
    try:
        dataset = parse_followup_csv(raw, filename)
    except CsvValidationError as exc:
        return error_response(400, exc.code, exc.message, exc.details)

    try:
        job = manager.submit(
            dataset, seed=seed, demo_delay_seconds=float(demo_delay_seconds)
        )
    except CapacityExceeded as exc:
        return error_response(
            429,
            "CAPACITY_EXCEEDED",
            "全站已有 2 批正在训练，请稍后重试。",
            {"max_concurrent": exc.max_jobs, "active_slots": exc.active_jobs},
            headers={"Retry-After": "5"},
        )

    base = "/api/v1/validations"
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "state": job.state,
            "status_url": f"{base}/{job.job_id}",
            "report_url": None,
            "cancel_url": f"{base}/{job.job_id}/cancel",
            "capacity": _capacity_dict(manager),
            "message": "已受理并开始计算；折外概率将在完成后的报告中按原行序给出。",
        },
    )


@app.get(
    "/api/v1/validations",
    response_model=JobListResponse,
    tags=["验证任务"],
    summary="列出内存中的任务及名额状态",
)
def list_validations(request: Request):
    manager = manager_of(request)
    jobs = manager.list_jobs()
    capacity = _capacity_dict(manager)
    return {
        "jobs": [job_status_payload(job, manager) for job in jobs],
        "capacity": capacity,
    }


@app.get(
    "/api/v1/validations/{job_id}",
    response_model=JobStatus,
    tags=["验证任务"],
    summary="查询任务状态与结果摘要",
    responses={404: {"model": ErrorResponse}},
)
def get_validation(request: Request, job_id: str):
    manager = manager_of(request)
    try:
        job = manager.get(job_id)
    except JobNotFound:
        return error_response(404, "JOB_NOT_FOUND", "找不到该任务。", {"job_id": job_id})
    return job_status_payload(job, manager)


@app.post(
    "/api/v1/validations/{job_id}/cancel",
    response_model=CancelResponse,
    tags=["验证任务"],
    summary="取消任务（确认计算已停止后才返回）",
    description=(
        "仅当本批计算线程实际停止后才返回 200 并释放名额；"
        "若任务已完成或已失败，返回 409 并保留原终态；"
        "取消与完成并发时只保留一种终态，取消后不会发布迟到结果。"
    ),
    responses={
        200: {"model": CancelResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse, "description": "任务已处于其他终态"},
    },
)
def cancel_validation(request: Request, job_id: str):
    manager = manager_of(request)
    try:
        job = manager.cancel(job_id)
    except JobNotFound:
        return error_response(404, "JOB_NOT_FOUND", "找不到该任务。", {"job_id": job_id})
    except CancelNotAllowed as exc:
        return error_response(
            409,
            "CANCEL_NOT_ALLOWED",
            "任务已处于终态，无法取消。",
            {"current_state": exc.state},
        )
    except CancelConfirmationTimeout:
        return error_response(
            409,
            "CANCEL_NOT_CONFIRMED",
            "取消请求已发出，但计算线程尚未停止，名额未释放。",
            {"job_id": job_id},
        )
    return {
        "job_id": job.job_id,
        "state": job.state,
        "message": "计算已停止，名额已释放。",
        "cancel_confirmed_at": job.cancel_confirmed_at,
        "capacity": _capacity_dict(manager),
    }


@app.get(
    "/api/v1/validations/{job_id}/report",
    tags=["验证任务"],
    summary="下载完整 JSON 报告（仅成功的任务）",
    response_class=Response,
    responses={
        200: {
            "content": {"application/json": {}},
            "description": "完整报告：折外概率按原行序、每折 AUC、预处理与模型参数",
        },
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse, "description": "失败、取消或仍在运行的任务不提供下载"},
    },
)
def download_report(request: Request, job_id: str):
    manager = manager_of(request)
    try:
        job = manager.get(job_id)
    except JobNotFound:
        return error_response(404, "JOB_NOT_FOUND", "找不到该任务。", {"job_id": job_id})

    if job.state in ACTIVE_STATES:
        return error_response(
            409,
            "REPORT_NOT_READY",
            "任务仍在进行中，暂无报告可下载。",
            {"current_state": job.state},
        )
    if job.state != COMPLETED or job.report_bytes is None:
        failure = job.failure or {}
        return error_response(
            409,
            "REPORT_UNAVAILABLE",
            "该任务没有可下载的报告。",
            {
                "current_state": job.state,
                "failure_code": failure.get("code"),
                "reason": failure.get("message"),
            },
        )

    return Response(
        content=job.report_bytes,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="validation-report-{job.job_id}.json"'
            )
        },
    )


@app.get(
    "/api/v1/validations/{job_id}/oof.csv",
    tags=["验证任务"],
    summary="下载折外概率 CSV（按原行序，仅成功的任务）",
    responses={
        200: {"content": {"text/csv": {}}},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
def download_oof(request: Request, job_id: str):
    manager = manager_of(request)
    try:
        job = manager.get(job_id)
    except JobNotFound:
        return error_response(404, "JOB_NOT_FOUND", "找不到该任务。", {"job_id": job_id})
    if job.state != COMPLETED or job.result is None:
        return error_response(
            409,
            "REPORT_UNAVAILABLE",
            "该任务没有可下载的折外概率。",
            {"current_state": job.state},
        )

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["row_index", "subject_id", "outcome", "fold", "positive_probability"])
    for row in job.result["oof_predictions"]:
        writer.writerow(
            [
                row["row_index"],
                row["subject_id"],
                row["outcome"],
                row["fold"],
                f"{row['positive_probability']:.10f}",
            ]
        )
    return Response(
        content=buffer.getvalue().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="oof-{job.job_id}.csv"'
            )
        },
    )
