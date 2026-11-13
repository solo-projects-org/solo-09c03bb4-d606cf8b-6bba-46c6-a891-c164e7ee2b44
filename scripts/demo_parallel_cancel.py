#!/usr/bin/env python3
"""Live demo against a running API: parallel runs, reject, cancel, finish.

Start the service first:

    docker compose up --build
    # or: uvicorn app.main:app --port 8000

Then run:

    python3 scripts/demo_parallel_cancel.py

The script uses only the standard library.  It submits two slow batches
(8-second synthetic delay per fold, ~24s each), shows that a third batch is
rejected with 429, cancels the first batch and times how long the confirmation
takes, and verifies the second batch keeps training and finishes with AUCs.
"""

from __future__ import annotations

import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
DATA = Path(__file__).resolve().parent.parent / "data"


def multipart(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----demo{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"\r\nContent-Type: {content_type}\r\n\r\n'
        ).encode()
    )
    chunks.append(file_path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def request(method: str, path: str, *, body: bytes | None = None, content_type: str | None = None):
    req = urllib.request.Request(f"{BASE_URL}{path}", data=body, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw.decode("utf-8", "replace")}


def submit(path: Path, delay: float) -> tuple[int, dict]:
    body, content_type = multipart(
        {"seed": "42", "demo_delay_seconds": str(delay)}, "file", path
    )
    return request("POST", "/api/v1/validations", body=body, content_type=content_type)


def status(job_id: str) -> dict:
    return request("GET", f"/api/v1/validations/{job_id}")[1]


def wait(job_id: str, states: set[str], timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = status(job_id)
        if last.get("state") in states:
            return last
        time.sleep(0.4)
    raise RuntimeError(f"timeout waiting for {states}; last={last}")


def describe(payload: dict) -> str:
    progress = payload.get("progress", {})
    fold_aucs = (payload.get("result_summary") or {}).get("fold_aucs", [])
    aucs = ", ".join(
        f"fold {fold['fold']}: "
        + ("null" if fold["auc"] is None else f"{fold['auc']:.3f}")
        for fold in fold_aucs
    )
    return (
        f"state={payload['state']} stage={progress.get('stage')} "
        f"fold={progress.get('fold')} percent={progress.get('percent')} "
        f"{('| ' + aucs) if aucs else ''}"
    )


def main() -> int:
    print(f"API: {BASE_URL}")
    code, health = request("GET", "/api/v1/health")
    if code != 200:
        print(f"health check failed ({code}): {health}")
        return 1
    print(f"health: {health}")
    print(f"capacity before demo: {health['capacity']}\n")

    slow = DATA / "followup_slow_demo.csv"
    if not slow.exists():
        print(f"missing {slow}; run python3 scripts/generate_demo_data.py")
        return 1

    code_a, job_a = submit(slow, delay=8)
    code_b, job_b = submit(slow, delay=8)
    if code_a != 202 or code_b != 202:
        print(f"could not start two batches: {code_a} {job_a} / {code_b} {job_b}")
        return 1
    id_a, id_b = job_a["job_id"], job_b["job_id"]
    print(f"batch A: {id_a}")
    print(f"batch B: {id_b}")
    print(f"capacity after two submissions: {job_b['capacity']}")

    code_c, job_c = submit(slow, delay=8)
    print(f"\nthird batch -> HTTP {code_c} {job_c.get('error', job_c)}")
    if code_c != 429:
        print("expected 429 for the third concurrent batch")

    time.sleep(2)
    print(f"\nA while running: {describe(status(id_a))}")
    print(f"B while running: {describe(status(id_b))}")

    started = time.monotonic()
    code_cancel, cancelled = request("POST", f"/api/v1/validations/{id_a}/cancel")
    elapsed = time.monotonic() - started
    print(
        f"\ncancel A -> HTTP {code_cancel} after {elapsed:.2f}s "
        f"(state={cancelled.get('state')}, capacity={cancelled.get('capacity')})"
    )
    print(f"A after cancel: {describe(status(id_a))}")
    print(
        "A report (must be unavailable): "
        f"HTTP {request('GET', f'/api/v1/validations/{id_a}/report')[0]}"
    )

    print(f"\nB after A was cancelled: {describe(status(id_b))}")
    final_b = wait(id_b, {"completed", "failed", "cancelled"}, timeout=120)
    print(f"B final: {describe(final_b)}")
    if final_b["state"] == "completed":
        print(f"B report: HTTP {request('GET', f'/api/v1/validations/{id_b}/report')[0]}")
        # The slow demo batch resumes a slot for another submission.
        code_d, job_d = submit(DATA / "followup_demo.csv", delay=0)
        print(f"new submission after slot release -> HTTP {code_d}")
        if code_d == 202:
            done_d = wait(job_d["job_id"], {"completed", "failed"})
            print(f"normal batch final: {describe(done_d)}")

    print("\n--- validation fold with a single class (AUC null) ---")
    code_e, job_e = submit(DATA / "followup_validation_single.csv", delay=0)
    if code_e == 202:
        done_e = wait(job_e["job_id"], {"completed", "failed"})
        print(f"single-class validation fold: {describe(done_e)}")
        for fold in (done_e.get("result_summary") or {}).get("fold_aucs", []):
            if fold["auc"] is None:
                print(f"  fold {fold['fold']} auc_reason: {fold['auc_reason']}")

    print("\n--- training fold with a single class (whole batch fails) ---")
    code_f, job_f = submit(DATA / "followup_train_single.csv", delay=0)
    if code_f == 202:
        done_f = wait(job_f["job_id"], {"completed", "failed"})
        print(f"failure: {done_f.get('failure')}")
        report_code = request(
            "GET", f"/api/v1/validations/{job_f['job_id']}/report"
        )[0]
        print(f"report (must be 409): HTTP {report_code}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
