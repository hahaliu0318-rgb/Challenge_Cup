from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from . import __version__
from .config import load_config
from .image_probe import ImageValidationError
from .jobs import JobService
from .schemas import InferRequest, JobAccepted
from .worker_manager import WorkerManager


class JsonFormatter(logging.Formatter):
    EXTRA_FIELDS = ("event", "request_id", "worker", "task", "duration_sec", "error")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self.EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging(log_dir: str | Path) -> None:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(path / "router.log", encoding="utf-8"))
    formatter = JsonFormatter()
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    _configure_logging(config["runtime"]["log_dir"])
    Path(config["runtime"]["state_dir"]).mkdir(parents=True, exist_ok=True)
    manager = WorkerManager(config)
    jobs = JobService(config, manager)
    app.state.config = config
    app.state.manager = manager
    app.state.jobs = jobs
    await manager.start()
    await jobs.start()
    try:
        yield
    finally:
        await jobs.close()
        await manager.close()


app = FastAPI(
    title="Remote Sensing Multi-Model Router",
    version=__version__,
    description="Routes server-side remote-sensing images to Qwen LoRA, GeoLLaVA-8K, or ZoomSearch.",
    lifespan=lifespan,
)


def _jobs(request: Request) -> JobService:
    return request.app.state.jobs


@app.exception_handler(ImageValidationError)
async def image_validation_handler(_: Request, exc: ImageValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "version": __version__}


@app.get("/readyz")
async def readyz(request: Request):
    config = request.app.state.config
    checks: dict[str, Any] = {}
    ready = True
    for name, model in config["models"].items():
        if not model.get("enabled", True):
            checks[name] = {
                "enabled": False,
                "phase": model.get("phase"),
                "reason": model.get("disabled_reason", "disabled by configuration"),
            }
            continue
        paths = [model["python"], model["worker_script"], model["working_dir"], *model.get("required_paths", [])]
        for key, value in model.get("env", {}).items():
            if "RUNTIME" in key:
                continue
            if key.endswith("_PATH") or key.endswith("_DIR") or key.endswith("_ROOT"):
                paths.append(value)
        model_checks = []
        for raw in paths:
            path = Path(str(raw))
            exists = path.exists()
            readable = os.access(path, os.R_OK) if exists else False
            model_checks.append({"path": str(path), "exists": exists, "readable": readable})
            ready = ready and exists and readable
        checks[name] = model_checks
    payload = {"status": "ready" if ready else "not_ready", "checks": checks}
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@app.post("/v1/routes/preview")
async def route_preview(payload: InferRequest, request: Request) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_jobs(request).preview, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/jobs", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def submit_job(payload: InferRequest, request: Request) -> JobAccepted:
    try:
        job_id = await _jobs(request).submit_durable(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JobAccepted(job_id=job_id, status="queued", status_url=f"/v1/jobs/{job_id}")


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    job = _jobs(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request):
    try:
        accepted = _jobs(request).cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    if not accepted:
        raise HTTPException(status_code=409, detail="job is already running or terminal")
    return {"job_id": job_id, "status": "cancellation_requested"}


@app.post("/v1/infer")
async def infer(
    payload: InferRequest,
    request: Request,
    trace: bool = Query(default=True),
):
    try:
        job_id = await _jobs(request).submit_durable(payload)
        job = await _jobs(request).wait(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if job["status"] == "succeeded":
        result = job["result"]
        return result if trace else {"answer": result["answer"]}
    if job["status"] == "cancelled":
        raise HTTPException(status_code=409, detail=job.get("error") or "job cancelled")
    raise HTTPException(status_code=500, detail=job.get("error") or "inference failed")


@app.get("/v1/workers")
async def workers(request: Request) -> dict[str, Any]:
    return await request.app.state.manager.status()
