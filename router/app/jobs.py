from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .image_probe import validate_and_probe_images
from .output_parser import augment_prompt, parse_output, repair_prompt
from .routing import choose_route
from .schemas import InferRequest
from .task_classifier import TaskDecision, classify_by_rules, parse_model_classification
from .worker_manager import JobCancelled, WorkerManager


LOGGER = logging.getLogger("router.jobs")
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
PENDING_STATUSES = {"queued", "classifying", "queued_waiting_gpu", "loading_model", "running"}


class JobStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    prepared_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    route_worker TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL
                )
                """
            )
            self.connection.commit()

    def create(self, request: dict[str, Any], prepared: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        with self.lock:
            self.connection.execute(
                "INSERT INTO jobs (id,status,request_json,prepared_json,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                (job_id, "queued", json.dumps(request, ensure_ascii=False), json.dumps(prepared, ensure_ascii=False), now, now),
            )
            self.connection.commit()
        return job_id

    def update(self, job_id: str, **values: Any) -> None:
        if not values:
            return
        values["updated_at"] = time.time()
        columns = ", ".join(f"{key}=?" for key in values)
        with self.lock:
            self.connection.execute(
                f"UPDATE jobs SET {columns} WHERE id=?",
                (*values.values(), job_id),
            )
            self.connection.commit()

    def get_row(self, job_id: str) -> sqlite3.Row | None:
        with self.lock:
            return self.connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self.get_row(job_id)
        if row is None:
            return None
        result = {
            "job_id": row["id"],
            "status": row["status"],
            "route_worker": row["route_worker"],
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "cancel_requested": bool(row["cancel_requested"]),
        }
        if row["result_json"]:
            result["result"] = json.loads(row["result_json"])
        return result

    def request_and_prepared(self, job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        row = self.get_row(job_id)
        if row is None:
            raise KeyError(job_id)
        return json.loads(row["request_json"]), json.loads(row["prepared_json"])

    def pending_ids(self) -> list[str]:
        placeholders = ",".join("?" for _ in PENDING_STATUSES)
        with self.lock:
            rows = self.connection.execute(
                f"SELECT id FROM jobs WHERE status IN ({placeholders}) ORDER BY created_at",
                tuple(PENDING_STATUSES),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def cleanup(self, retention_days: int) -> int:
        cutoff = time.time() - retention_days * 86400
        with self.lock:
            cursor = self.connection.execute(
                "DELETE FROM jobs WHERE completed_at IS NOT NULL AND completed_at < ?",
                (cutoff,),
            )
            self.connection.commit()
            return int(cursor.rowcount)

    def close(self) -> None:
        with self.lock:
            self.connection.close()


class JobService:
    def __init__(self, config: dict[str, Any], manager: WorkerManager):
        self.config = config
        self.manager = manager
        runtime = config["runtime"]
        self.store = JobStore(runtime["jobs_db"])
        self.events: dict[str, asyncio.Event] = {}
        self.cancel_events: dict[str, asyncio.Event] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.submission_tasks: set[asyncio.Task] = set()
        self.retention_days = int(runtime.get("job_retention_days", 7))

    async def start(self) -> None:
        self.store.cleanup(self.retention_days)
        for job_id in self.store.pending_ids():
            self.store.update(job_id, status="queued", error=None)
            self._schedule(job_id)

    async def close(self) -> None:
        for task in list(self.submission_tasks):
            task.cancel()
        if self.submission_tasks:
            await asyncio.gather(*self.submission_tasks, return_exceptions=True)
        for event in self.cancel_events.values():
            event.set()
        if self.tasks:
            for task in list(self.tasks.values()):
                task.cancel()
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.store.close()

    def prepare(self, request: InferRequest) -> dict[str, Any]:
        received_at = time.time()
        runtime = self.config["runtime"]
        paths, metadata = validate_and_probe_images(
            request.image_paths(),
            runtime["allowed_image_roots"],
            int(runtime.get("max_image_file_size_bytes", 1_500_000_000)),
            int(runtime.get("max_image_pixels", 500_000_000)),
        )
        task = classify_by_rules(request.text, len(paths), request.task_type)
        validated_at = time.time()
        return {
            "images": paths,
            "image_meta": metadata,
            "task": task.to_dict(),
            "needs_model_classifier": request.task_type is None and task.confidence < 0.8,
            "created_at": received_at,
            "enqueued_at": validated_at,
            "validation_sec": validated_at - received_at,
        }

    def preview(self, request: InferRequest) -> dict[str, Any]:
        prepared = self.prepare(request)
        task = TaskDecision(**prepared["task"])
        route = choose_route(task, prepared["image_meta"], int(self.config["runtime"]["low_resolution_max_edge"]))
        warnings = ["qwen_classifier_deferred_in_preview"] if prepared["needs_model_classifier"] else []
        return {
            "task": task.to_dict(),
            "route": route.to_dict(),
            "image_meta": prepared["image_meta"],
            "warnings": warnings,
        }

    async def submit(self, request: InferRequest) -> str:
        prepared = await asyncio.to_thread(self.prepare, request)
        request_payload = request.model_dump()
        job_id = self.store.create(request_payload, prepared)
        self._schedule(job_id)
        LOGGER.info(
            "job submitted",
            extra={"event": "job_submitted", "request_id": job_id, "task": prepared["task"]["type"]},
        )
        return job_id

    async def submit_durable(self, request: InferRequest) -> str:
        task = asyncio.create_task(self.submit(request), name="router-durable-submission")
        self.submission_tasks.add(task)
        task.add_done_callback(self._finish_submission)
        return await asyncio.shield(task)

    def _finish_submission(self, task: asyncio.Task) -> None:
        self.submission_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    def _schedule(self, job_id: str) -> None:
        self.events[job_id] = asyncio.Event()
        self.cancel_events[job_id] = asyncio.Event()
        task = asyncio.create_task(self._run_job(job_id), name=f"router-job-{job_id}")
        self.tasks[job_id] = task
        task.add_done_callback(lambda _: self._forget_runtime_state(job_id))

    def _forget_runtime_state(self, job_id: str) -> None:
        self.tasks.pop(job_id, None)
        self.events.pop(job_id, None)
        self.cancel_events.pop(job_id, None)

    async def _set_status(self, job_id: str, status: str) -> None:
        values: dict[str, Any] = {"status": status}
        if status == "running":
            row = self.store.get_row(job_id)
            if row is not None and row["started_at"] is None:
                values["started_at"] = time.time()
        self.store.update(job_id, **values)

    async def _run_job(self, job_id: str) -> None:
        cancel_event = self.cancel_events[job_id]
        try:
            request_payload, prepared = self.store.request_and_prepared(job_id)
            task = TaskDecision(**prepared["task"])
            if prepared.get("needs_model_classifier"):
                await self._set_status(job_id, "classifying")
                classification = await self.manager.invoke(
                    "qwen",
                    {"op": "classify", "text": request_payload["text"]},
                    status_callback=lambda status: self._set_status(job_id, status),
                    cancel_event=cancel_event,
                )
                parsed = parse_model_classification(str(classification.get("answer", "")))
                task = parsed or TaskDecision("vqa", None, "classifier_parse_fallback", 0.40)

            route = choose_route(
                task,
                prepared["image_meta"],
                int(self.config["runtime"]["low_resolution_max_edge"]),
            )
            self.store.update(job_id, route_worker=route.worker)
            if prepared.get("needs_model_classifier") and route.worker != "qwen":
                await self.manager.release_if_idle("qwen")
            prompt = augment_prompt(str(request_payload["text"]), task)
            worker_request = {
                "op": "infer",
                "images": prepared["images"],
                "text": prompt,
                "task_type": task.type,
                "max_new_tokens": 256 if task.type in {"caption", "change_caption"} else 128,
            }
            worker_result = await self.manager.invoke(
                route.worker,
                worker_request,
                status_callback=lambda status: self._set_status(job_id, status),
                cancel_event=cancel_event,
            )
            raw_answer = str(worker_result.get("answer", "")).strip()
            answer, parsed_ok, warnings = parse_output(raw_answer, task, request_payload["text"], prepared["image_meta"])
            if not parsed_ok and not cancel_event.is_set():
                worker_request["text"] = repair_prompt(str(request_payload["text"]), raw_answer, task)
                repaired = await self.manager.invoke(
                    route.worker,
                    worker_request,
                    status_callback=lambda status: self._set_status(job_id, status),
                    cancel_event=cancel_event,
                    retry=False,
                )
                repaired_raw = str(repaired.get("answer", "")).strip()
                repaired_answer, repaired_ok, repaired_warnings = parse_output(
                    repaired_raw, task, request_payload["text"], prepared["image_meta"]
                )
                raw_answer = repaired_raw
                answer = repaired_answer
                warnings = repaired_warnings
                if not repaired_ok:
                    warnings.append("format_repair_failed")
                worker_result = repaired

            now = time.time()
            started_at = self.store.get_row(job_id)["started_at"] or now
            result = {
                "request_id": job_id,
                "status": "succeeded",
                "answer": answer,
                "raw_answer": raw_answer,
                "task": task.to_dict(),
                "route": route.to_dict(),
                "image_meta": prepared["image_meta"],
                "timing": {
                    "queue_sec": round(
                        max(0.0, started_at - float(prepared.get("enqueued_at", prepared["created_at"]))), 6
                    ),
                    "validation_sec": round(float(prepared.get("validation_sec", 0.0)), 6),
                    "load_sec": round(float(worker_result.get("load_seconds", 0.0)), 6),
                    "inference_sec": round(float(worker_result.get("inference_seconds", 0.0)), 6),
                    "total_sec": round(now - float(prepared["created_at"]), 6),
                },
                "worker_gpus": worker_result.get("worker_gpus", []),
                "backend": worker_result.get("backend", {}),
                "warnings": warnings,
            }
            self.store.update(
                job_id,
                status="succeeded",
                result_json=json.dumps(result, ensure_ascii=False),
                error=None,
                completed_at=now,
            )
            LOGGER.info(
                "job succeeded",
                extra={
                    "event": "job_succeeded",
                    "request_id": job_id,
                    "worker": route.worker,
                    "task": task.type,
                    "duration_sec": result["timing"]["total_sec"],
                },
            )
        except JobCancelled as exc:
            self.store.update(job_id, status="cancelled", error=str(exc), completed_at=time.time())
            LOGGER.info(
                "job cancelled",
                extra={"event": "job_cancelled", "request_id": job_id, "error": str(exc)},
            )
        except asyncio.CancelledError:
            self.store.update(job_id, status="queued", error="gateway shutdown; job will resume")
            raise
        except Exception as exc:
            self.store.update(
                job_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                completed_at=time.time(),
            )
            LOGGER.exception(
                "job failed",
                extra={"event": "job_failed", "request_id": job_id, "error": f"{type(exc).__name__}: {exc}"},
            )
        finally:
            self.events[job_id].set()

    async def wait(self, job_id: str) -> dict[str, Any]:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] not in TERMINAL_STATUSES:
            event = self.events.get(job_id)
            if event is not None:
                await event.wait()
        job = self.store.get(job_id)
        assert job is not None
        return job

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self.store.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] in TERMINAL_STATUSES or job["status"] == "running":
            return False
        self.store.update(job_id, cancel_requested=1)
        event = self.cancel_events.get(job_id)
        if event:
            event.set()
        return True
