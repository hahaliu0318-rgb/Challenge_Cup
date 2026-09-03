from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .gpu import choose_gpu_candidate, query_gpus


LOGGER = logging.getLogger("router.workers")
StatusCallback = Callable[[str], Awaitable[None] | None]


class WorkerError(RuntimeError):
    pass


class JobCancelled(WorkerError):
    pass


@dataclass
class WorkerSlot:
    name: str
    process: asyncio.subprocess.Process
    gpus: list[int]
    load_seconds: float
    started_at: float
    last_used_at: float
    stderr_task: asyncio.Task | None = None


async def _notify(callback: StatusCallback | None, status: str) -> None:
    if callback is None:
        return
    result = callback(status)
    if inspect.isawaitable(result):
        await result


class WorkerManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.all_model_configs: dict[str, dict[str, Any]] = config["models"]
        self.model_configs = {
            name: model for name, model in self.all_model_configs.items() if model.get("enabled", True)
        }
        self.disabled_models = {
            name: model for name, model in self.all_model_configs.items() if not model.get("enabled", True)
        }
        self.poll_interval = float(config["runtime"].get("gpu_poll_interval_sec", 15))
        self.idle_ttl = float(config["runtime"].get("worker_idle_ttl_sec", 900))
        self.slots: dict[str, WorkerSlot] = {}
        self.leases: dict[int, str] = {}
        self.worker_locks = {name: asyncio.Lock() for name in self.model_configs}
        self.lifecycle_lock = asyncio.Lock()
        self._reaper_task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        self._closed = False
        self._reaper_task = asyncio.create_task(self._idle_reaper(), name="router-worker-reaper")

    async def close(self) -> None:
        self._closed = True
        if self._reaper_task:
            self._reaper_task.cancel()
            await asyncio.gather(self._reaper_task, return_exceptions=True)
        for name in list(self.slots):
            await self.stop_worker(name)

    def _slot_alive(self, name: str) -> bool:
        slot = self.slots.get(name)
        return bool(slot and slot.process.returncode is None)

    async def _drain_stderr(self, name: str, stream: asyncio.StreamReader) -> None:
        while True:
            line = await stream.readline()
            if not line:
                return
            LOGGER.info("worker=%s %s", name, line.decode(errors="replace").rstrip())

    async def _evict_idle_heavy_peer(self, requested: str) -> bool:
        if requested not in {"geollava", "zoomsearch"}:
            return False
        peer = "zoomsearch" if requested == "geollava" else "geollava"
        if not self._slot_alive(peer) or self.worker_locks[peer].locked():
            return False
        async with self.worker_locks[peer]:
            if self._slot_alive(peer):
                LOGGER.info("evicting idle heavy worker %s for %s", peer, requested)
                await self._stop_slot(peer)
                return True
        return False

    async def _reserve_gpus(
        self,
        name: str,
        status_callback: StatusCallback | None,
        cancel_event: asyncio.Event | None,
    ) -> list[int]:
        model_config = self.model_configs[name]
        last_status = None
        while not self._closed:
            if cancel_event and cancel_event.is_set():
                raise JobCancelled("job cancelled while waiting for GPU")
            try:
                inventory = await asyncio.to_thread(query_gpus)
            except Exception as exc:
                LOGGER.warning("GPU inventory failed: %s", exc)
                inventory = []
            async with self.lifecycle_lock:
                if self._slot_alive(name):
                    return list(self.slots[name].gpus)
                candidate = choose_gpu_candidate(
                    inventory,
                    model_config["gpu_candidates"],
                    model_config["min_free_mib"],
                    self.leases,
                ) if inventory else None
                if candidate:
                    for gpu in candidate:
                        self.leases[gpu] = name
                    return candidate
            if await self._evict_idle_heavy_peer(name):
                continue
            if last_status != "queued_waiting_gpu":
                await _notify(status_callback, "queued_waiting_gpu")
                last_status = "queued_waiting_gpu"
            try:
                await asyncio.wait_for(
                    cancel_event.wait() if cancel_event else asyncio.sleep(self.poll_interval),
                    timeout=self.poll_interval,
                )
            except asyncio.TimeoutError:
                pass
        raise WorkerError("worker manager is shutting down")

    async def _start_slot(
        self,
        name: str,
        status_callback: StatusCallback | None,
        cancel_event: asyncio.Event | None,
    ) -> tuple[WorkerSlot, bool]:
        if self._slot_alive(name):
            return self.slots[name], False
        if name in self.slots:
            await self._stop_slot(name)
        gpus = await self._reserve_gpus(name, status_callback, cancel_event)
        model_config = self.model_configs[name]
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in model_config.get("env", {}).items()})
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpus)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PYTHONUNBUFFERED"] = "1"
        command = [str(model_config["python"]), str(model_config["worker_script"])]
        process: asyncio.subprocess.Process | None = None
        stderr_task: asyncio.Task | None = None
        try:
            if cancel_event and cancel_event.is_set():
                raise JobCancelled("job cancelled before model loading")
            await _notify(status_callback, "loading_model")
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(model_config["working_dir"]),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert process.stdout and process.stderr
            stderr_task = asyncio.create_task(self._drain_stderr(name, process.stderr))
            timeout = float(model_config.get("startup_timeout_sec", 900))
            ready_task = asyncio.create_task(process.stdout.readline())
            cancel_task = asyncio.create_task(cancel_event.wait()) if cancel_event else None
            waiters = {ready_task, *([cancel_task] if cancel_task else [])}
            done, pending = await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if not done:
                raise asyncio.TimeoutError(f"{name} worker startup timed out")
            if cancel_task and cancel_task in done and cancel_task.result():
                if not ready_task.done():
                    ready_task.cancel()
                    await asyncio.gather(ready_task, return_exceptions=True)
                raise JobCancelled("job cancelled during model loading")
            line = ready_task.result()
            if not line:
                raise WorkerError(f"{name} worker exited during startup")
            payload = json.loads(line.decode())
            if payload.get("event") != "ready":
                raise WorkerError(payload.get("error") or f"unexpected {name} startup response: {payload}")
            now = time.time()
            slot = WorkerSlot(
                name=name,
                process=process,
                gpus=gpus,
                load_seconds=float(payload.get("load_seconds", 0.0)),
                started_at=now,
                last_used_at=now,
                stderr_task=stderr_task,
            )
            self.slots[name] = slot
            LOGGER.info("worker %s ready pid=%s gpus=%s", name, process.pid, gpus)
            return slot, True
        except BaseException:
            if process and process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)
                except ProcessLookupError:
                    pass
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            if stderr_task:
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
            async with self.lifecycle_lock:
                for gpu in gpus:
                    if self.leases.get(gpu) == name:
                        self.leases.pop(gpu, None)
            raise

    async def _stop_slot(self, name: str) -> None:
        slot = self.slots.pop(name, None)
        if not slot:
            return
        process = slot.process
        if process.returncode is None:
            try:
                if process.stdin:
                    request = {"id": str(uuid.uuid4()), "op": "shutdown"}
                    process.stdin.write((json.dumps(request) + "\n").encode())
                    await process.stdin.drain()
                await asyncio.wait_for(process.wait(), timeout=30)
            except Exception:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
        if slot.stderr_task:
            slot.stderr_task.cancel()
            await asyncio.gather(slot.stderr_task, return_exceptions=True)
        async with self.lifecycle_lock:
            for gpu in slot.gpus:
                if self.leases.get(gpu) == name:
                    self.leases.pop(gpu, None)
        LOGGER.info("worker %s stopped", name)

    async def stop_worker(self, name: str) -> None:
        if name not in self.worker_locks:
            return
        async with self.worker_locks[name]:
            await self._stop_slot(name)

    async def invoke(
        self,
        name: str,
        request: dict[str, Any],
        status_callback: StatusCallback | None = None,
        cancel_event: asyncio.Event | None = None,
        retry: bool = True,
    ) -> dict[str, Any]:
        if name not in self.model_configs:
            raise WorkerError(f"unknown worker: {name}")
        async with self.worker_locks[name]:
            attempts = 2 if retry else 1
            last_error: Exception | None = None
            for attempt in range(attempts):
                if cancel_event and cancel_event.is_set():
                    raise JobCancelled("job cancelled before inference")
                try:
                    slot, freshly_started = await self._start_slot(name, status_callback, cancel_event)
                    if cancel_event and cancel_event.is_set():
                        raise JobCancelled("job cancelled after model loading")
                    await _notify(status_callback, "running")
                    wire_request = dict(request)
                    wire_request.setdefault("id", str(uuid.uuid4()))
                    assert slot.process.stdin and slot.process.stdout
                    slot.process.stdin.write((json.dumps(wire_request, ensure_ascii=False) + "\n").encode())
                    await slot.process.stdin.drain()
                    timeout = float(self.model_configs[name].get("inference_timeout_sec", 900))
                    line = await asyncio.wait_for(slot.process.stdout.readline(), timeout=timeout)
                    if not line:
                        raise WorkerError(f"{name} worker exited without a response")
                    payload = json.loads(line.decode())
                    if payload.get("id") != wire_request["id"]:
                        raise WorkerError(f"{name} worker response id mismatch")
                    if not payload.get("ok"):
                        raise WorkerError(payload.get("error") or f"{name} worker failed")
                    slot.last_used_at = time.time()
                    payload["load_seconds"] = slot.load_seconds if freshly_started else 0.0
                    payload["worker_gpus"] = list(slot.gpus)
                    return payload
                except JobCancelled:
                    raise
                except Exception as exc:
                    last_error = exc
                    LOGGER.exception("worker %s invocation attempt %s failed", name, attempt + 1)
                    await self._stop_slot(name)
                    if attempt + 1 >= attempts:
                        break
            raise WorkerError(f"{name} failed after {attempts} attempt(s): {last_error}")

    async def release_if_idle(self, name: str) -> None:
        if name not in self.worker_locks or self.worker_locks[name].locked():
            return
        async with self.worker_locks[name]:
            if self._slot_alive(name):
                await self._stop_slot(name)

    async def _idle_reaper(self) -> None:
        while not self._closed:
            await asyncio.sleep(min(30.0, max(5.0, self.idle_ttl / 3)))
            await self._reap_idle_once(time.time())

    async def _reap_idle_once(self, now: float | None = None) -> None:
        current = time.time() if now is None else now
        for name, slot in list(self.slots.items()):
            if current - slot.last_used_at < self.idle_ttl or self.worker_locks[name].locked():
                continue
            await self.stop_worker(name)

    async def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in self.model_configs:
            slot = self.slots.get(name)
            result[name] = {
                "state": "ready" if self._slot_alive(name) else "stopped",
                "pid": slot.process.pid if slot and self._slot_alive(name) else None,
                "gpus": list(slot.gpus) if slot and self._slot_alive(name) else [],
                "busy": self.worker_locks[name].locked(),
                "last_used_at": slot.last_used_at if slot else None,
                "idle_ttl_sec": self.idle_ttl,
            }
        for name, model in self.disabled_models.items():
            result[name] = {
                "state": "disabled",
                "phase": model.get("phase"),
                "reason": model.get("disabled_reason", "disabled by configuration"),
            }
        result["leases"] = {str(gpu): worker for gpu, worker in sorted(self.leases.items())}
        try:
            result["gpu_inventory"] = [gpu.to_dict() for gpu in await asyncio.to_thread(query_gpus)]
        except Exception as exc:
            result["gpu_inventory_error"] = str(exc)
        return result
