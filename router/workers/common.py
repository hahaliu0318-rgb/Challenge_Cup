from __future__ import annotations

import contextlib
import json
import sys
import time
import traceback
from typing import Any, Callable


PROTOCOL_STDOUT = sys.stdout


def emit(payload: dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTOCOL_STDOUT.flush()


def serve(
    load_runtime: Callable[[], Any],
    handle: Callable[[Any, dict[str, Any]], dict[str, Any]],
    close_runtime: Callable[[Any], None] | None = None,
) -> None:
    runtime: Any = None
    try:
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            runtime = load_runtime()
        emit({"event": "ready", "load_seconds": round(time.perf_counter() - started, 6)})
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request_id = None
            try:
                request = json.loads(line)
                request_id = request.get("id")
                if request.get("op") == "shutdown":
                    emit({"id": request_id, "ok": True, "event": "shutdown"})
                    break
                started = time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    result = handle(runtime, request)
                result = dict(result)
                result.update({"id": request_id, "ok": True})
                result.setdefault("inference_seconds", round(time.perf_counter() - started, 6))
                emit(result)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                emit(
                    {
                        "id": request_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"event": "startup_error", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if runtime is not None and close_runtime is not None:
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    close_runtime(runtime)
            except Exception:
                traceback.print_exc(file=sys.stderr)
