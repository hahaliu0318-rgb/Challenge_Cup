from __future__ import annotations

import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from .common import serve
except ImportError:
    from common import serve


def _runtime_dir() -> Path:
    return Path(os.environ.get("ZOOM_RUNTIME_DIR", "/tmp/router_zoom_runtime")).resolve()


def load_runtime() -> dict[str, Any]:
    code_root = os.environ["ZOOM_CODE_ROOT"]
    llava_root = os.environ["ZOOM_LLAVA_ROOT"]
    for path in (code_root, llava_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch
    from vlm.config import supported_VLM

    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"ZoomSearch requires exactly two visible GPUs, found {torch.cuda.device_count()}")
    with torch.cuda.device(1):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("ZoomSearch search GPU must support bfloat16")

    runtime_dir = _runtime_dir()
    shutil.rmtree(runtime_dir, ignore_errors=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    model = supported_VLM["llava_onevision_qwen2_7b_ov"](
        model_path=os.environ["ZOOM_MODEL_PATH"],
        max_new_tokens=int(os.environ.get("ZOOM_MAX_NEW_TOKENS", "1024")),
        max_step=int(os.environ.get("ZOOM_MAX_STEP", "10")),
        max_depth=int(os.environ.get("ZOOM_MAX_DEPTH", "50")),
        bias_value=float(os.environ.get("ZOOM_BIAS_VALUE", "0.3")),
        search_model_path=os.environ["ZOOM_SEARCH_MODEL_PATH"],
        search_device="cuda:1",
        vlm_device="cuda:0",
        search_image_policy="paper_llava_ov",
        save_intermediate=False,
        intermediate_image_path=str(runtime_dir),
    )
    return {"model": model, "runtime_dir": runtime_dir}


def _cleanup_sample(runtime: dict[str, Any], message: list[dict[str, Any]]) -> None:
    import torch

    for entry in message:
        value = entry.get("value")
        if hasattr(value, "close"):
            try:
                value.close()
            except Exception:
                pass
    try:
        runtime["model"].release_sample_state()
    finally:
        shutil.rmtree(runtime["runtime_dir"], ignore_errors=True)
        runtime["runtime_dir"].mkdir(parents=True, exist_ok=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def handle(runtime: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    if request.get("op", "infer") != "infer":
        raise ValueError("ZoomSearch worker only supports infer")
    if len(request["images"]) != 1:
        raise ValueError("ZoomSearch phase 1 supports exactly one image")
    import torch

    message: list[dict[str, Any]] = [
        {"type": "image", "value": str(request["images"][0])},
        {"type": "text", "value": str(request["text"])},
    ]
    try:
        with torch.inference_mode():
            answer = runtime["model"].generate(message, zoom=True)
        return {
            "answer": str(answer),
            "backend": {
                "actual_search_image_size": int(getattr(runtime["model"], "last_search_image_size", 0)),
            },
        }
    finally:
        _cleanup_sample(runtime, message)


def close_runtime(runtime: dict[str, Any]) -> None:
    try:
        runtime["model"].release_sample_state()
    except Exception:
        pass
    shutil.rmtree(runtime.get("runtime_dir", _runtime_dir()), ignore_errors=True)
    runtime.clear()
    gc.collect()


if __name__ == "__main__":
    serve(load_runtime, handle, close_runtime)
