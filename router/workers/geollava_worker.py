from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

try:
    from .common import serve
except ImportError:
    from common import serve


def load_runtime() -> dict[str, Any]:
    scripts_dir = os.environ["GEOLLAVA_SCRIPTS_DIR"]
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from xlrs_lite_eval import load_runtime as load_geollava_runtime

    return load_geollava_runtime(
        model_path=Path(os.environ["GEOLLAVA_MODEL_PATH"]),
        attention=os.environ.get("GEOLLAVA_ATTENTION", "flash_attention_2"),
        max_memory_per_gpu_gib=int(os.environ.get("GEOLLAVA_MAX_MEMORY_GIB", "21")),
    )


def build_record(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": request.get("id"),
        "question": str(request["text"]),
        "multi-choice options": "",
        "answer": "",
        "image_paths": [str(path) for path in request["images"]],
    }


def handle(runtime: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    if request.get("op", "infer") != "infer":
        raise ValueError("GeoLLaVA worker only supports infer")
    from xlrs_lite_eval import infer_record

    record = build_record(request)
    result = infer_record(
        record=record,
        dataset_root=Path("/"),
        runtime=runtime,
        conv_template=os.environ.get("GEOLLAVA_CONV_TEMPLATE", "vicuna_v1"),
        max_new_tokens=int(request.get("max_new_tokens", 128)),
    )
    return {
        "answer": result["response"],
        "backend": {
            "prompt_token_count": result.get("prompt_token_count"),
            "generated_token_count": result.get("generated_token_count"),
            "timing_seconds": result.get("timing_seconds"),
        },
    }


def close_runtime(runtime: dict[str, Any]) -> None:
    import gc
    import torch

    runtime.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    serve(load_runtime, handle, close_runtime)
