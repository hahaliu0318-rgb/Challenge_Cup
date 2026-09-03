from __future__ import annotations

import os
import sys
from typing import Any

try:
    from .common import serve
except ImportError:
    from common import serve


CLASSIFIER_PROMPT = """Classify the remote-sensing request below.
Return JSON only with keys task, subtask, confidence.
Allowed task values: caption, vqa, counting, grounding, detection,
scene_classification, change_caption, color, position,
spatial_relationship, complex_reasoning, land_use_classification,
object_classification, motion_state.
Use subtask null when unnecessary.
Request: {text}
"""


def load_runtime() -> dict[str, Any]:
    utils_dir = os.environ["QWEN_UTILS_DIR"]
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    from qwen25vl_utils import choose_input_device, load_model_and_processor

    max_pixels = int(os.environ.get("QWEN_MAX_PIXELS", "1204224"))
    model, processor = load_model_and_processor(
        model_id=os.environ["QWEN_MODEL_PATH"],
        adapter_path=os.environ["QWEN_ADAPTER_PATH"],
        max_pixels=max_pixels,
        attn_implementation="default",
        load_in_4bit=False,
        local_files_only=True,
    )
    return {
        "model": model,
        "processor": processor,
        "input_device": choose_input_device("auto"),
    }


def handle(runtime: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    from qwen25vl_utils import (
        build_text_messages,
        generate_answer,
        generate_from_messages,
        image_to_qwen_uri,
    )

    operation = request.get("op", "infer")
    if operation == "classify":
        prompt = CLASSIFIER_PROMPT.format(text=str(request["text"]))
        answer = generate_from_messages(
            model=runtime["model"],
            processor=runtime["processor"],
            messages=build_text_messages(prompt),
            input_device=runtime["input_device"],
            max_new_tokens=128,
            temperature=0.0,
        )
        return {"answer": answer}
    if operation != "infer":
        raise ValueError(f"unsupported operation: {operation}")
    image_uris = [image_to_qwen_uri(path) for path in request["images"]]
    answer = generate_answer(
        model=runtime["model"],
        processor=runtime["processor"],
        image_uris=image_uris,
        question=str(request["text"]),
        input_device=runtime["input_device"],
        max_new_tokens=int(request.get("max_new_tokens", 256)),
        temperature=0.0,
    )
    return {"answer": answer}


def close_runtime(runtime: dict[str, Any]) -> None:
    import gc
    import torch

    runtime.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    serve(load_runtime, handle, close_runtime)
