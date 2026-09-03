"""Shared Qwen2.5-VL inference helpers.

These helpers keep the official Qwen inference flow in one place:
apply_chat_template -> process_vision_info -> processor -> model.generate -> decode.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for remote sensing image understanding. "
    "Answer the user's question concisely and directly."
)


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "data", "file"}


def image_to_qwen_uri(image: str | Path) -> str:
    image_str = str(image)
    if is_url(image_str):
        return image_str
    return Path(image_str).expanduser().resolve().as_uri()


def apply_gpu_selection(gpu_id: str | None) -> None:
    """Limit visible CUDA devices before torch/transformers are initialized."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if gpu_id is None or gpu_id == "":
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def choose_input_device(device_arg: str) -> str:
    import torch

    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_processor(
    model_id: str = DEFAULT_MODEL_ID,
    adapter_path: str | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    attn_implementation: str = "default",
    load_in_4bit: bool = False,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "local_files_only": local_files_only,
    }
    if attn_implementation != "default":
        model_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    processor_kwargs: dict[str, Any] = {}
    processor_kwargs["local_files_only"] = local_files_only
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    except OSError as exc:
        raise OSError(
            f"Cannot load model or processor from {model_id!r}. If this machine cannot access "
            "HuggingFace, download Qwen/Qwen2.5-VL-3B-Instruct on a machine with network access, "
            "copy the complete model directory to this server, then run with "
            "--model-id /path/to/Qwen2.5-VL-3B-Instruct --local-files-only."
        ) from exc
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Loading a LoRA adapter requires peft. Install it with `pip install peft`.") from exc
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=local_files_only)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def build_messages(
    image_uris: Sequence[str],
    question: str,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    user_content = [{"type": "image", "image": image_uri} for image_uri in image_uris]
    user_content.append({"type": "text", "text": question})

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def build_text_messages(
    question: str,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def _generation_kwargs(max_new_tokens: int, temperature: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        kwargs.update({"do_sample": False})
    return kwargs


def generate_from_messages(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    input_device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    return batch_generate_from_messages(
        model=model,
        processor=processor,
        messages_batch=[messages],
        input_device=input_device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )[0]


def generate_answer(
    model: Any,
    processor: Any,
    image_uris: Sequence[str],
    question: str,
    input_device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> str:
    messages = build_messages(image_uris=image_uris, question=question, system_prompt=system_prompt)
    return generate_from_messages(
        model=model,
        processor=processor,
        messages=messages,
        input_device=input_device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def generate_answer_profiled(
    model: Any,
    processor: Any,
    image_uris: Sequence[str],
    question: str,
    input_device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> tuple[str, dict[str, float | int]]:
    """Generate one answer and measure end-to-end latency, TTFT and token rate.

    TTFT starts immediately before image/text preprocessing and ends when the
    first decoded output token becomes available from ``generate``.
    """
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import TextIteratorStreamer

    messages = build_messages(image_uris, question, system_prompt)
    started = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(input_device)
    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    kwargs = dict(inputs)
    kwargs.update(_generation_kwargs(max_new_tokens, temperature))
    kwargs["streamer"] = streamer
    error: list[BaseException] = []

    def run_generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(**kwargs)
        except BaseException as exc:  # propagated on the consumer thread
            error.append(exc)

    worker = threading.Thread(target=run_generate, daemon=True)
    worker.start()
    pieces: list[str] = []
    first_token_time: float | None = None
    for piece in streamer:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        pieces.append(piece)
    worker.join()
    if error:
        raise error[0]
    if input_device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    finished = time.perf_counter()
    answer = "".join(pieces).strip()
    output_tokens = len(processor.tokenizer.encode(answer, add_special_tokens=False))
    ttft = (first_token_time or finished) - started
    decode_seconds = max(0.0, finished - (first_token_time or finished))
    return answer, {
        "e2e_seconds": finished - started,
        "ttft_seconds": ttft,
        "output_tokens": output_tokens,
        "decode_tokens_per_second": output_tokens / decode_seconds if decode_seconds > 0 else 0.0,
    }


def batch_generate_from_messages(
    model: Any,
    processor: Any,
    messages_batch: Sequence[list[dict[str, Any]]],
    input_device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> list[str]:
    import torch
    from qwen_vl_utils import process_vision_info

    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(input_device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **_generation_kwargs(max_new_tokens=max_new_tokens, temperature=temperature),
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [text.strip() for text in output_texts]
