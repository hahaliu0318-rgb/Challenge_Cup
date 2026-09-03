"""LoRA fine-tuning for Qwen2.5-VL on processed full_meta JSONL data.

The source JSONL and images are read-only. Adapter checkpoints are written to
the configured output directory.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_MODEL_ID = "models/Qwen2.5-VL-3B-Instruct"
DEFAULT_TRAIN_FILE = Path("/opt/challenge-data/datasets/processed/qwen/qwen_train_full_meta.jsonl")
DEFAULT_VAL_FILE = Path("/opt/challenge-data/datasets/processed/qwen/qwen_val_full_meta.jsonl")
DEFAULT_OUTPUT_DIR = Path("/opt/challenge-work/qwen25vl_lora_outputs")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for remote sensing image understanding. "
    "Answer the user's question concisely and directly."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with LoRA.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="qwen25vl_lora_rs")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gpu-id", help="CUDA GPU id(s) to expose, e.g. 0 or 1 or 0,1.")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume Trainer state from a checkpoint directory, e.g. output-dir/checkpoint-13000.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--eval-ratio", type=float, default=0.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-system-prompt", action="store_true")

    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", choices=["default", "flash_attention_2", "sdpa", "eager"], default="default")
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for PEFT LoRA.",
    )
    parser.add_argument("--freeze-vision-tower", action="store_true")
    parser.add_argument("--report-to", default="none", help="Trainer report_to value, e.g. none, tensorboard, wandb.")
    return parser.parse_args()


def apply_gpu_selection(gpu_id: str | None) -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def print_cuda_selection() -> None:
    import torch

    print(f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', '')}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    visible_count = torch.cuda.device_count()
    print(f"Visible CUDA device count: {visible_count}")
    for index in range(visible_count):
        print(f"  cuda:{index} -> {torch.cuda.get_device_name(index)}")


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "file", "data"}


def image_to_qwen_uri(image: str | Path) -> str:
    image_str = str(image)
    if is_url(image_str):
        return image_str
    return Path(image_str).expanduser().resolve().as_uri()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "samples", "annotations", "train", "validation", "val"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"Cannot find a list of training samples in {path}.")


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".json":
        return read_json(path)

    try:
        return read_jsonl(path)
    except ValueError:
        return read_json(path)


def infer_data_root(annotation_file: Path) -> Path:
    resolved = annotation_file.resolve()
    parts = resolved.parts
    if "datasets" in parts:
        index = parts.index("datasets")
        if index > 0:
            return Path(*parts[:index])
    return Path.cwd()


def resolve_image_path(image_ref: str, data_root: Path, annotation_file: Path) -> Path | str:
    if is_url(image_ref):
        return image_ref

    raw = Path(image_ref)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([data_root / image_ref, annotation_file.parent / image_ref, Path.cwd() / image_ref])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve image path {image_ref!r}. Tried data_root={data_root}, "
        f"annotation_dir={annotation_file.parent}, cwd={Path.cwd()}."
    )


def strip_image_tokens(text: str) -> str:
    return text.replace("<image>", "").strip()


def conversation_value(record: dict[str, Any], speaker: str) -> str | None:
    for turn in record.get("conversations", []):
        if turn.get("from") == speaker and turn.get("value") is not None:
            return str(turn["value"])
    return None


def ensure_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def infer_task_from_id(sample_id: Any) -> str | None:
    text = str(sample_id or "").lower()
    for task in ("caption", "vqa", "counting", "grounding", "change"):
        if f"_{task}_" in text or text.endswith(f"_{task}"):
            return task
    return None


@dataclass
class TrainSample:
    id: str
    images: list[str]
    question: str
    answer: str
    task: str | None


def normalize_sample(record: dict[str, Any]) -> TrainSample:
    images = ensure_str_list(record.get("images") or record.get("image"))
    question = record.get("question") or conversation_value(record, "human")
    answer = record.get("answer") or conversation_value(record, "gpt")
    sample_id = record.get("id") or record.get("question_id") or record.get("image_id")

    if not images or not question or answer is None:
        raise ValueError(f"Sample is missing images/question/answer: {record}")

    return TrainSample(
        id=str(sample_id),
        images=images,
        question=strip_image_tokens(str(question)),
        answer=str(answer),
        task=record.get("task") or infer_task_from_id(sample_id),
    )


class FullMetaDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        annotation_file: Path,
        data_root: Path,
        system_prompt: str | None,
    ) -> None:
        self.samples = [normalize_sample(record) for record in records]
        self.annotation_file = annotation_file
        self.data_root = data_root
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        resolved_images = [resolve_image_path(image, self.data_root, self.annotation_file) for image in sample.images]
        image_uris = [image_to_qwen_uri(image) for image in resolved_images]
        user_content = [{"type": "image", "image": image_uri} for image_uri in image_uris]
        user_content.append({"type": "text", "text": sample.question})

        prompt_messages: list[dict[str, Any]] = []
        if self.system_prompt:
            prompt_messages.append({"role": "system", "content": self.system_prompt})
        prompt_messages.append({"role": "user", "content": user_content})

        full_messages = list(prompt_messages)
        full_messages.append({"role": "assistant", "content": sample.answer})

        return {
            "id": sample.id,
            "task": sample.task,
            "messages": full_messages,
            "prompt_messages": prompt_messages,
        }


class QwenVLDataCollator:
    def __init__(self, processor: Any, max_length: int | None = None) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        messages_batch = [feature["messages"] for feature in features]
        prompt_messages_batch = [feature["prompt_messages"] for feature in features]

        texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in messages_batch
        ]
        prompt_texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in prompt_messages_batch
        ]

        image_inputs, video_inputs = process_vision_info(messages_batch)
        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.max_length and self.max_length > 0:
            processor_kwargs.update({"truncation": True, "max_length": self.max_length})
        inputs = self.processor(**processor_kwargs)

        labels = inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        for row_index, prompt_text in enumerate(prompt_texts):
            prompt_ids = self.processor.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=bool(self.max_length and self.max_length > 0),
                max_length=self.max_length if self.max_length and self.max_length > 0 else None,
            ).input_ids
            prompt_len = min(len(prompt_ids), labels.shape[1])
            labels[row_index, :prompt_len] = -100

        inputs["labels"] = labels
        return inputs


def sample_records(records: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or max_samples >= len(records):
        return records
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[index] for index in indices[:max_samples]]


def split_train_eval(records: list[dict[str, Any]], eval_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if eval_ratio <= 0:
        return records, []
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * eval_ratio))
    return shuffled[eval_size:], shuffled[:eval_size]


def torch_dtype_from_arg(value: str):
    if value == "auto":
        return "auto"
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def freeze_vision_tower_if_present(model: Any) -> None:
    for attr in ("visual", "vision_tower"):
        module = getattr(model, attr, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        for param in model.model.visual.parameters():
            param.requires_grad = False


def main() -> int:
    args = parse_args()
    apply_gpu_selection(args.gpu_id)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments
    print_cuda_selection()

    train_file = args.train_file
    train_data_root = args.data_root or infer_data_root(train_file)
    system_prompt = None if args.no_system_prompt else args.system_prompt

    train_records = read_records(train_file)
    eval_records: list[dict[str, Any]] = []
    if args.validation_file and args.validation_file.exists():
        eval_records = read_records(args.validation_file)
    elif args.eval_ratio > 0:
        train_records, eval_records = split_train_eval(train_records, args.eval_ratio, args.seed)

    train_records = sample_records(train_records, args.max_train_samples, args.seed)
    eval_records = sample_records(eval_records, args.max_eval_samples, args.seed + 1)
    train_task_counts = Counter(normalize_sample(record).task or "unknown" for record in train_records)
    eval_task_counts = Counter(normalize_sample(record).task or "unknown" for record in eval_records)
    print(f"Loaded train samples: {len(train_records)}")
    print(f"Train task counts: {json.dumps(dict(sorted(train_task_counts.items())), ensure_ascii=False)}")
    if eval_records:
        print(f"Loaded eval samples: {len(eval_records)}")
        print(f"Eval task counts: {json.dumps(dict(sorted(eval_task_counts.items())), ensure_ascii=False)}")

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_from_arg(args.torch_dtype),
        "device_map": "auto",
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation != "default":
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=args.local_files_only,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if args.freeze_vision_tower:
        freeze_vision_tower_if_present(model)

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = FullMetaDataset(train_records, train_file, train_data_root, system_prompt)
    eval_dataset = None
    if eval_records:
        eval_file = args.validation_file if args.validation_file and args.validation_file.exists() else train_file
        eval_data_root = args.data_root or infer_data_root(eval_file)
        eval_dataset = FullMetaDataset(eval_records, eval_file, eval_data_root, system_prompt)

    output_dir = args.output_dir.expanduser().resolve()
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": args.run_name,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "seed": args.seed,
    }
    eval_strategy_value = "steps" if eval_dataset is not None else "no"
    training_arg_names = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" in training_arg_names:
        training_kwargs["eval_strategy"] = eval_strategy_value
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy_value
    training_args = TrainingArguments(**training_kwargs)

    collator = QwenVLDataCollator(processor=processor, max_length=args.max_length)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    resume_from_checkpoint = None
    if args.resume_from_checkpoint is not None:
        resume_from_checkpoint = str(args.resume_from_checkpoint.expanduser().resolve())
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(str(output_dir / "final_adapter"))
    print(f"Saved final LoRA adapter to {output_dir / 'final_adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
