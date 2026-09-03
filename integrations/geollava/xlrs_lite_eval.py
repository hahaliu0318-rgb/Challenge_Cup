#!/usr/bin/env python3

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import time


OFFICIAL_TASK_PAIRS = (
    "Complex reasoning/Anomaly Detection and Interpretation",
    "Complex reasoning/Environmental condition reasoning",
    "Complex reasoning/Route planning",
    "Counting/Counting with changing detection",
    "Counting/Counting with complex reasoning",
    "Counting/Overall counting",
    "Counting/Regional counting",
    "Land use classification/Overall Land use classification",
    "Land use classification/Regional Land use classification",
    "Object properties/Object classification",
    "Object properties/Object color",
    "Object properties/Object motion state",
    "Object spatial relationship/Object spatial relationship",
)


def extract_answer(response):
    """Mirror lmms-eval v0.3.5 XLRS answer extraction deterministically."""
    if type(response) is dict:
        response = ""
    if not isinstance(response, str):
        response = str(response or "")
    response = response.strip()
    for prefix in (
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option isThe correct option is",
        "Best answer:Best option:",
    ):
        response = response.replace(prefix, "")

    if not re.search("[ABCDE]", response):
        return ""
    matches = re.findall(r"\(([a-eA-E])\)", response)
    if not matches:
        matches = re.findall(r"(?:^|\s)?([a-eA-E])(?:$|[\s,.])?", response)
    if not matches:
        matches = re.findall(r"[a-eA-E]", response)
    return "".join(sorted({match.upper() for match in matches}))


def is_exact_match(predicted, answer):
    return set(predicted or "") == set(answer or "")


def task_pair_for(record):
    category = record["category"]
    if category in OFFICIAL_TASK_PAIRS:
        return category
    l2_category = record.get("l2_category", record.get("l2-category"))
    return f"{category}/{l2_category}"


def record_id_for(record):
    """Return the evaluator-wide stable ID, not XLRS's task-local index."""
    return int(record.get("_record_id", record["index"]))


def build_question(record):
    question = record["question"].rstrip()
    options = record["multi-choice options"]
    if isinstance(options, str):
        return question + "\n" + options.lstrip()

    option_prompt = "The choices are listed below:\n" + "\n".join(options) + "\n"
    pair = task_pair_for(record)
    if pair == "Land use classification/Overall Land use classification":
        post_prompt = (
            "\nSelect the best answer(s) for the multiple-choice question based on "
            "the image. There may be more than one correct option. Only respond "
            "with the letter(s) corresponding to the correct answer(s) (A, B, C, D), "
            "with multiple choices separated by spaces.The answer(s) is(are):"
        )
    else:
        post_prompt = (
            "\nSelect the best answer for the multiple-choice question based on the "
            "image. Only respond with the letter corresponding to the correct answer "
            "(A, B, C, D).\nThe answer is:"
        )
    return question + option_prompt + post_prompt


def build_chat_prompt(question, image_count=1, conv_template="vicuna_v1"):
    if conv_template != "vicuna_v1":
        raise ValueError(f"unsupported conversation template: {conv_template}")
    system = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        "questions."
    )
    image_tokens = " ".join(["<image>"] * image_count)
    return f"{system} USER: {image_tokens}\n{question} ASSISTANT:"


def load_records(dataset_root):
    dataset_root = Path(dataset_root)
    data_file = dataset_root / "data.jsonl"
    if not data_file.is_file():
        raise FileNotFoundError(f"dataset annotation not found: {data_file}")

    records = []
    seen_indices = set()
    for line_number, line in enumerate(
        data_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = {
            "index",
            "question",
            "multi-choice options",
            "answer",
            "category",
            "l2-category",
            "image_paths",
        } - set(record)
        if missing:
            raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
        index = int(record["index"])
        if index in seen_indices:
            raise ValueError(f"duplicate index: {index}")
        seen_indices.add(index)
        if not record["image_paths"]:
            raise ValueError(f"index {index} has no image paths")
        for relative_path in record["image_paths"]:
            image_path = dataset_root / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(f"index {index} image not found: {image_path}")
        if not record["answer"] or not set(record["answer"]).issubset(set("ABCDE")):
            raise ValueError(f"index {index} has invalid answer: {record['answer']!r}")
        pair = task_pair_for(record)
        if pair not in OFFICIAL_TASK_PAIRS:
            raise ValueError(f"index {index} has unknown official task pair: {pair}")
        records.append(record)
    return records


class HFDiskRecordSource:
    dataset_kind = "hf_disk"

    def __init__(self, dataset):
        if "image" not in dataset.column_names:
            raise ValueError("Hugging Face dataset is missing the image column")
        self._dataset = dataset
        metadata_dataset = dataset.remove_columns(["image"])
        self.metadata_records = []
        self._position_by_record_id = {}
        for position, record in enumerate(metadata_dataset):
            metadata_record = dict(record)
            metadata_record["_record_id"] = position
            metadata_record["_dataset_position"] = position
            self.metadata_records.append(metadata_record)
            self._position_by_record_id[position] = position

    @classmethod
    def from_disk(cls, dataset_root):
        from datasets import DatasetDict, load_from_disk

        loaded = load_from_disk(str(dataset_root))
        if isinstance(loaded, DatasetDict):
            if "train" not in loaded:
                raise ValueError("Hugging Face dataset does not contain train split")
            loaded = loaded["train"]
        return cls(loaded)

    def materialize(self, metadata_record):
        record_id = record_id_for(metadata_record)
        if record_id not in self._position_by_record_id:
            raise KeyError(f"dataset record ID not found: {record_id}")
        position = self._position_by_record_id[record_id]
        record = dict(self._dataset[position])
        source_index = int(metadata_record["index"])
        if int(record["index"]) != source_index:
            raise RuntimeError(
                f"dataset position/source-index mismatch: expected {source_index}, "
                f"got {record['index']}"
            )
        return record


class LocalJsonlRecordSource:
    dataset_kind = "local_jsonl"

    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.metadata_records = load_records(self.dataset_root)

    def materialize(self, metadata_record):
        return metadata_record


def open_record_source(dataset_root):
    dataset_root = Path(dataset_root)
    if (dataset_root / "data.jsonl").is_file():
        return LocalJsonlRecordSource(dataset_root)
    if (dataset_root / "dataset_dict.json").is_file() or (
        dataset_root / "train" / "state.json"
    ).is_file():
        return HFDiskRecordSource.from_disk(dataset_root)
    raise FileNotFoundError(
        f"unsupported dataset root (no data.jsonl or saved Arrow dataset): "
        f"{dataset_root}"
    )


def load_predictions(path):
    path = Path(path)
    if not path.exists():
        return {}
    predictions = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        index = int(row["index"])
        if index in predictions:
            raise ValueError(f"duplicate prediction index {index} on line {line_number}")
        predictions[index] = row
    return predictions


def rewrite_predictions(path, predictions, discard_errors=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for index in sorted(predictions):
            row = predictions[index]
            if discard_errors and row.get("error"):
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)


def aggregate_predictions(predictions, expected_total=None):
    rows = list(predictions.values()) if isinstance(predictions, dict) else list(predictions)
    per_pair = defaultdict(lambda: {"count": 0, "correct": 0})
    per_l1 = defaultdict(lambda: {"count": 0, "correct": 0})
    correct_total = 0
    parse_failures = 0
    errors = 0
    generation_seconds = []

    for row in rows:
        pair = task_pair_for(row)
        category = pair.split("/", 1)[0]
        correct = is_exact_match(row.get("parsed_answer", ""), row.get("answer", ""))
        correct_total += int(correct)
        per_pair[pair]["count"] += 1
        per_pair[pair]["correct"] += int(correct)
        per_l1[category]["count"] += 1
        per_l1[category]["correct"] += int(correct)
        parse_failures += int(not row.get("parsed_answer"))
        errors += int(bool(row.get("error")))
        if row.get("timing_seconds", {}).get("generation") is not None:
            generation_seconds.append(float(row["timing_seconds"]["generation"]))

    def finalize(groups):
        return {
            key: {
                "count": value["count"],
                "correct": value["correct"],
                "accuracy": value["correct"] / value["count"],
            }
            for key, value in sorted(groups.items())
        }

    finalized_pairs = finalize(per_pair)
    finalized_l1 = finalize(per_l1)
    observed_macro = (
        statistics.fmean(item["accuracy"] for item in finalized_pairs.values())
        if finalized_pairs
        else None
    )
    observed_pairs = set(finalized_pairs)
    full_macro_available = set(OFFICIAL_TASK_PAIRS).issubset(observed_pairs)
    official_full_macro = (
        statistics.fmean(finalized_pairs[pair]["accuracy"] for pair in OFFICIAL_TASK_PAIRS)
        if full_macro_available
        else None
    )
    expected_total = len(rows) if expected_total is None else int(expected_total)

    return {
        "metric_scope": "XLRS-Bench-lite available subset",
        "scoring_rule": "set(parsed_answer) == set(reference_answer)",
        "prediction_count": len(rows),
        "expected_prediction_count": expected_total,
        "correct_count": correct_total,
        "micro_accuracy": correct_total / len(rows) if rows else None,
        "observed_task_macro_accuracy": observed_macro,
        "observed_task_pair_count": len(finalized_pairs),
        "official_task_pair_count": len(OFFICIAL_TASK_PAIRS),
        "official_full_macro_available": full_macro_available,
        "official_full_macro_accuracy": official_full_macro,
        "coverage_complete": len(rows) == expected_total,
        "parse_failure_count": parse_failures,
        "inference_error_count": errors,
        "per_l1_category": finalized_l1,
        "per_task_pair": finalized_pairs,
        "timing_seconds": {
            "generation_sum": round(sum(generation_seconds), 6),
            "generation_mean": (
                round(statistics.fmean(generation_seconds), 6)
                if generation_seconds
                else None
            ),
        },
    }


def is_official_full_evaluation(
    metrics,
    *,
    dataset_kind,
    dataset_total,
    selected_total,
):
    return (
        dataset_kind == "hf_disk"
        and dataset_total == 3080
        and selected_total == 3080
        and metrics["coverage_complete"]
        and metrics["inference_error_count"] == 0
        and metrics["observed_task_pair_count"] == len(OFFICIAL_TASK_PAIRS)
    )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def collect_gpu_memory_mib(cuda):
    mib = 1024**2
    return [
        {
            "logical_device": index,
            "name": cuda.get_device_name(index),
            "allocated": round(cuda.memory_allocated(index) / mib, 1),
            "reserved": round(cuda.memory_reserved(index) / mib, 1),
            "peak_allocated": round(cuda.max_memory_allocated(index) / mib, 1),
            "peak_reserved": round(cuda.max_memory_reserved(index) / mib, 1),
        }
        for index in range(cuda.device_count())
    ]


def synchronize_all_cuda_devices(cuda):
    for index in range(cuda.device_count()):
        cuda.synchronize(index)


def load_runtime(model_path, attention, max_memory_per_gpu_gib):
    import flash_attn
    import torch
    import transformers
    from longva.model.builder import load_pretrained_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    max_memory = {
        **{
            index: f"{max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        },
        "cpu": "64GiB",
    }
    started = time.perf_counter()
    tokenizer, model, image_processor, context_length = load_pretrained_model(
        str(model_path),
        None,
        "llava_qwen",
        device_map="auto",
        attn_implementation=attention,
        max_memory=max_memory,
    )
    model.eval()
    synchronize_all_cuda_devices(torch.cuda)
    meta_parameter_count = sum(int(parameter.is_meta) for parameter in model.parameters())
    if meta_parameter_count:
        raise RuntimeError(
            f"model has {meta_parameter_count} unresolved meta parameters after loading"
        )
    vision_parameter = next(model.get_vision_tower().parameters())
    vision_parameter_abs_mean = float(
        vision_parameter.detach().float().abs().mean().item()
    )
    return {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "context_length": int(context_length),
        "load_seconds": time.perf_counter() - started,
        "versions": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "flash_attn": flash_attn.__version__,
        },
        "memory_after_load": collect_gpu_memory_mib(torch.cuda),
        "model_parameter_checks": {
            "meta_parameter_count": meta_parameter_count,
            "vision_first_parameter_abs_mean": vision_parameter_abs_mean,
        },
    }


def load_record_images(record, dataset_root):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    images = []
    image_metadata = []
    if "image_paths" in record:
        for relative_path in record["image_paths"]:
            image_path = Path(dataset_root) / relative_path
            with Image.open(image_path) as source:
                original_mode = source.mode
                rgb = source.convert("RGB")
                rgb.load()
            images.append(rgb)
            image_metadata.append(
                {
                    "path": str(image_path),
                    "source": "local_file",
                    "width": rgb.width,
                    "height": rgb.height,
                    "original_mode": original_mode,
                }
            )
    elif "image" in record:
        for source in record["image"]:
            original_mode = source.mode
            original_filename = getattr(source, "filename", None)
            rgb = source.convert("RGB")
            rgb.load()
            source.close()
            images.append(rgb)
            image_metadata.append(
                {
                    "path": str(original_filename) if original_filename else None,
                    "source": "hf_arrow",
                    "width": rgb.width,
                    "height": rgb.height,
                    "original_mode": original_mode,
                }
            )
    else:
        raise ValueError(f"index {record.get('index')} has no image field")
    if not images:
        raise ValueError(f"index {record.get('index')} has no images")
    return images, image_metadata


def infer_record(record, dataset_root, runtime, conv_template, max_new_tokens):
    import torch
    from longva.constants import IMAGE_TOKEN_INDEX
    from longva.mm_utils import process_images, tokenizer_image_token

    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    image_processor = runtime["image_processor"]

    images, image_metadata = load_record_images(record, dataset_root)
    try:
        chat_prompt = build_chat_prompt(
            build_question(record),
            image_count=len(images),
            conv_template=conv_template,
        )

        preprocess_started = time.perf_counter()
        image_tensor = process_images(images, image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [item.to(model.device, dtype=torch.float16) for item in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(
            chat_prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(model.device)
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        attention_mask = input_ids.ne(pad_token_id).to(model.device)
        synchronize_all_cuda_devices(torch.cuda)
        preprocess_seconds = time.perf_counter() - preprocess_started

        generation_started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                images=image_tensor,
                image_sizes=[image.size for image in images],
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1,
                use_cache=True,
                max_new_tokens=max_new_tokens,
            )
        synchronize_all_cuda_devices(torch.cuda)
        generation_seconds = time.perf_counter() - generation_started
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return {
            "response": response,
            "parsed_answer": extract_answer(response),
            "image_metadata": image_metadata,
            "prompt_token_count": int(input_ids.shape[-1]),
            "generated_token_count": int(output_ids.shape[-1]),
            "timing_seconds": {
                "preprocess": round(preprocess_seconds, 6),
                "generation": round(generation_seconds, 6),
            },
        }
    finally:
        for image in images:
            image.close()


def select_records(records, indices, limit):
    if indices:
        requested = {int(value) for value in indices.split(",") if value.strip()}
        selected = [record for record in records if record_id_for(record) in requested]
        found = {record_id_for(record) for record in selected}
        if found != requested:
            raise ValueError(f"requested indices not found: {sorted(requested - found)}")
    else:
        selected = list(records)
    if limit:
        selected = selected[:limit]
    return selected


def run_evaluation(args):
    import torch

    dataset_root = Path(args.dataset_root).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "run_config.json"

    record_source = open_record_source(dataset_root)
    all_records = record_source.metadata_records
    selected = select_records(all_records, args.indices, args.limit)
    selected_indices = {record_id_for(record) for record in selected}
    existing = load_predictions(predictions_path)
    unexpected = set(existing) - selected_indices
    if unexpected:
        raise ValueError(
            f"output contains indices outside this selection: {sorted(unexpected)[:10]}"
        )
    if args.retry_errors and any(row.get("error") for row in existing.values()):
        rewrite_predictions(predictions_path, existing, discard_errors=True)
        existing = load_predictions(predictions_path)
    completed = {
        index
        for index, row in existing.items()
        if not row.get("error") or not args.retry_errors
    }
    pending = [record for record in selected if record_id_for(record) not in completed]

    print(
        json.dumps(
            {
                "event": "evaluation_start",
                "selected": len(selected),
                "resumed": len(selected) - len(pending),
                "pending": len(pending),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if pending:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        runtime = load_runtime(
            model_path,
            attention=args.attention,
            max_memory_per_gpu_gib=args.max_memory_per_gpu_gib,
        )
    else:
        runtime = None

    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "dataset_kind": record_source.dataset_kind,
        "dataset_annotation": str(
            dataset_root
            / (
                "data.jsonl"
                if record_source.dataset_kind == "local_jsonl"
                else "dataset_dict.json"
            )
        ),
        "dataset_total_records": len(all_records),
        "selected_record_count": len(selected),
        "selected_indices": sorted(selected_indices),
        "record_identity": {
            "prediction_index": (
                "Arrow row position" if record_source.dataset_kind == "hf_disk"
                else "source index"
            ),
            "source_index_field": "source_index",
        },
        "model": "initiacms/GeoLLaVA-8K",
        "model_path": str(model_path),
        "generation": {
            "batch_size": 1,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "conv_template": args.conv_template,
        },
        "runtime": {
            "attention": args.attention,
            "device_map": "auto",
            "max_memory_per_gpu_gib": args.max_memory_per_gpu_gib,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hf_home": os.environ.get("HF_HOME"),
            "seed": args.seed,
            "versions": runtime["versions"] if runtime else None,
            "model_load_seconds": round(runtime["load_seconds"], 6) if runtime else None,
            "gpu_memory_after_load_mib": runtime["memory_after_load"] if runtime else None,
            "model_parameter_checks": (
                runtime["model_parameter_checks"] if runtime else None
            ),
        },
        "metric_reference": {
            "implementation": "lmms-eval v0.3.5 lmms_eval/tasks/xlrs/mcq_utils.py",
            "micro": "exact option-set matches / all samples",
            "macro": "mean of task-pair accuracies",
        },
    }
    write_json(config_path, config)

    run_started = time.perf_counter()
    stopped_on_error = False
    for ordinal, record in enumerate(pending, start=1):
        index = record_id_for(record)
        source_index = int(record["index"])
        pair = task_pair_for(record)
        item_started = time.perf_counter()
        base = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "index": index,
            "source_index": source_index,
            "category": record["category"],
            "l2_category": record["l2-category"],
            "task_pair": pair,
            "answer": record["answer"],
            "question": record["question"],
            "image_paths": list(record.get("image_paths", [])),
            "dataset_path": record.get("path"),
        }
        try:
            materialized_record = record_source.materialize(record)
            inference = infer_record(
                materialized_record,
                dataset_root,
                runtime,
                conv_template=args.conv_template,
                max_new_tokens=args.max_new_tokens,
            )
            row = {
                **base,
                **inference,
                "correct": is_exact_match(
                    inference["parsed_answer"], record["answer"]
                ),
                "error": None,
            }
        except Exception as error:
            row = {
                **base,
                "response": "",
                "parsed_answer": "",
                "correct": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "timing_seconds": {
                    "total": round(time.perf_counter() - item_started, 6)
                },
            }
            stopped_on_error = args.fail_fast
        append_jsonl(predictions_path, row)
        existing[index] = row
        elapsed = time.perf_counter() - run_started
        done_now = ordinal
        rate = elapsed / done_now
        eta = rate * (len(pending) - done_now)
        print(
            json.dumps(
                {
                    "event": "prediction",
                    "ordinal": ordinal,
                    "pending_total": len(pending),
                    "index": index,
                    "task_pair": pair,
                    "parsed_answer": row["parsed_answer"],
                    "answer": row["answer"],
                    "correct": row["correct"],
                    "error": row["error"],
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(eta, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stopped_on_error:
            break

    selected_predictions = {
        index: row for index, row in existing.items() if index in selected_indices
    }
    metrics = aggregate_predictions(
        selected_predictions,
        expected_total=len(selected),
    )
    full_evaluation = is_official_full_evaluation(
        metrics,
        dataset_kind=record_source.dataset_kind,
        dataset_total=len(all_records),
        selected_total=len(selected),
    )
    if not full_evaluation:
        metrics["official_full_macro_available"] = False
        metrics["official_full_macro_accuracy"] = None
    if full_evaluation:
        metrics["metric_scope"] = "XLRS-Bench-lite full pinned snapshot (3,080 rows)"
    elif record_source.dataset_kind == "hf_disk":
        metrics["metric_scope"] = "XLRS-Bench-lite full-snapshot smoke subset"
    elif len(all_records) == 206 and len(selected) == 206:
        metrics["metric_scope"] = "XLRS-Bench-lite part0 subset (206 local rows)"
    else:
        metrics["metric_scope"] = "XLRS-Bench-lite part0 smoke subset"
    metrics["wall_seconds_this_invocation"] = round(
        time.perf_counter() - run_started, 6
    )
    metrics["gpu_memory_mib"] = collect_gpu_memory_mib(torch.cuda)
    write_json(metrics_path, metrics)
    print(json.dumps({"event": "metrics", **metrics}, ensure_ascii=False), flush=True)
    return 1 if stopped_on_error or metrics["inference_error_count"] else 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GeoLLaVA-8K on local or saved-Arrow XLRS-Bench-lite."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--indices",
        default="",
        help="comma-separated stable record IDs (Arrow row positions for hf_disk)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--conv-template", default="vicuna_v1")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    raise SystemExit(run_evaluation(parse_args()))


if __name__ == "__main__":
    main()
