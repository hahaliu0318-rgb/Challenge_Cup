"""Evaluate Qwen2.5-VL on processed full_meta JSONL data.

The input file is read-only. Summary CSV and per-sample predictions are written
under the configured output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from qwen25vl_utils import (
    DEFAULT_MODEL_ID,
    apply_gpu_selection,
    choose_input_device,
    generate_answer_profiled,
    image_to_qwen_uri,
    is_url,
    load_model_and_processor,
)


DEFAULT_DATA_PATH = Path("/opt/challenge-data/datasets/processed/qwen/qwen_val_full_meta.jsonl")
SUMMARY_COLUMNS = [
    "Model",
    "Data",
    "Success",
    "Failure",
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "CLAIR", "Avg_L",
    "VQA Category", "VQA Presence", "VQA Quantity", "VQA Color", "VQA Shape",
    "VQA Size", "VQA Position", "VQA Direction", "VQA Scene", "VQA Reasoning",
    "VQA Unknown", "VQA All",
    "Count MAE",
    "Count Acc±1",
    "Ground Unique@0.5", "Ground Unique@0.7",
    "Ground Non-Unique@0.5", "Ground Non-Unique@0.7",
    "Ground All@0.5", "Ground All@0.7",
    "Total Params", "LoRA Params", "Model size GiB", "Peak GPU GiB", "Peak CPU GiB",
    "Mean E2E s", "P95 E2E s", "Mean TTFT s", "P95 TTFT s", "Decode tok/s",
]


@dataclass
class EvalSample:
    id: str
    dataset: str | None
    task: str
    images: list[str]
    question: str
    answer: str
    references: list[str]
    bboxes: Any | None
    bbox_format: str | None
    eval_meta: dict[str, Any]
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on full_meta JSONL data.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional PEFT LoRA adapter directory, for example qwen25vl_lora_outputs/.../final_adapter.",
    )
    parser.add_argument("--annotation-file", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--sample-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional JSONL/details file or plain-text ID list. When provided, select exactly "
            "these annotation records in file order and bypass sampling/max-samples/shuffle."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root used to resolve relative image paths. If omitted, inferred from annotation path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Total number of samples to evaluate. Use <=0 to evaluate all selected records.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help=(
            "Offset before sampling. For sequential sampling this is a global offset; "
            "for balanced-task sampling it is applied inside each task group."
        ),
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["balanced-task", "sequential"],
        default="balanced-task",
        help="Sample records evenly by task, or keep the old sequential behavior.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--standardized-random-sample",
        action="store_true",
        help=(
            "Randomly sample each non-VQA top-level task and each canonical VQA subtype. "
            "This bypasses --sampling-strategy/--max-samples/--start-index/--shuffle."
        ),
    )
    parser.add_argument("--samples-per-task", type=int, default=1000)
    parser.add_argument("--samples-per-vqa-subtype", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/challenge-work/qwen25vl_eval_results"),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument(
        "--gpu-id",
        help="CUDA GPU id(s) to expose, e.g. 0 or 1 or 0,1. Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["default", "flash_attention_2", "sdpa", "eager"],
        default="default",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse samples and resolve image paths. Do not load the model.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return records


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

    raise ValueError(f"Cannot find a list of samples in {path}.")


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".json":
        return read_json(path)

    try:
        return read_jsonl(path)
    except ValueError:
        return read_json(path)


def record_id(record: dict[str, Any]) -> str:
    value = record.get("id") or record.get("question_id") or record.get("Question_id")
    if value is None:
        raise ValueError(f"Annotation record has no id/question_id: {record}")
    return str(value)


def read_sample_ids(path: Path) -> list[str]:
    """Read IDs from RSThinker details JSONL, a JSON list, or one ID per line."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    ids: list[str] = []
    if not stripped:
        raise ValueError(f"Sample ID file is empty: {path}")
    if stripped.startswith("["):
        values = json.loads(text)
        if not isinstance(values, list):
            raise ValueError(f"Expected a JSON list in {path}")
        for value in values:
            if isinstance(value, dict):
                ids.append(record_id(value))
            else:
                ids.append(str(value).strip())
    else:
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    ids.append(record_id(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"Invalid ID record at {path}:{line_number}: {exc}") from exc
            else:
                ids.append(line)
    ids = [value for value in ids if value]
    duplicates = [sample_id for sample_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate sample IDs in {path}: {duplicates[:10]}")
    if not ids:
        raise ValueError(f"No sample IDs found in {path}")
    return ids


def select_records_by_ids(records: list[dict[str, Any]], ids_file: Path) -> list[dict[str, Any]]:
    requested = read_sample_ids(ids_file)
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        sample_id = record_id(record)
        if sample_id in by_id:
            duplicates.append(sample_id)
        by_id[sample_id] = record
    if duplicates:
        raise ValueError(f"Duplicate IDs in annotation file: {duplicates[:10]}")
    missing = [sample_id for sample_id in requested if sample_id not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} requested sample IDs are absent from the annotation file; "
            f"first missing IDs: {missing[:10]}"
        )
    return [by_id[sample_id] for sample_id in requested]


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
    if "_change_caption_" in text or text.endswith("_change_caption"):
        return "change_caption"
    for task in ("caption", "vqa", "counting", "grounding"):
        if f"_{task}_" in text or text.endswith(f"_{task}"):
            return task
    if "_change_" in text or text.endswith("_change"):
        return "change_caption"
    return None


def normalize_sample(record: dict[str, Any]) -> EvalSample:
    images = ensure_str_list(record.get("images") or record.get("image"))
    question = record.get("question") or conversation_value(record, "human")
    answer_value = record.get("answer") or conversation_value(record, "gpt")
    sample_id = record.get("id") or record.get("question_id") or record.get("image_id")
    task = record.get("task") or infer_task_from_id(sample_id)

    if not images or not question or answer_value is None or not task:
        raise ValueError(f"Sample is missing required fields: {record}")

    return EvalSample(
        id=str(sample_id),
        dataset=record.get("dataset"),
        task=str(task),
        images=images,
        question=strip_image_tokens(str(question)),
        answer=str(answer_value[0] if isinstance(answer_value, list) else answer_value),
        references=[str(v) for v in answer_value] if isinstance(answer_value, list) else [str(answer_value)],
        bboxes=record.get("bboxes"),
        bbox_format=record.get("bbox_format"),
        eval_meta=record.get("eval_meta") or {},
        raw=record,
    )


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
        candidates.extend(
            [
                data_root / image_ref,
                annotation_file.parent / image_ref,
                Path.cwd() / image_ref,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve image path {image_ref!r}. Tried data_root={data_root}, "
        f"annotation_dir={annotation_file.parent}, cwd={Path.cwd()}."
    )


def task_name(record: dict[str, Any]) -> str:
    sample_id = record.get("id") or record.get("question_id") or record.get("image_id")
    return str(record.get("task") or infer_task_from_id(sample_id) or "unknown")


def take_balanced_by_task(
    records: list[dict[str, Any]],
    max_samples: int,
    start_index: int,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_order = []
    for record in records:
        task = task_name(record)
        if task not in grouped:
            task_order.append(task)
        grouped[task].append(record)

    rng = random.Random(seed)
    task_records = {}
    for task in task_order:
        rows = list(grouped[task])
        if shuffle:
            rng.shuffle(rows)
        task_records[task] = rows[start_index:]

    if max_samples <= 0:
        quotas = {task: len(task_records[task]) for task in task_order}
    else:
        quotas = {task: 0 for task in task_order}
        remaining = min(max_samples, sum(len(rows) for rows in task_records.values()))
        active_tasks = [task for task in task_order if task_records[task]]

        while remaining > 0 and active_tasks:
            progressed = False
            for task in list(active_tasks):
                if remaining <= 0:
                    break
                if quotas[task] >= len(task_records[task]):
                    active_tasks.remove(task)
                    continue
                quotas[task] += 1
                remaining -= 1
                progressed = True
            if not progressed:
                break

    selected = []
    positions = {task: 0 for task in task_order}
    while len(selected) < sum(quotas.values()):
        progressed = False
        for task in task_order:
            index = positions[task]
            if index < quotas[task]:
                selected.append(task_records[task][index])
                positions[task] += 1
                progressed = True
        if not progressed:
            break
    return selected


def select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_ids_file is not None:
        return select_records_by_ids(records, args.sample_ids_file.expanduser().resolve())
    if args.sampling_strategy == "balanced-task":
        return take_balanced_by_task(
            records=records,
            max_samples=args.max_samples,
            start_index=args.start_index,
            shuffle=args.shuffle,
            seed=args.seed,
        )

    if args.shuffle:
        records = list(records)
        random.seed(args.seed)
        random.shuffle(records)
    end = None if args.max_samples <= 0 else args.start_index + args.max_samples
    return records[args.start_index : end]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu_n(prediction: str, references: list[str], n: int) -> float:
    """Sentence BLEU-n with clipped counts, brevity penalty and add-one smoothing."""
    pred = tokenize(prediction)
    refs = [tokenize(value) for value in references]
    if not pred or not refs:
        return 0.0
    precisions = []
    for order in range(1, n + 1):
        pred_counts = Counter(ngrams(pred, order))
        max_ref: Counter[tuple[str, ...]] = Counter()
        for ref in refs:
            for gram, count in Counter(ngrams(ref, order)).items():
                max_ref[gram] = max(max_ref[gram], count)
        clipped = sum(min(count, max_ref[gram]) for gram, count in pred_counts.items())
        total = sum(pred_counts.values())
        precisions.append((clipped + 1.0) / (total + 1.0))
    ref_len = min((len(ref) for ref in refs), key=lambda value: (abs(value - len(pred)), value))
    bp = 1.0 if len(pred) > ref_len else math.exp(1.0 - ref_len / max(1, len(pred)))
    return bp * math.exp(sum(math.log(value) for value in precisions) / n)


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_token == right_token else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, references: list[str]) -> float:
    pred = tokenize(prediction)
    scores = []
    for reference in references:
        ref = tokenize(reference)
        common = lcs_length(pred, ref)
        precision = common / len(pred) if pred else 0.0
        recall = common / len(ref) if ref else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return max(scores, default=0.0)


def meteor_score(prediction: str, references: list[str]) -> float:
    """Dependency-free METEOR core (exact token matches + fragmentation penalty)."""
    pred = tokenize(prediction)
    best = 0.0
    for reference in references:
        ref = tokenize(reference)
        used: set[int] = set()
        matched_positions = []
        for token in pred:
            position = next((i for i, value in enumerate(ref) if value == token and i not in used), None)
            if position is not None:
                used.add(position)
                matched_positions.append(position)
        matches = len(matched_positions)
        if not matches:
            continue
        precision, recall = matches / len(pred), matches / len(ref)
        fmean = 10 * precision * recall / (recall + 9 * precision)
        chunks = 1 + sum(b != a + 1 for a, b in zip(matched_positions, matched_positions[1:]))
        best = max(best, fmean * (1 - 0.5 * (chunks / matches) ** 3))
    return best


VQA_TYPE_ALIASES = {
    "category": {"category", "object_category", "object category", "class", "object_class"},
    "presence": {"presence", "existence", "object_presence", "yes_no"},
    "quantity": {"quantity", "count", "counting", "number"},
    "color": {"color", "colour"}, "shape": {"shape"}, "size": {"size"},
    "position": {"position", "location", "spatial_position"},
    "direction": {"direction", "orientation"}, "scene": {"scene", "scene_type"},
    "reasoning": {"reasoning", "reason", "inference"},
}


def vqa_subtype(sample: EvalSample) -> str:
    candidates = []
    for source in (sample.raw, sample.eval_meta):
        for key in ("question_type", "vqa_type", "subtask", "type", "category"):
            if source.get(key) is not None:
                candidates.append(str(source[key]).lower().strip().replace("-", "_"))
    candidates.append(sample.id.lower())
    for canonical, aliases in VQA_TYPE_ALIASES.items():
        if any(alias in candidate for candidate in candidates for alias in aliases):
            return canonical.title()
    return "Unknown"


def standardized_random_sample(
    records: list[dict[str, Any]], samples_per_task: int, samples_per_vqa_subtype: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic stratified sampling for top-level tasks and canonical VQA subtypes."""
    if samples_per_task <= 0 or samples_per_vqa_subtype <= 0:
        raise ValueError("Standardized sampling quotas must be positive")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_vqa: list[str] = []
    for record in records:
        sample = normalize_sample(record)
        if sample.task == "vqa":
            subtype = vqa_subtype(sample)
            if subtype == "Unknown":
                unknown_vqa.append(sample.id)
                continue
            groups[f"vqa/{subtype}"].append(record)
        else:
            groups[f"task/{sample.task}"].append(record)

    rng = random.Random(seed)
    chosen_by_group: dict[str, list[dict[str, Any]]] = {}
    for group in sorted(groups):
        quota = samples_per_vqa_subtype if group.startswith("vqa/") else samples_per_task
        candidates = list(groups[group])
        chosen_by_group[group] = rng.sample(candidates, min(quota, len(candidates)))

    # Interleave strata so partial runs do not contain only one task.
    selected: list[dict[str, Any]] = []
    max_group_size = max((len(rows) for rows in chosen_by_group.values()), default=0)
    for index in range(max_group_size):
        for group in sorted(chosen_by_group):
            if index < len(chosen_by_group[group]):
                selected.append(chosen_by_group[group][index])
    audit = {
        "seed": seed,
        "requested_samples_per_task": samples_per_task,
        "requested_samples_per_vqa_subtype": samples_per_vqa_subtype,
        "available": {group: len(groups[group]) for group in sorted(groups)},
        "selected": {group: len(chosen_by_group[group]) for group in sorted(chosen_by_group)},
        "unknown_vqa_excluded_count": len(unknown_vqa),
        "unknown_vqa_excluded_ids": unknown_vqa,
    }
    return selected, audit


def grounding_group(sample: EvalSample) -> str:
    for source in (sample.eval_meta, sample.raw):
        for key in ("unique", "is_unique"):
            if isinstance(source.get(key), bool):
                return "Unique" if source[key] else "Non-Unique"
        for key in ("instance_count", "num_instances", "category_count", "object_count"):
            if isinstance(source.get(key), (int, float)):
                return "Unique" if source[key] == 1 else "Non-Unique"
        value = str(source.get("uniqueness", "")).lower()
        if value:
            return "Non-Unique" if "non" in value else "Unique"
    return "Unknown"


def percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def official_caption_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """COCO-caption metrics, preserving all references for each image/sample."""
    empty = {**{f"BLEU-{n}": None for n in range(1, 5)}, "METEOR": None, "ROUGE-L": None, "CIDEr": None, "CLAIR": None, "Avg_L": None}
    if not rows:
        return empty
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as exc:
        raise RuntimeError("Caption evaluation requires pycocoevalcap; install requirements.txt") from exc
    refs = {index: list(row["references"]) for index, row in enumerate(rows)}
    hyps = {index: [str(row["prediction"])] for index, row in enumerate(rows)}
    bleu, _ = Bleu(4).compute_score(refs, hyps)
    meteor_scorer = Meteor()
    try:
        meteor, _ = meteor_scorer.compute_score(refs, hyps)
    finally:
        if hasattr(meteor_scorer, "close"):
            meteor_scorer.close()
    rouge, _ = Rouge().compute_score(refs, hyps)
    cider, _ = Cider().compute_score(refs, hyps)
    return {
        **{f"BLEU-{n}": float(bleu[n - 1]) for n in range(1, 5)},
        "METEOR": float(meteor), "ROUGE-L": float(rouge), "CIDEr": float(cider),
        "CLAIR": None,  # Requires an explicitly selected external visual-language judge.
        "Avg_L": statistics.fmean(len(tokenize(row["prediction"])) for row in rows),
    }


class PeakRssMonitor:
    def __init__(self) -> None:
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        process = psutil.Process(os.getpid())
        def poll() -> None:
            while not self._stop.wait(0.05):
                self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join()
        return self.peak_bytes / (1024 ** 3)


def directory_size_gib(paths: list[str | None]) -> float | None:
    total = 0
    found = False
    for value in paths:
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_file():
            total += path.stat().st_size
            found = True
        elif path.is_dir():
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
                    found = True
    return total / (1024 ** 3) if found else None


def cider_lite(predictions: list[str], references: list[str]) -> float | None:
    if not predictions or not references:
        return None

    ref_tokens = [tokenize(ref) for ref in references]
    pred_tokens = [tokenize(pred) for pred in predictions]
    scores = []

    for n in range(1, 5):
        doc_freq: Counter[tuple[str, ...]] = Counter()
        ref_counters = []
        pred_counters = []

        for tokens in ref_tokens:
            counter = Counter(ngrams(tokens, n))
            ref_counters.append(counter)
            doc_freq.update(counter.keys())
        for tokens in pred_tokens:
            pred_counters.append(Counter(ngrams(tokens, n)))

        if not any(ref_counters):
            continue

        num_docs = len(ref_tokens)
        n_scores = []
        for pred_counter, ref_counter in zip(pred_counters, ref_counters):
            vocab = set(pred_counter) | set(ref_counter)
            if not vocab:
                n_scores.append(0.0)
                continue
            dot = pred_norm = ref_norm = 0.0
            for gram in vocab:
                idf = math.log((num_docs + 1.0) / (doc_freq.get(gram, 0) + 1.0)) + 1.0
                pred_weight = pred_counter.get(gram, 0) * idf
                ref_weight = ref_counter.get(gram, 0) * idf
                dot += pred_weight * ref_weight
                pred_norm += pred_weight * pred_weight
                ref_norm += ref_weight * ref_weight
            denom = math.sqrt(pred_norm) * math.sqrt(ref_norm)
            n_scores.append(dot / denom if denom else 0.0)
        scores.append(sum(n_scores) / len(n_scores))

    if not scores:
        return None
    return 10.0 * sum(scores) / len(scores)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.-]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def answer_tokens(text: str) -> list[str]:
    return tokenize(normalize_answer(text))


def is_yes_no_answer(reference: str) -> bool:
    return normalize_answer(reference) in {"yes", "no"}


def extract_yes_no(text: str) -> str | None:
    tokens = answer_tokens(text)
    if "yes" in tokens and "no" not in tokens:
        return "yes"
    if "no" in tokens and "yes" not in tokens:
        return "no"
    for token in tokens:
        if token in {"yes", "no"}:
            return token
    return None


def contains_reference_answer(prediction: str, reference: str) -> bool:
    pred_tokens = answer_tokens(prediction)
    ref_tokens = answer_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return False

    if len(ref_tokens) <= 4:
        for index in range(0, len(pred_tokens) - len(ref_tokens) + 1):
            if pred_tokens[index : index + len(ref_tokens)] == ref_tokens:
                return True

    return set(ref_tokens).issubset(set(pred_tokens))


def vqa_match(prediction: str, reference: str) -> tuple[bool, str]:
    """Normalized exact-match accuracy for a single VQA reference."""
    if exact_match(prediction, reference):
        return True, "normalized_exact"

    if is_yes_no_answer(reference):
        pred_yes_no = extract_yes_no(prediction)
        if pred_yes_no == normalize_answer(reference):
            return True, "yes_no_extracted"
        return False, "yes_no_mismatch"

    return False, "mismatch"


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def extract_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group(0))
    for token in tokenize(text):
        if token in NUMBER_WORDS:
            return float(NUMBER_WORDS[token])
    return None


def target_number(sample: EvalSample) -> float | None:
    value = sample.eval_meta.get("target_number")
    if isinstance(value, (int, float)):
        return float(value)
    return extract_number(sample.answer)


def parse_bbox(text: str) -> list[float] | None:
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
        value = bbox_from_json(parsed)
        if value is not None:
            return value
    except json.JSONDecodeError:
        pass

    bbox_match = re.search(
        r"['\"]?(?:bbox_2d|bbox)['\"]?\s*:\s*\[([^\]]+)\]",
        cleaned,
        flags=re.IGNORECASE,
    )
    if bbox_match:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", bbox_match.group(1))
        if len(numbers) >= 4:
            return [float(v) for v in numbers[:4]]

    match = re.search(r"\[([^\]]+)\]", cleaned)
    if match:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        if len(numbers) >= 4:
            return [float(v) for v in numbers[:4]]
    return None


def bbox_from_json(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        bbox = value.get("bbox_2d") or value.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            return [float(v) for v in bbox[:4]]
        for nested_value in value.values():
            nested = bbox_from_json(nested_value)
            if nested is not None:
                return nested

    if isinstance(value, list):
        if len(value) >= 4 and all(isinstance(item, (int, float)) for item in value[:4]):
            return [float(v) for v in value[:4]]
        for item in value:
            nested = bbox_from_json(item)
            if nested is not None:
                return nested

    return None


def clamp_bbox(box: list[float], scale: float = 1000.0) -> list[float]:
    x1, y1, x2, y2 = box[:4]
    x1, x2 = sorted((max(0.0, min(scale, x1)), max(0.0, min(scale, x2))))
    y1, y2 = sorted((max(0.0, min(scale, y1)), max(0.0, min(scale, y2))))
    return [x1, y1, x2, y2]


def bbox_to_0_1000(box: list[float]) -> list[float]:
    """Normalize common bbox coordinate scales to the 0-1000 eval scale."""
    max_coord = max(abs(value) for value in box[:4])
    if max_coord <= 1.5:
        scale = 1000.0
    elif max_coord <= 100.0:
        scale = 10.0
    elif max_coord <= 512.0:
        scale = 1000.0 / 504.0
    else:
        scale = 1.0
    return clamp_bbox([value * scale for value in box[:4]], 1000.0)


def reference_bbox_0_1000(sample: EvalSample) -> list[float] | None:
    value = sample.eval_meta.get("bbox_qwen_0_1000")
    if isinstance(value, list) and len(value) >= 4:
        return clamp_bbox([float(v) for v in value[:4]], 1000.0)

    if isinstance(sample.bboxes, list) and sample.bboxes:
        first = sample.bboxes[0]
        if isinstance(first, list) and len(first) >= 4:
            box = [float(v) for v in first[:4]]
            if sample.bbox_format == "xyxy_norm_0_100":
                box = [v * 10.0 for v in box]
            return clamp_bbox(box, 1000.0)
    ref_box = parse_bbox(sample.answer)
    return bbox_to_0_1000(ref_box) if ref_box else None


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    a = clamp_bbox(box_a)
    b = clamp_bbox(box_b)
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def fmt(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def compute_summary(
    model_name: str,
    data_name: str,
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    system_stats: dict[str, float | int | None],
) -> dict[str, str]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successes:
        by_task[row["task"]].append(row)

    caption_rows = by_task.get("caption", []) + by_task.get("change_caption", [])
    vqa_rows = by_task.get("vqa", [])
    count_rows = by_task.get("counting", [])
    ground_rows = by_task.get("grounding", [])

    caption_scores = official_caption_metrics(caption_rows)

    vqa_scores = {}
    for subtype in ["Category", "Presence", "Quantity", "Color", "Shape", "Size", "Position", "Direction", "Scene", "Reasoning", "Unknown"]:
        rows = [row for row in vqa_rows if row.get("vqa_subtype") == subtype]
        vqa_scores[f"VQA {subtype}"] = sum(bool(row.get("vqa_match")) for row in rows) / len(rows) if rows else None
    vqa_scores["VQA All"] = sum(bool(row.get("vqa_match")) for row in vqa_rows) / len(vqa_rows) if vqa_rows else None

    count_mae = count_acc = None
    count_errors = []
    for row in count_rows:
        pred_num = row.get("pred_number")
        ref_num = row.get("target_number")
        if pred_num is not None and ref_num is not None:
            count_errors.append(abs(pred_num - ref_num))
    if count_errors:
        count_mae = sum(count_errors) / len(count_errors)
        count_acc = sum(error <= 1.0 for error in count_errors) / len(count_errors)

    grounding_scores = {}
    for group in ("Unique", "Non-Unique", "All"):
        rows = ground_rows if group == "All" else [row for row in ground_rows if row.get("grounding_group") == group]
        ious = [row["iou"] for row in rows if row.get("iou") is not None]
        for threshold in (0.5, 0.7):
            grounding_scores[f"Ground {group}@{threshold}"] = sum(iou >= threshold for iou in ious) / len(ious) if ious else None

    return {
        "Model": model_name,
        "Data": data_name,
        "Success": str(len(successes)),
        "Failure": str(len(failures)),
        **{key: fmt(value) for key, value in caption_scores.items()},
        **{key: fmt(value) for key, value in vqa_scores.items()},
        "Count MAE": fmt(count_mae),
        "Count Acc±1": fmt(count_acc),
        **{key: fmt(value) for key, value in grounding_scores.items()},
        **{key: fmt(value) for key, value in system_stats.items()},
    }


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    apply_gpu_selection(args.gpu_id)

    annotation_file = args.annotation_file
    data_root = args.data_root or infer_data_root(annotation_file)
    run_name = args.run_name or datetime.now().strftime("qwen25vl_eval_%Y%m%d_%H%M%S")
    output_dir = args.output_dir.expanduser().resolve()
    summary_path = output_dir / f"{run_name}_summary.csv"
    details_path = output_dir / f"{run_name}_details.jsonl"
    failures_path = output_dir / f"{run_name}_failures.jsonl"
    manifest_path = output_dir / f"{run_name}_metric_manifest.json"
    selected_path = output_dir / f"{run_name}_selected_samples.jsonl"
    sampling_audit_path = output_dir / f"{run_name}_sampling_audit.json"

    all_records = read_records(annotation_file)
    sampling_audit: dict[str, Any] | None = None
    if args.standardized_random_sample:
        if args.sample_ids_file is not None:
            raise ValueError("--standardized-random-sample and --sample-ids-file are mutually exclusive")
        records, sampling_audit = standardized_random_sample(
            all_records, args.samples_per_task, args.samples_per_vqa_subtype, args.seed
        )
    else:
        records = select_records(all_records, args)
    samples = [normalize_sample(record) for record in records]
    if args.sample_ids_file is not None:
        print(f"Exact sample IDs: {args.sample_ids_file.expanduser().resolve()} ({len(samples)} IDs)")
    selected_counts = Counter(sample.task for sample in samples)
    print(
        "Selected samples by task: "
        + json.dumps(dict(sorted(selected_counts.items())), ensure_ascii=False)
    )
    selected_rows = []
    for order, sample in enumerate(samples):
        selected_rows.append({
            "order": order, "id": sample.id, "dataset": sample.dataset, "task": sample.task,
            "vqa_subtype": vqa_subtype(sample) if sample.task == "vqa" else None,
            "images": sample.images, "question": sample.question, "answer": sample.answer,
        })
    write_jsonl(selected_path, selected_rows)
    if sampling_audit is not None:
        sampling_audit_path.parent.mkdir(parents=True, exist_ok=True)
        sampling_audit_path.write_text(
            json.dumps(sampling_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Standardized sampling audit: " + json.dumps(sampling_audit["selected"], ensure_ascii=False))
    print(f"Recorded exact selected samples in {selected_path}")

    if args.dry_run:
        resolved = []
        for sample in samples:
            image_paths = [
                str(resolve_image_path(image, data_root=data_root, annotation_file=annotation_file))
                for image in sample.images
            ]
            resolved.append(
                {
                    "id": sample.id,
                    "dataset": sample.dataset,
                    "task": sample.task,
                    "images": sample.images,
                    "resolved_images": image_paths,
                    "question": sample.question,
                    "answer": sample.answer,
                    "references": sample.references,
                    "vqa_subtype": vqa_subtype(sample) if sample.task == "vqa" else None,
                    "grounding_group": grounding_group(sample) if sample.task == "grounding" else None,
                }
            )
        write_jsonl(details_path, resolved)
        print(f"Dry run OK: parsed {len(samples)} samples.")
        print(f"Wrote resolved sample details to {details_path}")
        return 0

    if any(sample.task in {"caption", "change_caption"} for sample in samples):
        try:
            import pycocoevalcap  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Standard caption metrics require pycocoevalcap. Run `pip install -r requirements.txt` before evaluation."
            ) from exc

    input_device = choose_input_device(args.device)
    rss_monitor = PeakRssMonitor()
    rss_monitor.start()
    model, processor = load_model_and_processor(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
    )

    import torch
    from tqdm import tqdm

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    profiles: list[dict[str, float | int]] = []

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            resolved_images = [
                resolve_image_path(image, data_root=data_root, annotation_file=annotation_file)
                for image in sample.images
            ]
            image_uris = [image_to_qwen_uri(image) for image in resolved_images]
            prediction, profile = generate_answer_profiled(
                model=model,
                processor=processor,
                image_uris=image_uris,
                question=sample.question,
                input_device=input_device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            latency = float(profile["e2e_seconds"])
            profiles.append(profile)

            row = {
                "id": sample.id,
                "dataset": sample.dataset,
                "task": sample.task,
                "question": sample.question,
                "answer": sample.answer,
                "references": sample.references,
                "prediction": prediction,
                "latency": latency,
                "profile": profile,
                "images": sample.images,
                "resolved_images": [str(path) for path in resolved_images],
            }

            if sample.task == "counting":
                row["pred_number"] = extract_number(prediction)
                row["target_number"] = target_number(sample)
            elif sample.task == "vqa":
                matches = [vqa_match(prediction, reference) for reference in sample.references]
                matched = any(value[0] for value in matches)
                match_type = next((value[1] for value in matches if value[0]), "mismatch")
                row["vqa_match"] = matched
                row["vqa_match_type"] = match_type
                row["vqa_subtype"] = vqa_subtype(sample)
            elif sample.task == "grounding":
                pred_box_raw = parse_bbox(prediction)
                pred_box = bbox_to_0_1000(pred_box_raw) if pred_box_raw else None
                ref_box = reference_bbox_0_1000(sample)
                row["pred_bbox_raw"] = pred_box_raw
                row["pred_bbox_0_1000"] = pred_box
                row["ref_bbox_0_1000"] = ref_box
                row["iou"] = bbox_iou(pred_box, ref_box) if pred_box and ref_box else None
                row["grounding_group"] = grounding_group(sample)

            successes.append(row)
        except Exception as exc:  # Keep evaluating after individual sample failures.
            failures.append(
                {
                    "id": sample.id,
                    "dataset": sample.dataset,
                    "task": sample.task,
                    "question": sample.question,
                    "answer": sample.answer,
                    "images": sample.images,
                    "error": repr(exc),
                }
            )

    peak_gpu_gib = None
    if torch.cuda.is_available():
        peak_gpu_gib = torch.cuda.max_memory_allocated() / (1024**3)
    peak_cpu_gib = rss_monitor.stop()
    e2e = [float(value["e2e_seconds"]) for value in profiles]
    ttft = [float(value["ttft_seconds"]) for value in profiles]
    rates = [float(value["decode_tokens_per_second"]) for value in profiles if float(value["decode_tokens_per_second"]) > 0]
    system_stats: dict[str, float | int | None] = {
        "Total Params": sum(parameter.numel() for parameter in model.parameters()),
        "LoRA Params": sum(parameter.numel() for name, parameter in model.named_parameters() if "lora_" in name.lower()),
        "Model size GiB": directory_size_gib([args.model_id, args.adapter_path]),
        "Peak GPU GiB": peak_gpu_gib,
        "Peak CPU GiB": peak_cpu_gib,
        "Mean E2E s": statistics.fmean(e2e) if e2e else None,
        "P95 E2E s": percentile95(e2e),
        "Mean TTFT s": statistics.fmean(ttft) if ttft else None,
        "P95 TTFT s": percentile95(ttft),
        "Decode tok/s": statistics.fmean(rates) if rates else None,
    }

    summaries = [compute_summary(
        model_name=args.model_id,
        data_name=str(annotation_file),
        successes=successes,
        failures=failures,
        system_stats=system_stats,
    )]
    dataset_names = sorted({str(row["dataset"]) for row in successes if row.get("dataset")})
    for dataset_name in dataset_names:
        summaries.append(compute_summary(
            model_name=args.model_id,
            data_name=dataset_name,
            successes=[row for row in successes if str(row.get("dataset")) == dataset_name],
            failures=[row for row in failures if str(row.get("dataset")) == dataset_name],
            system_stats=system_stats,
        ))

    write_summary_csv(summary_path, summaries)
    write_jsonl(details_path, successes)
    write_jsonl(failures_path, failures)
    manifest_path.write_text(json.dumps({
        "caption": {
            "implementation": "pycocoevalcap",
            "metrics": ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "Avg_L"],
            "scope": "caption and change_caption; all available references are retained",
            "CLAIR": "NA unless an external visual-language judge and prompt/version are explicitly configured",
        },
        "vqa": {
            "metric": "normalized exact-match accuracy (yes/no extracted for concise sentence answers)",
            "canonical_subtypes": list(VQA_TYPE_ALIASES),
            "aliases": {key: sorted(values) for key, values in VQA_TYPE_ALIASES.items()},
            "unmapped_label": "Unknown",
        },
        "grounding": {
            "metric": "IoU xyxy; Acc@0.5 and Acc@0.7",
            "groups": ["Unique", "Non-Unique", "All"],
            "unknown_uniqueness_is_included_only_in_all": True,
        },
        "efficiency": {
            "e2e_and_ttft_unit": "seconds", "decode_speed_unit": "generated tokenizer tokens/second",
            "gpu_memory": "torch CUDA max_memory_allocated GiB", "cpu_memory": "peak process RSS GiB",
            "model_size": "local base model plus adapter files on disk GiB; NA for remote model IDs",
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Wrote summary CSV to {summary_path}")
    print(f"Wrote details JSONL to {details_path}")
    print(f"Wrote failures JSONL to {failures_path}")
    print(f"Wrote metric manifest to {manifest_path}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
