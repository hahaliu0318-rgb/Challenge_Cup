from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


CANONICAL_TASKS = {
    "caption",
    "vqa",
    "counting",
    "grounding",
    "detection",
    "scene_classification",
    "change_caption",
    "change_detection",
    "color",
    "position",
    "spatial_relationship",
    "complex_reasoning",
    "land_use_classification",
    "object_classification",
    "motion_state",
}

ALIASES = {
    "describe": "caption",
    "image_caption": "caption",
    "qa": "vqa",
    "question_answering": "vqa",
    "count": "counting",
    "quantity": "counting",
    "referring": "grounding",
    "bbox": "grounding",
    "scene": "scene_classification",
    "classification": "scene_classification",
    "change": "change_caption",
    "change_description": "change_caption",
    "spatial": "spatial_relationship",
    "reasoning": "complex_reasoning",
    "land_use": "land_use_classification",
    "object": "object_classification",
    "motion": "motion_state",
}

CONTROLLED_SUBTASKS = {
    "counting",
    "color",
    "position",
    "grounding",
    "spatial_relationship",
    "temporal_change",
    "scene",
    "land_use",
    "object",
    "motion",
    "reasoning",
    "detection",
}

SUBTASK_ALIASES = {
    "count": "counting",
    "quantity": "counting",
    "spatial": "spatial_relationship",
    "change": "temporal_change",
    "scene_classification": "scene",
    "land_use_classification": "land_use",
    "object_classification": "object",
    "motion_state": "motion",
    "complex_reasoning": "reasoning",
}

EXPLICIT_SUBTASKS = {
    "counting": "counting",
    "color": "color",
    "position": "position",
    "grounding": "grounding",
    "detection": "detection",
    "spatial_relationship": "spatial_relationship",
    "change_caption": "temporal_change",
    "change_detection": "temporal_change",
    "scene_classification": "scene",
    "land_use_classification": "land_use",
    "object_classification": "object",
    "motion_state": "motion",
    "complex_reasoning": "reasoning",
}


@dataclass(frozen=True)
class TaskDecision:
    type: str
    subtask: str | None
    source: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_task_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"", "auto", "none", "null"}:
        return None
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in CANONICAL_TASKS:
        raise ValueError(f"unsupported task_type: {value}")
    return normalized


RULES: list[tuple[str, str | None, float, tuple[str, ...]]] = [
    ("change_caption", "temporal_change", 0.99, (r"\b(compare|change[sd]?|difference|before|after)\b", r"变化|改变|前后|时相|对比")),
    ("grounding", "referring", 0.99, (r"\b(locate|bounding\s*box|bbox|coordinates?|draw a box)\b", r"定位|框出|边界框|坐标|包围盒")),
    ("counting", "quantity", 0.99, (r"\b(how many|count(?:ing)?|number of)\b", r"多少|计数|数量|几个|数一数")),
    ("color", "color", 0.98, (r"\b(what|which)\s+colou?r\b|\bcolou?r\s+of\b", r"什么颜色|哪种颜色|颜色")),
    ("caption", None, 0.97, (r"\b(describe|caption|summari[sz]e the image)\b", r"描述.*图|图像描述|概述.*图")),
    ("position", "position", 0.95, (r"\b(where is|where are|upper[- ]?left|upper[- ]?right|lower[- ]?left|lower[- ]?right|which area|position)\b", r"在哪里|位于|位置|方位|左上|右上|左下|右下")),
    ("spatial_relationship", "spatial", 0.94, (r"\b(relative to|relationship between|north of|south of|east of|west of)\b", r"相对位置|空间关系|以北|以南|以东|以西")),
    ("land_use_classification", "land_use", 0.93, (r"\b(land use|land cover)\b", r"土地利用|土地覆盖|地类")),
    ("scene_classification", "scene", 0.91, (r"\b(scene|category|classify|what type of area)\b", r"场景类别|场景类型|分类|什么类型区域")),
    ("motion_state", "motion", 0.90, (r"\b(moving|stationary|motion state)\b", r"运动状态|移动中|静止")),
    ("complex_reasoning", "reasoning", 0.90, (r"\b(why|reason|route planning|anomal(?:y|ous)|environmental condition|infer)\b", r"为什么|推理|原因|路径规划|异常|环境条件")),
    ("detection", "detection", 0.90, (r"\b(detect all|find all objects)\b", r"检测所有|找出所有目标")),
]


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def classify_by_rules(text: str, image_count: int, explicit_task: str | None = None) -> TaskDecision:
    explicit = normalize_task_type(explicit_task)
    detected: TaskDecision | None = None
    if image_count == 2 and _matches(text, RULES[0][3]):
        detected = TaskDecision("change_caption", "temporal_change", "image_and_text_rule", 0.99)
    else:
        for task, subtask, confidence, patterns in RULES:
            if _matches(text, patterns):
                detected = TaskDecision(task, subtask, "text_rule", confidence)
                break

    if explicit:
        subtask = EXPLICIT_SUBTASKS.get(explicit)
        if explicit == "vqa" and detected and detected.type in {"counting", "color", "position", "grounding", "spatial_relationship"}:
            subtask = detected.type
        return TaskDecision(explicit, subtask, "explicit", 1.0)
    if image_count == 2:
        return detected or TaskDecision("change_caption", "temporal_change", "image_count_rule", 0.95)
    return detected or TaskDecision("vqa", None, "rule_fallback", 0.40)


def parse_model_classification(raw: str) -> TaskDecision | None:
    text = raw.strip()
    candidates = [text]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        try:
            task = normalize_task_type(str(payload.get("task") or payload.get("type") or ""))
        except ValueError:
            continue
        if not task:
            continue
        raw_subtask = payload.get("subtask")
        subtask = None
        if raw_subtask:
            normalized_subtask = str(raw_subtask).strip().lower().replace("-", "_").replace(" ", "_")
            normalized_subtask = SUBTASK_ALIASES.get(normalized_subtask, normalized_subtask)
            if normalized_subtask in CONTROLLED_SUBTASKS:
                subtask = normalized_subtask
        if task in {"counting", "color", "position", "grounding", "spatial_relationship", "detection"}:
            subtask = task
        elif task == "change_caption":
            subtask = "temporal_change"
        confidence = payload.get("confidence", 0.80)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.80
        return TaskDecision(task, subtask, "qwen_classifier", confidence)
    return None
