from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .task_classifier import TaskDecision


LOCAL_DETAIL_TASKS = {
    "counting",
    "color",
    "position",
    "grounding",
    "detection",
    "spatial_relationship",
}

SEMANTIC_TASKS = {
    "caption",
    "vqa",
    "scene_classification",
    "complex_reasoning",
    "land_use_classification",
    "object_classification",
    "motion_state",
}


@dataclass(frozen=True)
class RouteDecision:
    worker: str
    model: str
    reason: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def effective_route_task(task: TaskDecision) -> str:
    if task.type == "vqa" and task.subtask in LOCAL_DETAIL_TASKS:
        return str(task.subtask)
    return task.type


def choose_route(
    task: TaskDecision,
    image_meta: list[dict[str, Any]],
    low_resolution_max_edge: int = 1024,
) -> RouteDecision:
    if len(image_meta) == 2:
        return RouteDecision(
            worker="qwen",
            model="Qwen2.5-VL-3B-Instruct + Full LoRA",
            reason=["two_images", "phase1_change_route"],
        )
    max_edge = max(int(item["max_edge"]) for item in image_meta)
    if max_edge <= low_resolution_max_edge:
        return RouteDecision(
            worker="qwen",
            model="Qwen2.5-VL-3B-Instruct + Full LoRA",
            reason=["single_image", f"max_edge_le_{low_resolution_max_edge}"],
        )

    route_task = effective_route_task(task)
    if route_task in LOCAL_DETAIL_TASKS:
        return RouteDecision(
            worker="zoomsearch",
            model="LLaVA-OneVision-Qwen2-7B + VisRAG-Ret",
            reason=["single_image", f"max_edge_gt_{low_resolution_max_edge}", "local_detail_task", route_task],
        )
    return RouteDecision(
        worker="geollava",
        model="GeoLLaVA-8K",
        reason=["single_image", f"max_edge_gt_{low_resolution_max_edge}", "semantic_or_general_task", route_task],
    )
