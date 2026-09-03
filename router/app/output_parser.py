from __future__ import annotations

import json
import re
from typing import Any

from .task_classifier import TaskDecision


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def has_multiple_choice(text: str) -> bool:
    return len(re.findall(r"(?:^|\s)[(（]?[A-E][)）.、:]", text, flags=re.IGNORECASE)) >= 2


def extract_choice(text: str) -> str | None:
    upper = text.strip().upper()
    patterns = [
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*[(\[]?([A-E])[)\]]?",
        r"^\s*[(\[]?([A-E])[)\].,:\s]",
        r"\b([A-E])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return None


def allows_multiple_choices(text: str) -> bool:
    return bool(
        re.search(
            r"more than one correct option|multiple choices?|answer\(s\)|letter\(s\)",
            text,
            flags=re.IGNORECASE,
        )
    )


def extract_choices(text: str) -> list[str]:
    upper = text.upper()
    matches = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", upper)
    result: list[str] = []
    for letter in matches:
        if letter not in result:
            result.append(letter)
    return result


def extract_number(text: str) -> int | float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if match:
        value = float(match.group(0))
        return int(value) if value.is_integer() else value
    lowered = text.lower()
    for token, value in NUMBER_WORDS.items():
        if re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", lowered):
            return value
    return None


def extract_bbox(text: str, image_meta: dict[str, Any] | None = None) -> list[int] | None:
    candidates: list[Any] = []
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            candidates.append(payload.get("bbox_2d") or payload.get("bbox"))
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text):
        candidates.append([float(match.group(index)) for index in range(1, 5)])
    for value in candidates:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            continue
        try:
            box = [float(item) for item in value]
        except (TypeError, ValueError):
            continue
        if max(abs(item) for item in box) <= 1.0:
            box = [item * 1000.0 for item in box]
        elif max(box) > 1000.0 and image_meta:
            width = max(1.0, float(image_meta["width"]))
            height = max(1.0, float(image_meta["height"]))
            box = [box[0] / width * 1000, box[1] / height * 1000, box[2] / width * 1000, box[3] / height * 1000]
        box = [int(round(max(0.0, min(1000.0, item)))) for item in box]
        if box[2] > box[0] and box[3] > box[1]:
            return box
    return None


def parse_output(raw: str, task: TaskDecision, prompt: str, image_meta: list[dict[str, Any]]) -> tuple[Any, bool, list[str]]:
    raw = str(raw).strip()
    warnings: list[str] = []
    if has_multiple_choice(prompt):
        if allows_multiple_choices(prompt):
            choices = extract_choices(raw)
            if choices:
                return " ".join(choices), True, warnings
            return raw, False, ["multiple_choice_parse_failed"]
        choice = extract_choice(raw)
        if choice:
            return choice, True, warnings
        return raw, False, ["multiple_choice_parse_failed"]
    route_task = task.subtask if task.type == "vqa" and task.subtask else task.type
    if route_task == "counting":
        number = extract_number(raw)
        if number is not None:
            return number, True, warnings
        return raw, False, ["count_parse_failed"]
    if route_task in {"grounding", "detection"}:
        bbox = extract_bbox(raw, image_meta[0] if image_meta else None)
        if bbox:
            return {"bbox_2d": bbox}, True, warnings
        return raw, False, ["bbox_parse_failed"]
    if not raw:
        return raw, False, ["empty_model_response"]
    return raw, True, warnings


def augment_prompt(text: str, task: TaskDecision) -> str:
    if has_multiple_choice(text):
        if re.search(
            r"(?:only\s+respond|respond\s+with\s+only|(?:respond|return|answer)\s+with)"
            r"[^\n]{0,80}\bletter(?:\(s\))?|"
            r"return\s+(?:the\s+)?final\s+answer\s+as\s+exactly\s+one\s+option",
            text,
            flags=re.IGNORECASE,
        ):
            return text
        return text + "\nReturn the final answer as exactly one option letter (A, B, C, D, or E)."
    route_task = task.subtask if task.type == "vqa" and task.subtask else task.type
    if route_task == "counting":
        return text + "\nAnswer with the numeric count clearly and concisely."
    if route_task in {"grounding", "detection"}:
        return text + '\nReturn one bounding box as JSON: {"bbox_2d":[x1,y1,x2,y2]}, normalized to 0-1000.'
    return text


def repair_prompt(text: str, raw: str, task: TaskDecision) -> str:
    route_task = task.subtask if task.type == "vqa" and task.subtask else task.type
    if has_multiple_choice(text):
        instruction = (
            "Return only the option letters separated by spaces, for example: A C."
            if allows_multiple_choices(text)
            else "Return only one option letter: A, B, C, D, or E."
        )
    elif route_task == "counting":
        instruction = "Return the numeric count."
    else:
        instruction = 'Return exactly one JSON object: {"bbox_2d":[x1,y1,x2,y2]} using 0-1000 coordinates.'
    return f"{text}\n\nYour previous answer was: {raw}\nReformat it without changing the conclusion. {instruction}"
