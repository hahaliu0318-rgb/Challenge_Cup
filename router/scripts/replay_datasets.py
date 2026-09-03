from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay VRSBench, LEVIR-CC, XLRS, or MME-RS records through the router.")
    parser.add_argument("--dataset", required=True, choices=["vrs", "levir", "xlrs", "mme"])
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("annotation root must be a JSON list or JSONL")
    return payload


def infer_task(record: dict[str, Any]) -> str | None:
    explicit = record.get("task") or record.get("Category")
    if explicit:
        value = str(explicit).lower()
        return {"count": "counting", "position": "position", "color": "color"}.get(value, value)
    sample_id = str(record.get("id") or record.get("Question_id") or "").lower()
    for task in ("change_caption", "grounding", "counting", "caption", "vqa"):
        if task in sample_id:
            return task
    return None


def qwen_conversation_text(record: dict[str, Any]) -> str:
    for turn in record.get("conversations", []):
        if str(turn.get("from", "")).lower() in {"human", "user"}:
            return re.sub(r"<image>\s*", "", str(turn.get("value", ""))).strip()
    return str(record.get("question") or record.get("Text") or "").strip()


def resolve_images(record: dict[str, Any], annotation: Path, image_root: Path | None) -> list[str]:
    raw = record.get("images") or record.get("image") or record.get("Image")
    values = raw if isinstance(raw, list) else [raw]
    result: list[str] = []
    for value in values:
        path = Path(str(value))
        if not path.is_absolute():
            path = (image_root / path if image_root else annotation.parent / path).resolve()
        result.append(str(path))
    return result


def normalize(record: dict[str, Any], dataset: str, annotation: Path, image_root: Path | None) -> dict[str, Any]:
    images = resolve_images(record, annotation, image_root)
    text = qwen_conversation_text(record)
    if dataset == "mme":
        choices = record.get("Answer choices") or []
        text = text + "\n" + "\n".join(str(choice) for choice in choices)
    return {"image": images[0] if len(images) == 1 else images, "text": text, "task_type": infer_task(record)}


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def main() -> int:
    args = parse_args()
    annotation = Path(args.annotation).resolve()
    image_root = Path(args.image_root).resolve() if args.image_root else None
    endpoint = "/v1/routes/preview" if args.preview else "/v1/infer"
    for record in read_records(annotation)[: args.limit]:
        payload = normalize(record, args.dataset, annotation, image_root)
        response = post_json(args.base_url.rstrip("/") + endpoint, payload)
        print(json.dumps({"request": payload, "response": response}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
