from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError


class ImageValidationError(ValueError):
    pass


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_and_probe_images(
    raw_paths: Iterable[str],
    allowed_roots: Iterable[str | Path],
    max_file_size_bytes: int = 1_500_000_000,
    max_image_pixels: int = 500_000_000,
) -> tuple[list[str], list[dict[str, Any]]]:
    roots = [Path(root).expanduser().resolve(strict=True) for root in allowed_roots]
    resolved_paths: list[str] = []
    metadata: list[dict[str, Any]] = []

    Image.MAX_IMAGE_PIXELS = None
    for raw in raw_paths:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            raise ImageValidationError(f"image path must be absolute: {raw}")
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ImageValidationError(f"image does not exist: {raw}") from exc
        if not any(_within(resolved, root) for root in roots):
            raise ImageValidationError(f"image path is outside allowed roots: {resolved}")
        if not resolved.is_file():
            raise ImageValidationError(f"image path is not a file: {resolved}")
        if not os.access(resolved, os.R_OK):
            raise ImageValidationError(f"image is not readable: {resolved}")
        size_bytes = resolved.stat().st_size
        if size_bytes <= 0 or size_bytes > max_file_size_bytes:
            raise ImageValidationError(f"invalid image file size: {resolved} ({size_bytes} bytes)")

        try:
            with Image.open(resolved) as image:
                width, height = image.size
                image_format = image.format
                mode = image.mode
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageValidationError(f"invalid or corrupt image: {resolved}") from exc
        if width <= 0 or height <= 0:
            raise ImageValidationError(f"invalid image dimensions: {resolved}")
        if width * height > max_image_pixels:
            raise ImageValidationError(
                f"image exceeds pixel limit: {resolved} ({width * height} > {max_image_pixels})"
            )
        try:
            with Image.open(resolved) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageValidationError(f"invalid or corrupt image: {resolved}") from exc

        resolved_paths.append(str(resolved))
        metadata.append(
            {
                "path": str(resolved),
                "width": int(width),
                "height": int(height),
                "max_edge": int(max(width, height)),
                "pixels": int(width * height),
                "format": image_format,
                "mode": mode,
                "size_bytes": int(size_bytes),
            }
        )
    return resolved_paths, metadata
