from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str | list[str]
    text: str = Field(min_length=1, max_length=4096)
    task_type: str | None = Field(default=None, max_length=64)

    @field_validator("image")
    @classmethod
    def validate_image_count(cls, value: str | list[str]) -> str | list[str]:
        values = [value] if isinstance(value, str) else value
        if not 1 <= len(values) <= 2:
            raise ValueError("image must contain one or two server paths")
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError("every image path must be a non-empty string")
        if any(len(item) > 4096 for item in values):
            raise ValueError("image path must not exceed 4096 characters")
        return value

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value

    @field_validator("task_type")
    @classmethod
    def normalize_task_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower().replace("-", "_").replace(" ", "_")
        return None if value in {"", "auto", "none", "null"} else value

    def image_paths(self) -> list[str]:
        return [self.image] if isinstance(self.image, str) else list(self.image)


class JobAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str


class CompactAnswer(BaseModel):
    answer: Any
