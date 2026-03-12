from __future__ import annotations

from typing import Any


class DogPoseApiError(Exception):
    """Base application error surfaced as a JSON API response."""

    status_code = 400
    code = "dog_pose_api_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class InvalidInputError(DogPoseApiError):
    status_code = 400
    code = "invalid_input"


class ModelNotFoundError(DogPoseApiError):
    status_code = 404
    code = "model_not_found"


class InferenceExecutionError(DogPoseApiError):
    status_code = 500
    code = "inference_execution_error"
