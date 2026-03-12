from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends

from dog_pose_mvp.api.repository import ModelRepository
from dog_pose_mvp.api.service import InferenceService
from dog_pose_mvp.visualization import DEFAULT_DOG_POSE_MODEL_PATH


def get_default_model_path() -> str:
    """Use the same explicit default checkpoint path shown in the Streamlit demo."""
    return os.getenv("DOG_POSE_MODEL_PATH", str(DEFAULT_DOG_POSE_MODEL_PATH))


@lru_cache(maxsize=1)
def get_model_repository() -> ModelRepository:
    return ModelRepository(default_model_path=get_default_model_path())


def get_inference_service(
    repository: ModelRepository = Depends(get_model_repository),
) -> InferenceService:
    return InferenceService(repository)
