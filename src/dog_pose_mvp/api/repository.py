from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from ultralytics import YOLO

from dog_pose_mvp.api.exceptions import InferenceExecutionError, ModelNotFoundError
from dog_pose_mvp.visualization import load_model, resolve_default_model_path


@dataclass(frozen=True)
class ModelDescriptor:
    requested_path: str | None
    resolved_path: Path
    exists: bool
    is_default: bool


@dataclass
class LoadedModel:
    descriptor: ModelDescriptor
    model: YOLO
    prediction_lock: Lock


class ModelRepository:
    """Loads and caches YOLO checkpoints for API requests."""

    def __init__(self, default_model_path: str | None = None) -> None:
        self._default_model_path = default_model_path
        self._models: dict[str, LoadedModel] = {}
        self._repository_lock = Lock()

    def describe_model(self, model_path: str | None = None) -> ModelDescriptor:
        raw_path = model_path or self._default_model_path or resolve_default_model_path()
        resolved_path = Path(raw_path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = Path.cwd() / resolved_path

        normalized_path = resolved_path.resolve()
        return ModelDescriptor(
            requested_path=model_path,
            resolved_path=normalized_path,
            exists=normalized_path.is_file(),
            is_default=model_path is None,
        )

    def get_loaded_model(self, model_path: str | None = None) -> LoadedModel:
        descriptor = self.describe_model(model_path)
        if not descriptor.exists:
            raise ModelNotFoundError(
                f"Model checkpoint was not found: {descriptor.resolved_path}",
                details={"resolved_model_path": str(descriptor.resolved_path)},
            )

        key = str(descriptor.resolved_path)
        with self._repository_lock:
            loaded_model = self._models.get(key)
            if loaded_model is not None:
                return loaded_model

            try:
                model = load_model(key)
            except Exception as exc:  # pragma: no cover - defensive wrapper
                raise InferenceExecutionError(
                    "Could not load the requested model checkpoint.",
                    details={"resolved_model_path": key, "reason": str(exc)},
                ) from exc

            loaded_model = LoadedModel(
                descriptor=descriptor,
                model=model,
                prediction_lock=Lock(),
            )
            self._models[key] = loaded_model
            return loaded_model
