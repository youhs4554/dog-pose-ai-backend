from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JsonScalar = str | int | float | bool | None


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable error message.")
    details: dict[str, Any] = Field(default_factory=dict, description="Optional structured metadata.")


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app_name: str
    version: str
    default_model_path: str
    default_model_exists: bool


class ModelInfoResponse(BaseModel):
    requested_model_path: str | None = None
    resolved_model_path: str
    exists: bool
    is_default: bool


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class KeypointPrediction(BaseModel):
    joint: str
    x: float
    y: float
    confidence: float | None = None
    visible: bool


class PoseDetection(BaseModel):
    instance: int
    confidence: float | None = None
    box: BoundingBox | None = None
    visible_keypoints: int
    keypoints: list[KeypointPrediction]


class ImagePredictionResponse(BaseModel):
    model_path: str
    image_width: int
    image_height: int
    confidence_threshold: float
    keypoint_threshold: float
    image_size: int
    detections: list[PoseDetection]
    overlay_image_base64: str | None = None
    overlay_media_type: str | None = None


class FrameRecord(BaseModel):
    instance: int
    joint: str
    x: float | None = None
    y: float | None = None
    confidence: float | None = None


class FrameAnalysisResponse(BaseModel):
    frame_index: int
    time_sec: float
    metrics: dict[str, JsonScalar]
    records: list[FrameRecord]
    overlay_image_base64: str | None = None
    overlay_media_type: str | None = None


class VideoAnalysisResponse(BaseModel):
    model_path: str
    source_fps: float
    analyzed_fps: float
    total_frames: int
    total_duration_sec: float
    frame_step: int
    analyzed_frames: list[FrameAnalysisResponse]
    trend: list[dict[str, JsonScalar]] = Field(default_factory=list)
    gait_summary: list[dict[str, JsonScalar]] = Field(default_factory=list)
    gait_stats: dict[str, JsonScalar] = Field(default_factory=dict)
    gait_note: str
    gait_status: str
    gait_reasons: list[str] = Field(default_factory=list)
    interpretation_note: str
    mpl_summary: list[dict[str, JsonScalar]] = Field(default_factory=list)
    mpl_stats: dict[str, JsonScalar] = Field(default_factory=dict)
    mpl_note: str
    mpl_status: str
    mpl_reasons: list[str] = Field(default_factory=list)
    mpl_primary_side: str
    mpl_evidence_markdown: str
    playback_video_base64: str | None = None
    playback_video_media_type: str | None = None
    playback_video_data_url: str | None = None
