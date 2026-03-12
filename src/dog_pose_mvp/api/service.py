from __future__ import annotations

import base64
import html
import math
import tempfile
from importlib import metadata
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from dog_pose_mvp.api.exceptions import InferenceExecutionError, InvalidInputError, ModelNotFoundError
from dog_pose_mvp.api.repository import ModelRepository
from dog_pose_mvp.api.schemas import (
    BoundingBox,
    FrameAnalysisResponse,
    FrameRecord,
    HealthResponse,
    ImagePredictionResponse,
    KeypointPrediction,
    ModelInfoResponse,
    PoseDetection,
    VideoAnalysisResponse,
)
from dog_pose_mvp.gait import analyze_video as run_video_analysis
from dog_pose_mvp.skeleton import DOG_KEYPOINT_NAMES
from dog_pose_mvp.visualization import (
    draw_pose_result,
    load_image,
    predict_image as run_image_prediction,
    validate_dog_keypoints,
)


def _project_version() -> str:
    try:
        return metadata.version("dog-pose-mvp")
    except metadata.PackageNotFoundError:  # pragma: no cover - local editable fallback
        return "0.1.0"


def _encode_bytes_base64(raw_bytes: bytes | None) -> str | None:
    if not raw_bytes:
        return None
    return base64.b64encode(raw_bytes).decode("ascii")


def _build_data_url(base64_payload: str | None, media_type: str | None) -> str | None:
    if not base64_payload or not media_type:
        return None
    return f"data:{media_type};base64,{base64_payload}"


def _encode_image_base64(image_bgr: np.ndarray) -> str:
    success, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not success:
        raise InferenceExecutionError("Could not encode overlay preview image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _sanitize_json_value(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _sanitize_json_value(value.to_dict())
    if isinstance(value, np.ndarray):
        return _sanitize_json_value(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _build_pose_detections(result: Any, keypoint_threshold: float) -> list[PoseDetection]:
    validate_dog_keypoints(result)
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return []

    xy = result.keypoints.xy.cpu().numpy()
    kp_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
    box_xyxy = result.boxes.xyxy.cpu().numpy() if getattr(result.boxes, "xyxy", None) is not None else None
    box_conf = result.boxes.conf.cpu().numpy() if getattr(result.boxes, "conf", None) is not None else None

    detections: list[PoseDetection] = []
    for det_index, points in enumerate(xy):
        keypoints: list[KeypointPrediction] = []
        visible_keypoints = 0
        for point_index, point in enumerate(points):
            confidence = float(kp_conf[det_index][point_index]) if kp_conf is not None else None
            is_visible = confidence is None or confidence >= keypoint_threshold
            if is_visible:
                visible_keypoints += 1
            keypoints.append(
                KeypointPrediction(
                    joint=DOG_KEYPOINT_NAMES[point_index],
                    x=float(point[0]),
                    y=float(point[1]),
                    confidence=confidence,
                    visible=is_visible,
                )
            )

        box = None
        if box_xyxy is not None:
            x1, y1, x2, y2 = box_xyxy[det_index]
            box = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))

        detections.append(
            PoseDetection(
                instance=det_index,
                confidence=float(box_conf[det_index]) if box_conf is not None else None,
                box=box,
                visible_keypoints=visible_keypoints,
                keypoints=keypoints,
            )
        )

    return detections


def build_video_preview_html(response: VideoAnalysisResponse) -> str:
    video_markup = "<p>No playback video is available for this analysis.</p>"
    if response.playback_video_data_url and response.playback_video_media_type:
        escaped_data_url = html.escape(response.playback_video_data_url, quote=True)
        escaped_media_type = html.escape(response.playback_video_media_type, quote=True)
        video_markup = (
            f'<video controls preload="metadata" style="width:100%;max-width:960px;border-radius:12px">'
            f'<source src="{escaped_data_url}" type="{escaped_media_type}">'
            "Your client does not support inline video playback."
            "</video>"
        )

    status = html.escape(response.gait_status)
    note = html.escape(response.gait_note)
    mpl_status = html.escape(response.mpl_status)
    model_path = html.escape(response.model_path)
    analyzed_frames = len(response.analyzed_frames)

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Dog Pose Video Preview</title>
    <style>
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 24px;
        color: #0f172a;
        background: #f8fafc;
      }}
      .card {{
        max-width: 1040px;
        background: white;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
      }}
      .meta {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
      }}
      .meta div {{
        background: #f1f5f9;
        border-radius: 10px;
        padding: 12px;
      }}
      code {{
        word-break: break-all;
      }}
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Dog Pose Playback Preview</h1>
      <div class="meta">
        <div><strong>Gait status</strong><br>{status}</div>
        <div><strong>MPL status</strong><br>{mpl_status}</div>
        <div><strong>Analyzed frames</strong><br>{analyzed_frames}</div>
        <div><strong>Analyzed FPS</strong><br>{response.analyzed_fps:.2f}</div>
      </div>
      <p><strong>Note:</strong> {note}</p>
      <p><strong>Model:</strong> <code>{model_path}</code></p>
      {video_markup}
    </div>
  </body>
</html>
"""


class InferenceService:
    def __init__(self, repository: ModelRepository) -> None:
        self._repository = repository

    def get_health(self) -> HealthResponse:
        descriptor = self._repository.describe_model()
        return HealthResponse(
            app_name="dog-pose-fastapi",
            version=_project_version(),
            default_model_path=str(descriptor.resolved_path),
            default_model_exists=descriptor.exists,
        )

    def get_model_info(self, model_path: str | None = None) -> ModelInfoResponse:
        descriptor = self._repository.describe_model(model_path)
        return ModelInfoResponse(
            requested_model_path=descriptor.requested_path,
            resolved_model_path=str(descriptor.resolved_path),
            exists=descriptor.exists,
            is_default=descriptor.is_default,
        )

    def predict_image(
        self,
        *,
        image_bytes: bytes,
        filename: str | None,
        model_path: str | None,
        conf_threshold: float,
        keypoint_threshold: float,
        image_size: int,
        include_overlay: bool,
    ) -> ImagePredictionResponse:
        if not image_bytes:
            raise InvalidInputError(
                "Uploaded image is empty.",
                details={"filename": filename or ""},
            )

        try:
            image = Image.open(BytesIO(image_bytes))
            image.load()
        except UnidentifiedImageError as exc:
            raise InvalidInputError(
                "Uploaded file is not a supported image.",
                details={"filename": filename or ""},
            ) from exc

        image_bgr = load_image(image)
        loaded_model = self._repository.get_loaded_model(model_path)

        try:
            with loaded_model.prediction_lock:
                results = run_image_prediction(
                    loaded_model.model,
                    image_bgr,
                    conf=conf_threshold,
                    imgsz=image_size,
                )
            result = results[0]
            detections = _build_pose_detections(result, keypoint_threshold=keypoint_threshold)
            overlay_image_base64 = None
            overlay_media_type = None
            if include_overlay:
                rendered_bgr = draw_pose_result(
                    image_bgr,
                    result,
                    keypoint_conf=keypoint_threshold,
                )
                overlay_image_base64 = _encode_image_base64(rendered_bgr)
                overlay_media_type = "image/jpeg"
        except ModelNotFoundError:
            raise
        except Exception as exc:
            raise InferenceExecutionError(
                "Image inference failed.",
                details={"filename": filename or "", "reason": str(exc)},
            ) from exc

        image_height, image_width = image_bgr.shape[:2]
        return ImagePredictionResponse(
            model_path=str(loaded_model.descriptor.resolved_path),
            image_width=image_width,
            image_height=image_height,
            confidence_threshold=conf_threshold,
            keypoint_threshold=keypoint_threshold,
            image_size=image_size,
            detections=detections,
            overlay_image_base64=overlay_image_base64,
            overlay_media_type=overlay_media_type,
        )

    def analyze_video(
        self,
        *,
        video_bytes: bytes,
        filename: str | None,
        model_path: str | None,
        conf_threshold: float,
        keypoint_threshold: float,
        image_size: int,
        analysis_fps: float,
        max_frames: int,
        include_frame_previews: bool,
        include_playback_video: bool,
        include_trend_data: bool,
    ) -> VideoAnalysisResponse:
        if not video_bytes:
            raise InvalidInputError(
                "Uploaded video is empty.",
                details={"filename": filename or ""},
            )

        loaded_model = self._repository.get_loaded_model(model_path)
        suffix = Path(filename or "upload.mp4").suffix or ".mp4"
        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(video_bytes)
                temp_path = Path(temp_file.name)

            with loaded_model.prediction_lock:
                analysis = run_video_analysis(
                    video_path=temp_path,
                    model=loaded_model.model,
                    conf_threshold=conf_threshold,
                    keypoint_threshold=keypoint_threshold,
                    image_size=image_size,
                    analysis_fps=analysis_fps,
                    max_frames=max_frames,
                )
        except ModelNotFoundError:
            raise
        except Exception as exc:
            raise InferenceExecutionError(
                "Video analysis failed.",
                details={"filename": filename or "", "reason": str(exc)},
            ) from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        analyzed_frames = [
            FrameAnalysisResponse(
                frame_index=frame.frame_index,
                time_sec=float(frame.time_sec),
                metrics=_sanitize_json_value(frame.metrics),
                records=[
                    FrameRecord(**_sanitize_json_value(record))
                    for record in frame.records
                ],
                overlay_image_base64=_encode_bytes_base64(frame.overlay_jpeg) if include_frame_previews else None,
                overlay_media_type="image/jpeg" if include_frame_previews else None,
            )
            for frame in analysis.analyzed_frames
        ]

        playback_video_base64 = (
            _encode_bytes_base64(analysis.playback_video_bytes)
            if include_playback_video
            else None
        )
        playback_video_media_type = "video/mp4" if include_playback_video and analysis.playback_video_bytes else None
        playback_video_data_url = _build_data_url(playback_video_base64, playback_video_media_type)

        return VideoAnalysisResponse(
            model_path=str(loaded_model.descriptor.resolved_path),
            source_fps=float(analysis.source_fps),
            analyzed_fps=float(analysis.analyzed_fps),
            total_frames=int(analysis.total_frames),
            total_duration_sec=float(analysis.total_duration_sec),
            frame_step=int(analysis.frame_step),
            analyzed_frames=analyzed_frames,
            trend=_sanitize_json_value(analysis.trend_df) if include_trend_data else [],
            gait_summary=_sanitize_json_value(analysis.gait_summary),
            gait_stats=_sanitize_json_value(analysis.gait_stats),
            gait_note=analysis.gait_note,
            gait_status=analysis.gait_status,
            gait_reasons=_sanitize_json_value(analysis.gait_reasons),
            interpretation_note=analysis.interpretation_note,
            mpl_summary=_sanitize_json_value(analysis.mpl_summary),
            mpl_stats=_sanitize_json_value(analysis.mpl_stats),
            mpl_note=analysis.mpl_note,
            mpl_status=analysis.mpl_status,
            mpl_reasons=_sanitize_json_value(analysis.mpl_reasons),
            mpl_primary_side=analysis.mpl_primary_side,
            mpl_evidence_markdown=analysis.mpl_evidence_markdown,
            playback_video_base64=playback_video_base64,
            playback_video_media_type=playback_video_media_type,
            playback_video_data_url=playback_video_data_url,
        )
