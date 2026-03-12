from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from dog_pose_mvp.api.dependencies import get_inference_service
from dog_pose_mvp.api.schemas import HealthResponse, ImagePredictionResponse, ModelInfoResponse, VideoAnalysisResponse
from dog_pose_mvp.api.service import InferenceService, build_video_preview_html

router = APIRouter(prefix="/api/v1", tags=["dog-pose"])
ServiceDep = Annotated[InferenceService, Depends(get_inference_service)]


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(service: ServiceDep) -> HealthResponse:
    return await run_in_threadpool(service.get_health)


@router.get("/model", response_model=ModelInfoResponse, summary="Resolve the active checkpoint")
async def get_model_info(
    service: ServiceDep,
    model_path: str | None = None,
) -> ModelInfoResponse:
    return await run_in_threadpool(service.get_model_info, model_path)


@router.post(
    "/inference/image",
    response_model=ImagePredictionResponse,
    summary="Run dog pose inference on a single image",
)
async def predict_image(
    service: ServiceDep,
    file: UploadFile = File(description="Input dog image."),
    model_path: Annotated[str | None, Form(description="Optional checkpoint path override.")] = None,
    conf_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
    keypoint_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.35,
    image_size: Annotated[int, Form(ge=64, le=2048)] = 640,
    include_overlay: Annotated[bool, Form()] = True,
) -> ImagePredictionResponse:
    image_bytes = await file.read()
    return await run_in_threadpool(
        service.predict_image,
        image_bytes=image_bytes,
        filename=file.filename,
        model_path=model_path,
        conf_threshold=conf_threshold,
        keypoint_threshold=keypoint_threshold,
        image_size=image_size,
        include_overlay=include_overlay,
    )


@router.post(
    "/inference/video",
    response_model=VideoAnalysisResponse,
    summary="Run gait analysis on an uploaded video",
)
async def predict_video(
    service: ServiceDep,
    file: UploadFile = File(description="Input gait video."),
    model_path: Annotated[str | None, Form(description="Optional checkpoint path override.")] = None,
    conf_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
    keypoint_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.35,
    image_size: Annotated[int, Form(ge=64, le=2048)] = 640,
    analysis_fps: Annotated[float, Form(ge=0.0, le=240.0)] = 0.0,
    max_frames: Annotated[int, Form(ge=0, le=10000)] = 0,
    include_frame_previews: Annotated[bool, Form()] = False,
    include_playback_video: Annotated[bool, Form()] = False,
    include_trend_data: Annotated[bool, Form()] = True,
    response_format: Annotated[Literal["json", "html"], Form()] = "json",
) -> VideoAnalysisResponse | HTMLResponse:
    video_bytes = await file.read()
    analysis_response = await run_in_threadpool(
        service.analyze_video,
        video_bytes=video_bytes,
        filename=file.filename,
        model_path=model_path,
        conf_threshold=conf_threshold,
        keypoint_threshold=keypoint_threshold,
        image_size=image_size,
        analysis_fps=analysis_fps,
        max_frames=max_frames,
        include_frame_previews=include_frame_previews,
        include_playback_video=include_playback_video or response_format == "html",
        include_trend_data=include_trend_data,
    )
    if response_format == "html":
        return HTMLResponse(content=build_video_preview_html(analysis_response))
    return analysis_response
