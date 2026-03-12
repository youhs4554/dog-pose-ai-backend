from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient
from PIL import Image

from dog_pose_mvp.api.dependencies import get_default_model_path, get_inference_service
from dog_pose_mvp.api.exceptions import ModelNotFoundError
from dog_pose_mvp.api.main import create_app
from dog_pose_mvp.api.repository import LoadedModel, ModelDescriptor
from dog_pose_mvp.api.schemas import (
    FrameAnalysisResponse,
    HealthResponse,
    ImagePredictionResponse,
    KeypointPrediction,
    ModelInfoResponse,
    PoseDetection,
    VideoAnalysisResponse,
)
from dog_pose_mvp.api.service import InferenceService, _sanitize_json_value, build_video_preview_html
from dog_pose_mvp.skeleton import DOG_KEYPOINT_NAMES
from dog_pose_mvp.visualization import DEFAULT_DOG_POSE_MODEL_PATH


def _make_png_bytes(width: int = 64, height: int = 48) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = torch.tensor([[10.0, 12.0, 110.0, 140.0]], dtype=torch.float32)
        self.conf = torch.tensor([0.92], dtype=torch.float32)

    def __len__(self) -> int:
        return 1


class _FakeKeypoints:
    def __init__(self) -> None:
        xy = np.stack([[index * 5.0, index * 7.0] for index in range(len(DOG_KEYPOINT_NAMES))])
        conf = np.full(len(DOG_KEYPOINT_NAMES), 0.95, dtype=np.float32)
        self.xy = torch.tensor(np.expand_dims(xy, axis=0), dtype=torch.float32)
        self.conf = torch.tensor(np.expand_dims(conf, axis=0), dtype=torch.float32)


class _FakeResult:
    def __init__(self) -> None:
        self.boxes = _FakeBoxes()
        self.keypoints = _FakeKeypoints()


class _FakeRepository:
    def __init__(self) -> None:
        self.descriptor = ModelDescriptor(
            requested_path=None,
            resolved_path=Path("/tmp/dog-pose-best.pt"),
            exists=True,
            is_default=True,
        )
        self.loaded_model = LoadedModel(
            descriptor=self.descriptor,
            model=object(),  # type: ignore[arg-type]
            prediction_lock=Lock(),
        )

    def describe_model(self, model_path: str | None = None) -> ModelDescriptor:
        return self.descriptor

    def get_loaded_model(self, model_path: str | None = None) -> LoadedModel:
        return self.loaded_model


class InferenceServiceTest(unittest.TestCase):
    def test_default_model_path_uses_streamlit_default_checkpoint_path(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            default_model_path = get_default_model_path()

        self.assertEqual(default_model_path, str(DEFAULT_DOG_POSE_MODEL_PATH))

    def test_default_model_path_prefers_env_override(self) -> None:
        with patch.dict("os.environ", {"DOG_POSE_MODEL_PATH": "/tmp/manual-best.pt"}, clear=True):
            default_model_path = get_default_model_path()

        self.assertEqual(default_model_path, "/tmp/manual-best.pt")

    def test_predict_image_returns_structured_detection_payload(self) -> None:
        service = InferenceService(_FakeRepository())

        with (
            patch("dog_pose_mvp.api.service.run_image_prediction", return_value=[_FakeResult()]),
            patch(
                "dog_pose_mvp.api.service.draw_pose_result",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ),
        ):
            response = service.predict_image(
                image_bytes=_make_png_bytes(),
                filename="dog.png",
                model_path=None,
                conf_threshold=0.25,
                keypoint_threshold=0.35,
                image_size=640,
                include_overlay=True,
            )

        self.assertEqual(response.model_path, "/tmp/dog-pose-best.pt")
        self.assertEqual(len(response.detections), 1)
        self.assertEqual(response.detections[0].visible_keypoints, len(DOG_KEYPOINT_NAMES))
        self.assertEqual(response.detections[0].keypoints[0].joint, "front_left_paw")
        self.assertEqual(response.overlay_media_type, "image/jpeg")
        self.assertTrue(response.overlay_image_base64)

    def test_sanitize_json_value_replaces_nan_with_none(self) -> None:
        payload = {
            "score": np.float32(0.91),
            "missing": float("nan"),
            "table": pd.DataFrame([{"value": np.float64(1.5)}, {"value": float("nan")}]),
        }

        sanitized = _sanitize_json_value(payload)

        self.assertAlmostEqual(float(sanitized["score"]), 0.91, places=6)
        self.assertIsNone(sanitized["missing"])
        self.assertEqual(sanitized["table"][0]["value"], 1.5)
        self.assertIsNone(sanitized["table"][1]["value"])

    def test_analyze_video_adds_data_url_for_postman_visualization(self) -> None:
        service = InferenceService(_FakeRepository())
        fake_analysis = SimpleNamespace(
            source_fps=30.0,
            analyzed_fps=15.0,
            total_frames=60,
            total_duration_sec=2.0,
            frame_step=2,
            analyzed_frames=[],
            trend_df=pd.DataFrame([]),
            gait_summary=pd.DataFrame([]),
            gait_stats={},
            gait_note="stable gait",
            gait_status="정상 패턴에 가까움",
            gait_reasons=[],
            interpretation_note="screening only",
            mpl_summary=pd.DataFrame([]),
            mpl_stats={},
            mpl_note="low concern",
            mpl_status="정상 패턴에 가까움",
            mpl_reasons=[],
            mpl_primary_side="판정 어려움",
            mpl_evidence_markdown="reference",
            playback_video_bytes=b"video-preview",
        )

        with patch("dog_pose_mvp.api.service.run_video_analysis", return_value=fake_analysis):
            response = service.analyze_video(
                video_bytes=b"video-bytes",
                filename="walk.mp4",
                model_path=None,
                conf_threshold=0.25,
                keypoint_threshold=0.35,
                image_size=640,
                analysis_fps=10.0,
                max_frames=30,
                include_frame_previews=False,
                include_playback_video=True,
                include_trend_data=True,
            )

        self.assertEqual(response.playback_video_media_type, "video/mp4")
        self.assertTrue(response.playback_video_base64)
        self.assertEqual(
            response.playback_video_data_url,
            f"data:video/mp4;base64,{response.playback_video_base64}",
        )

    def test_build_video_preview_html_embeds_video_player(self) -> None:
        response = VideoAnalysisResponse(
            model_path="/tmp/dog-pose-best.pt",
            source_fps=30.0,
            analyzed_fps=15.0,
            total_frames=120,
            total_duration_sec=4.0,
            frame_step=2,
            analyzed_frames=[],
            gait_note="stable gait",
            gait_status="정상 패턴에 가까움",
            interpretation_note="screening only",
            mpl_summary=[],
            mpl_stats={},
            mpl_note="low concern",
            mpl_status="정상 패턴에 가까움",
            mpl_primary_side="판정 어려움",
            mpl_evidence_markdown="reference",
            playback_video_base64="dmlkZW8=",
            playback_video_media_type="video/mp4",
            playback_video_data_url="data:video/mp4;base64,dmlkZW8=",
        )

        html_document = build_video_preview_html(response)

        self.assertIn("<video", html_document)
        self.assertIn("data:video/mp4;base64,dmlkZW8=", html_document)


class _StubService:
    def __init__(self) -> None:
        self.last_image_request: dict[str, object] | None = None
        self.last_video_request: dict[str, object] | None = None
        self.raise_not_found = False

    def get_health(self) -> HealthResponse:
        return HealthResponse(
            app_name="dog-pose-fastapi",
            version="0.1.0",
            default_model_path="/tmp/dog-pose-best.pt",
            default_model_exists=True,
        )

    def get_model_info(self, model_path: str | None = None) -> ModelInfoResponse:
        return ModelInfoResponse(
            requested_model_path=model_path,
            resolved_model_path="/tmp/dog-pose-best.pt",
            exists=True,
            is_default=model_path is None,
        )

    def predict_image(self, **kwargs: object) -> ImagePredictionResponse:
        if self.raise_not_found:
            raise ModelNotFoundError("checkpoint missing")

        self.last_image_request = kwargs
        return ImagePredictionResponse(
            model_path="/tmp/dog-pose-best.pt",
            image_width=64,
            image_height=48,
            confidence_threshold=0.3,
            keypoint_threshold=0.4,
            image_size=640,
            detections=[
                PoseDetection(
                    instance=0,
                    confidence=0.92,
                    visible_keypoints=1,
                    keypoints=[
                        KeypointPrediction(
                            joint="nose",
                            x=24.0,
                            y=18.0,
                            confidence=0.95,
                            visible=True,
                        )
                    ],
                )
            ],
            overlay_image_base64="aGVsbG8=",
            overlay_media_type="image/jpeg",
        )

    def analyze_video(self, **kwargs: object) -> VideoAnalysisResponse:
        self.last_video_request = kwargs
        include_playback_video = bool(kwargs.get("include_playback_video"))
        playback_video_base64 = "dmlkZW8=" if include_playback_video else None
        return VideoAnalysisResponse(
            model_path="/tmp/dog-pose-best.pt",
            source_fps=30.0,
            analyzed_fps=15.0,
            total_frames=120,
            total_duration_sec=4.0,
            frame_step=2,
            analyzed_frames=[
                FrameAnalysisResponse(
                    frame_index=0,
                    time_sec=0.0,
                    metrics={"detected": 1, "body_axis_angle_deg": 4.5},
                    records=[],
                )
            ],
            trend=[{"time_sec": 0.0, "body_axis_angle_deg": 4.5}],
            gait_summary=[{"Metric": "Cadence", "Value": "2.1 Hz"}],
            gait_stats={"rear_amplitude_balance": 0.85},
            gait_note="stable gait",
            gait_status="정상 패턴에 가까움",
            gait_reasons=["symmetry maintained"],
            interpretation_note="screening only",
            mpl_summary=[{"Metric": "Offloading", "Value": "Low"}],
            mpl_stats={"rear_offloading_ratio": 0.1},
            mpl_note="low concern",
            mpl_status="정상 패턴에 가까움",
            mpl_reasons=[],
            mpl_primary_side="판정 어려움",
            mpl_evidence_markdown="reference",
            playback_video_base64=playback_video_base64,
            playback_video_media_type="video/mp4" if include_playback_video else None,
            playback_video_data_url=(
                f"data:video/mp4;base64,{playback_video_base64}"
                if playback_video_base64
                else None
            ),
        )


class ApiRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _StubService()
        self.app = create_app()
        self.app.dependency_overrides[get_inference_service] = lambda: self.service
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        self.client.close()

    def test_health_endpoint_returns_status(self) -> None:
        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_image_endpoint_accepts_multipart_form(self) -> None:
        response = self.client.post(
            "/api/v1/inference/image",
            files={"file": ("dog.png", _make_png_bytes(), "image/png")},
            data={
                "conf_threshold": "0.3",
                "keypoint_threshold": "0.4",
                "image_size": "640",
                "include_overlay": "true",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["detections"][0]["keypoints"][0]["joint"], "nose")
        self.assertIsNotNone(self.service.last_image_request)
        assert self.service.last_image_request is not None
        self.assertEqual(self.service.last_image_request["filename"], "dog.png")
        self.assertEqual(self.service.last_image_request["conf_threshold"], 0.3)
        self.assertTrue(self.service.last_image_request["include_overlay"])

    def test_video_endpoint_accepts_multipart_form(self) -> None:
        response = self.client.post(
            "/api/v1/inference/video",
            files={"file": ("walk.mp4", b"fake-video-bytes", "video/mp4")},
            data={
                "analysis_fps": "10",
                "max_frames": "30",
                "include_frame_previews": "false",
                "include_playback_video": "false",
                "include_trend_data": "true",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["gait_status"], "정상 패턴에 가까움")
        self.assertIsNotNone(self.service.last_video_request)
        assert self.service.last_video_request is not None
        self.assertEqual(self.service.last_video_request["filename"], "walk.mp4")
        self.assertEqual(self.service.last_video_request["analysis_fps"], 10.0)
        self.assertEqual(self.service.last_video_request["max_frames"], 30)

    def test_video_endpoint_can_return_html_preview_for_postman(self) -> None:
        response = self.client.post(
            "/api/v1/inference/video",
            files={"file": ("walk.mp4", b"fake-video-bytes", "video/mp4")},
            data={
                "response_format": "html",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("<video", response.text)
        self.assertIsNotNone(self.service.last_video_request)
        assert self.service.last_video_request is not None
        self.assertTrue(self.service.last_video_request["include_playback_video"])

    def test_domain_error_is_translated_to_consistent_json(self) -> None:
        self.service.raise_not_found = True

        response = self.client.post(
            "/api/v1/inference/image",
            files={"file": ("dog.png", _make_png_bytes(), "image/png")},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "model_not_found")


if __name__ == "__main__":
    unittest.main()
