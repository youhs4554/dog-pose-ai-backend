from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from dog_pose_mvp.skeleton import (
    DOG_KEYPOINT_NAMES,
    DOG_SKELETON_EDGES,
    LEFT_KEYPOINTS,
    MIDLINE_KEYPOINTS,
    PALETTE,
    RIGHT_KEYPOINTS,
)


class PreviewBoxes:
    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor) -> None:
        self.xyxy = xyxy
        self.conf = conf

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class PreviewKeypoints:
    def __init__(self, xy: torch.Tensor, conf: torch.Tensor) -> None:
        self.xy = xy
        self.conf = conf


class PreviewResult:
    def __init__(self, boxes: PreviewBoxes, keypoints: PreviewKeypoints) -> None:
        self.boxes = boxes
        self.keypoints = keypoints


def validate_dog_keypoints(result: Any) -> None:
    if result.keypoints is None:
        return

    xy = result.keypoints.xy
    if xy is None or xy.ndim != 3 or xy.shape[1] == 0:
        return

    predicted_keypoints = int(xy.shape[1])
    expected_keypoints = len(DOG_KEYPOINT_NAMES)
    if predicted_keypoints != expected_keypoints:
        raise ValueError(
            f"Loaded checkpoint predicts {predicted_keypoints} keypoints, but dog-pose rendering requires "
            f"{expected_keypoints}. Use a dog-pose checkpoint produced by this project after training finishes."
        )


def resolve_default_model_path() -> str:
    latest_run = Path("runs/pose/latest_run.json")
    if latest_run.exists():
        payload = json.loads(latest_run.read_text(encoding="utf-8"))
        for key in ("best", "last"):
            candidate = Path(payload[key])
            if candidate.exists():
                return str(candidate)

    candidates = [
        Path("runs/pose/dog-pose-mps-1epoch/weights/best.pt"),
        Path("runs/pose/dog-pose-mps-1epoch/weights/last.pt"),
        Path("yolo26n-pose.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def load_image(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def resolve_label_path(image_path: str | Path) -> Path:
    image_path = Path(image_path)
    parts = list(image_path.parts)
    if "images" not in parts:
        raise FileNotFoundError(f"Could not infer label path from image path: {image_path}")

    image_index = parts.index("images")
    parts[image_index] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file does not exist for sample image: {label_path}")
    return label_path


def build_preview_result(image_path: str | Path, image_size: tuple[int, int]) -> PreviewResult:
    label_values = [
        float(value)
        for value in resolve_label_path(image_path).read_text(encoding="utf-8").strip().split()
    ]
    _, center_x, center_y, box_width, box_height, *raw_keypoints = label_values
    expected_length = len(DOG_KEYPOINT_NAMES) * 3
    if len(raw_keypoints) != expected_length:
        raise ValueError(
            f"Expected {expected_length} pose values in label, got {len(raw_keypoints)}. "
            "Check that the sample comes from the Ultralytics dog-pose dataset."
        )

    width, height = image_size
    points: list[list[float]] = []
    confidences: list[float] = []
    for index in range(len(DOG_KEYPOINT_NAMES)):
        x, y, visible = raw_keypoints[index * 3 : (index + 1) * 3]
        points.append([x * width, y * height])
        confidences.append(1.0 if visible > 0 else 0.0)

    x1 = (center_x - box_width / 2) * width
    x2 = (center_x + box_width / 2) * width
    y1 = (center_y - box_height / 2) * height
    y2 = (center_y + box_height / 2) * height

    return PreviewResult(
        boxes=PreviewBoxes(
            xyxy=torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
            conf=torch.tensor([1.0], dtype=torch.float32),
        ),
        keypoints=PreviewKeypoints(
            xy=torch.tensor([points], dtype=torch.float32),
            conf=torch.tensor([confidences], dtype=torch.float32),
        ),
    )


def predict_image(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 640,
) -> Any:
    return model.predict(image_bgr, conf=conf, imgsz=imgsz, verbose=False)


def _joint_color(name: str) -> tuple[int, int, int]:
    if name in LEFT_KEYPOINTS:
        return PALETTE.left_joint
    if name in RIGHT_KEYPOINTS:
        return PALETTE.right_joint
    if name in MIDLINE_KEYPOINTS:
        return PALETTE.mid_joint
    return PALETTE.mid_joint


def _bone_color(start_name: str, end_name: str) -> tuple[int, int, int]:
    if start_name in LEFT_KEYPOINTS and end_name in LEFT_KEYPOINTS:
        return PALETTE.left_bone
    if start_name in RIGHT_KEYPOINTS and end_name in RIGHT_KEYPOINTS:
        return PALETTE.right_bone
    return PALETTE.mid_bone


def draw_pose_result(
    image_bgr: np.ndarray,
    result: Any,
    keypoint_conf: float = 0.35,
) -> np.ndarray:
    validate_dog_keypoints(result)
    canvas = image_bgr.copy()
    boxes = result.boxes
    keypoints = result.keypoints

    if boxes is None or keypoints is None or len(boxes) == 0:
        return canvas

    xy = keypoints.xy.cpu().numpy()
    kp_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
    box_xyxy = boxes.xyxy.cpu().numpy()
    box_conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None

    for det_index, points in enumerate(xy):
        x1, y1, x2, y2 = box_xyxy[det_index].astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), PALETTE.box, 2)
        label = f"dog {box_conf[det_index]:.2f}" if box_conf is not None else "dog"
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            PALETTE.box,
            2,
            cv2.LINE_AA,
        )

        for start_idx, end_idx in DOG_SKELETON_EDGES:
            start_point = points[start_idx]
            end_point = points[end_idx]
            start_conf = kp_conf[det_index][start_idx] if kp_conf is not None else 1.0
            end_conf = kp_conf[det_index][end_idx] if kp_conf is not None else 1.0
            if start_conf < keypoint_conf or end_conf < keypoint_conf:
                continue

            start_name = DOG_KEYPOINT_NAMES[start_idx]
            end_name = DOG_KEYPOINT_NAMES[end_idx]
            cv2.line(
                canvas,
                tuple(start_point.astype(int)),
                tuple(end_point.astype(int)),
                _bone_color(start_name, end_name),
                3,
                cv2.LINE_AA,
            )

        for point_index, point in enumerate(points):
            confidence = kp_conf[det_index][point_index] if kp_conf is not None else 1.0
            if confidence < keypoint_conf:
                continue

            point_name = DOG_KEYPOINT_NAMES[point_index]
            center = tuple(point.astype(int))
            cv2.circle(canvas, center, 5, _joint_color(point_name), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 8, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def result_to_records(result: Any, keypoint_conf: float = 0.35) -> list[dict[str, Any]]:
    validate_dog_keypoints(result)
    records: list[dict[str, Any]] = []
    if result.keypoints is None:
        return records

    xy = result.keypoints.xy.cpu().numpy()
    kp_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
    for det_index, points in enumerate(xy):
        for point_index, point in enumerate(points):
            confidence = kp_conf[det_index][point_index] if kp_conf is not None else 1.0
            if confidence < keypoint_conf:
                continue
            records.append(
                {
                    "instance": det_index,
                    "joint": DOG_KEYPOINT_NAMES[point_index],
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "confidence": float(confidence),
                }
            )
    return records
