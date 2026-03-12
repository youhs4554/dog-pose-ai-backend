from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from dog_pose_mvp.skeleton import DOG_KEYPOINT_NAMES
from dog_pose_mvp.visualization import draw_pose_result, predict_image, result_to_records, validate_dog_keypoints

PAW_KEYPOINTS = [
    "front_left_paw",
    "front_right_paw",
    "rear_left_paw",
    "rear_right_paw",
]

FRAME_METRIC_LABELS = {
    "visible_keypoints": "Visible keypoints",
    "detection_conf": "Detection confidence",
    "body_length_px": "Body length (px)",
    "body_axis_angle_deg": "Body axis angle (deg)",
    "head_angle_deg": "Head angle (deg)",
    "head_height_norm": "Head height / body length",
    "tail_height_norm": "Tail height / body length",
    "tail_swing_norm": "Tail swing / body length",
    "front_step_width_norm": "Front step width / body length",
    "rear_step_width_norm": "Rear step width / body length",
    "front_stride_gap_norm": "Front stride gap / body length",
    "rear_stride_gap_norm": "Rear stride gap / body length",
    "rear_left_distal_angle_deg": "Rear left distal joint angle (deg)",
    "rear_right_distal_angle_deg": "Rear right distal joint angle (deg)",
    "rear_left_proximal_angle_deg": "Rear left proximal joint angle (deg)",
    "rear_right_proximal_angle_deg": "Rear right proximal joint angle (deg)",
    "rear_left_paw_drop_norm": "Rear left paw drop / body length",
    "rear_right_paw_drop_norm": "Rear right paw drop / body length",
    "rear_paw_drop_diff_norm": "Rear paw drop asymmetry / body length",
}

TREND_LABELS = {
    "front_left_paw_phase_norm": "Front left paw phase",
    "front_right_paw_phase_norm": "Front right paw phase",
    "rear_left_paw_phase_norm": "Rear left paw phase",
    "rear_right_paw_phase_norm": "Rear right paw phase",
    "body_axis_angle_deg": "Body axis angle (deg)",
    "head_height_norm": "Head height / body length",
    "tail_swing_norm": "Tail swing / body length",
    "front_stride_gap_norm": "Front stride gap / body length",
    "rear_stride_gap_norm": "Rear stride gap / body length",
}


@dataclass
class FrameAnalysis:
    frame_index: int
    time_sec: float
    overlay_jpeg: bytes
    metrics: dict[str, float | int]
    records: list[dict[str, Any]]


@dataclass
class _BufferedFrame:
    frame_index: int
    time_sec: float
    pose_preview_bgr: np.ndarray
    metrics: dict[str, float | int]
    records: list[dict[str, Any]]


@dataclass
class VideoAnalysis:
    source_fps: float
    analyzed_fps: float
    total_frames: int
    total_duration_sec: float
    frame_step: int
    analyzed_frames: list[FrameAnalysis]
    playback_video_bytes: bytes | None
    trend_df: pd.DataFrame
    gait_summary: pd.DataFrame
    gait_stats: dict[str, float]
    gait_note: str
    gait_status: str
    gait_reasons: list[str]
    interpretation_note: str
    mpl_summary: pd.DataFrame
    mpl_stats: dict[str, float]
    mpl_note: str
    mpl_status: str
    mpl_reasons: list[str]
    mpl_primary_side: str
    mpl_evidence_markdown: str


def _point_valid(point: np.ndarray) -> bool:
    return point.shape == (2,) and np.isfinite(point).all()


def _safe_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    if not (_point_valid(point_a) and _point_valid(point_b)):
        return float("nan")
    return float(np.linalg.norm(point_a - point_b))


def _safe_segment_angle_deg(point_a: np.ndarray, point_b: np.ndarray) -> float:
    if not (_point_valid(point_a) and _point_valid(point_b)):
        return float("nan")
    vector = point_b - point_a
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def _safe_joint_angle_deg(point_a: np.ndarray, joint_point: np.ndarray, point_c: np.ndarray) -> float:
    if not (_point_valid(point_a) and _point_valid(joint_point) and _point_valid(point_c)):
        return float("nan")

    vector_a = point_a - joint_point
    vector_c = point_c - joint_point
    norm_a = np.linalg.norm(vector_a)
    norm_c = np.linalg.norm(vector_c)
    if norm_a <= 1e-6 or norm_c <= 1e-6:
        return float("nan")

    cosine = float(np.clip(np.dot(vector_a, vector_c) / (norm_a * norm_c), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _safe_projection(
    point: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    scale: float,
) -> float:
    if not (_point_valid(point) and _point_valid(origin) and np.isfinite(axis).all() and scale > 1e-6):
        return float("nan")
    return float(np.dot(point - origin, axis) / scale)


def _safe_vertical_offset(point_a: np.ndarray, point_b: np.ndarray, scale: float) -> float:
    if not (_point_valid(point_a) and _point_valid(point_b) and scale > 1e-6):
        return float("nan")
    return float((point_b[1] - point_a[1]) / scale)


def _safe_difference(value_a: float, value_b: float) -> float:
    if not (np.isfinite(value_a) and np.isfinite(value_b)):
        return float("nan")
    return float(value_a - value_b)


def _safe_abs_difference(value_a: float, value_b: float) -> float:
    diff = _safe_difference(value_a, value_b)
    if not np.isfinite(diff):
        return float("nan")
    return float(abs(diff))


def _resize_for_preview(image_bgr: np.ndarray, max_width: int = 960) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if width <= max_width:
        return image_bgr

    scale = max_width / float(width)
    resized = cv2.resize(
        image_bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def _ensure_even_frame_size(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    target_height = height - (height % 2)
    target_width = width - (width % 2)
    if target_height == height and target_width == width:
        return image_bgr
    return image_bgr[:target_height, :target_width]


def _encode_jpeg(image_bgr: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 88],
    )
    if not success:
        raise ValueError("Could not encode analyzed frame preview.")
    return encoded.tobytes()


def _encode_preview(image_bgr: np.ndarray) -> bytes:
    preview = _ensure_even_frame_size(_resize_for_preview(image_bgr))
    return _encode_jpeg(preview)


def _create_preview_video_writer(
    preview_frame_bgr: np.ndarray,
    fps: float,
) -> tuple[cv2.VideoWriter | None, Path | None]:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = Path(temp_file.name)
    temp_file.close()

    height, width = preview_frame_bgr.shape[:2]
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if writer.isOpened():
        return writer, temp_path

    writer.release()
    temp_path.unlink(missing_ok=True)
    return None, None


def _encode_video_bytes(frames_bgr: list[np.ndarray], fps: float) -> bytes | None:
    if not frames_bgr:
        return None

    writer, temp_path = _create_preview_video_writer(frames_bgr[0], fps=fps)
    if writer is None or temp_path is None:
        return None

    try:
        for frame_bgr in frames_bgr:
            writer.write(frame_bgr)
    finally:
        writer.release()

    try:
        browser_bytes = _transcode_video_for_browser(temp_path)
        if browser_bytes is not None:
            return browser_bytes
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


def _transcode_video_for_browser(source_path: Path) -> bytes | None:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = Path(temp_file.name)
    temp_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        str(output_path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0 or not output_path.exists():
            return None
        return output_path.read_bytes()
    except FileNotFoundError:
        return None
    finally:
        output_path.unlink(missing_ok=True)


def _draw_metric_box(image_bgr: np.ndarray, metrics: dict[str, float | int]) -> np.ndarray:
    canvas = image_bgr.copy()
    overlay = canvas.copy()
    info_lines = [
        f"t={float(metrics.get('time_sec', 0.0)):.2f}s",
        f"body={_format_metric_value(metrics.get('body_axis_angle_deg'), 'deg')}",
        f"head={_format_metric_value(metrics.get('head_height_norm'), 'norm')}",
        f"tail={_format_metric_value(metrics.get('tail_swing_norm'), 'norm')}",
        f"front gap={_format_metric_value(metrics.get('front_stride_gap_norm'), 'norm')}",
        f"rear gap={_format_metric_value(metrics.get('rear_stride_gap_norm'), 'norm')}",
    ]
    info_lines = [line for line in info_lines if "n/a" not in line]
    if not info_lines:
        return canvas

    box_height = 16 + 28 * len(info_lines)
    cv2.rectangle(overlay, (12, 12), (360, 12 + box_height), (12, 18, 28), -1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, dst=canvas)

    for index, line in enumerate(info_lines):
        cv2.putText(
            canvas,
            line,
            (24, 40 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 247, 250),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _extract_primary_points(
    result: Any,
    keypoint_conf: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    validate_dog_keypoints(result)

    default_points = {name: np.array([np.nan, np.nan], dtype=float) for name in DOG_KEYPOINT_NAMES}
    default_conf = {name: float("nan") for name in DOG_KEYPOINT_NAMES}
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return default_points, default_conf, float("nan")

    box_conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
    det_index = int(np.argmax(box_conf)) if box_conf is not None else 0
    detection_conf = float(box_conf[det_index]) if box_conf is not None else float("nan")

    xy = result.keypoints.xy[det_index].cpu().numpy()
    kp_conf = result.keypoints.conf[det_index].cpu().numpy() if result.keypoints.conf is not None else None
    point_map: dict[str, np.ndarray] = {}
    conf_map: dict[str, float] = {}
    for point_index, point_name in enumerate(DOG_KEYPOINT_NAMES):
        confidence = float(kp_conf[point_index]) if kp_conf is not None else 1.0
        conf_map[point_name] = confidence
        if confidence >= keypoint_conf:
            point_map[point_name] = xy[point_index].astype(float)
        else:
            point_map[point_name] = np.array([np.nan, np.nan], dtype=float)

    return point_map, conf_map, detection_conf


def _body_reference_axes(point_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    tail_start = point_map["tail_start"]
    withers = point_map["withers"]
    nose = point_map["nose"]
    body_length = _safe_distance(tail_start, withers)
    if not np.isfinite(body_length) or body_length <= 1e-6:
        body_length = _safe_distance(tail_start, nose)

    if _point_valid(tail_start) and _point_valid(withers):
        origin = (tail_start + withers) / 2.0
    elif _point_valid(tail_start):
        origin = tail_start.copy()
    elif _point_valid(withers):
        origin = withers.copy()
    else:
        origin = np.array([np.nan, np.nan], dtype=float)

    forward_target = nose if _point_valid(nose) else withers
    if not (_point_valid(tail_start) and _point_valid(forward_target)):
        return origin, np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), float("nan")

    forward_vector = forward_target - tail_start
    norm = np.linalg.norm(forward_vector)
    if norm <= 1e-6:
        return origin, np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), float("nan")

    body_axis = forward_vector / norm
    lateral_axis = np.array([-body_axis[1], body_axis[0]])
    return origin, body_axis, lateral_axis, float(body_length)


def _frame_metrics_from_result(
    result: Any,
    frame_index: int,
    time_sec: float,
    keypoint_conf: float,
) -> dict[str, float | int]:
    point_map, conf_map, detection_conf = _extract_primary_points(result, keypoint_conf=keypoint_conf)
    origin, body_axis, lateral_axis, body_length = _body_reference_axes(point_map)
    withers = point_map["withers"]
    nose = point_map["nose"]
    tail_start = point_map["tail_start"]
    tail_end = point_map["tail_end"]
    rear_left_paw = point_map["rear_left_paw"]
    rear_left_knee = point_map["rear_left_knee"]
    rear_left_elbow = point_map["rear_left_elbow"]
    rear_right_paw = point_map["rear_right_paw"]
    rear_right_knee = point_map["rear_right_knee"]
    rear_right_elbow = point_map["rear_right_elbow"]

    metrics: dict[str, float | int] = {
        "frame_index": frame_index,
        "time_sec": float(time_sec),
        "detected": int(np.isfinite(detection_conf)),
        "visible_keypoints": int(sum(conf >= keypoint_conf for conf in conf_map.values() if np.isfinite(conf))),
        "detection_conf": detection_conf,
        "body_length_px": body_length,
        "body_axis_angle_deg": float(np.degrees(np.arctan2(body_axis[1], body_axis[0])))
        if np.isfinite(body_axis).all()
        else float("nan"),
        "head_angle_deg": _safe_segment_angle_deg(withers, nose),
        "head_height_norm": _safe_vertical_offset(nose, withers, body_length),
        "tail_height_norm": _safe_vertical_offset(tail_end, tail_start, body_length),
        "tail_swing_norm": _safe_projection(tail_end, tail_start, lateral_axis, body_length),
        "rear_left_distal_angle_deg": _safe_joint_angle_deg(rear_left_paw, rear_left_knee, rear_left_elbow),
        "rear_right_distal_angle_deg": _safe_joint_angle_deg(rear_right_paw, rear_right_knee, rear_right_elbow),
        "rear_left_proximal_angle_deg": _safe_joint_angle_deg(rear_left_knee, rear_left_elbow, tail_start),
        "rear_right_proximal_angle_deg": _safe_joint_angle_deg(rear_right_knee, rear_right_elbow, tail_start),
        "rear_left_paw_drop_norm": _safe_vertical_offset(tail_start, rear_left_paw, body_length),
        "rear_right_paw_drop_norm": _safe_vertical_offset(tail_start, rear_right_paw, body_length),
    }
    metrics["rear_paw_drop_diff_norm"] = _safe_abs_difference(
        float(metrics["rear_left_paw_drop_norm"]),
        float(metrics["rear_right_paw_drop_norm"]),
    )

    paw_phase_values: dict[str, float] = {}
    paw_width_values: dict[str, float] = {}
    for point_name in PAW_KEYPOINTS:
        point = point_map[point_name]
        paw_phase_values[point_name] = _safe_projection(point, origin, body_axis, body_length)
        paw_width_values[point_name] = _safe_projection(point, origin, lateral_axis, body_length)
        metrics[f"{point_name}_phase_norm"] = paw_phase_values[point_name]
        metrics[f"{point_name}_width_norm"] = paw_width_values[point_name]

    metrics["front_step_width_norm"] = _safe_abs_difference(
        paw_width_values["front_left_paw"],
        paw_width_values["front_right_paw"],
    )
    metrics["rear_step_width_norm"] = _safe_abs_difference(
        paw_width_values["rear_left_paw"],
        paw_width_values["rear_right_paw"],
    )
    metrics["front_stride_gap_norm"] = _safe_abs_difference(
        paw_phase_values["front_left_paw"],
        paw_phase_values["front_right_paw"],
    )
    metrics["rear_stride_gap_norm"] = _safe_abs_difference(
        paw_phase_values["rear_left_paw"],
        paw_phase_values["rear_right_paw"],
    )
    return metrics


def _rolling_window_size(analyzed_fps: float) -> int:
    window = max(1, int(round(max(float(analyzed_fps), 1.0) * 0.5)))
    if window % 2 == 0:
        window += 1
    return window


def _interpolate_missing_series(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if series.isna().all():
        return np.full_like(values, np.nan, dtype=float)
    return series.interpolate(limit_direction="both").to_numpy(dtype=float)


def _smooth_metric_series(
    values: np.ndarray,
    *,
    analyzed_fps: float,
) -> np.ndarray:
    interpolated = _interpolate_missing_series(values)
    if not np.isfinite(interpolated).any():
        return np.full_like(values, np.nan, dtype=float)

    window = _rolling_window_size(analyzed_fps)
    smoothed = pd.Series(interpolated, dtype="float64").rolling(window=window, center=True, min_periods=1).median()
    return smoothed.to_numpy(dtype=float)


def _stabilize_trend_metrics(trend_df: pd.DataFrame, analyzed_fps: float) -> pd.DataFrame:
    if trend_df.empty:
        return trend_df.copy()

    stabilized = trend_df.copy()
    observed_detected = (
        trend_df["detected"].to_numpy(dtype=float) >= 0.5
        if "detected" in trend_df.columns
        else np.ones(len(trend_df), dtype=bool)
    )
    excluded_columns = {"frame_index", "time_sec", "detected"}
    for column in stabilized.columns:
        if column in excluded_columns or not pd.api.types.is_numeric_dtype(stabilized[column]):
            continue

        values = stabilized[column].to_numpy(dtype=float).copy()
        values[~observed_detected] = np.nan
        smoothed = _smooth_metric_series(values, analyzed_fps=analyzed_fps)
        if column == "visible_keypoints":
            smoothed = np.where(
                np.isfinite(smoothed),
                np.clip(np.rint(smoothed), 0, len(DOG_KEYPOINT_NAMES)),
                np.nan,
            )
        stabilized[column] = smoothed

    if "detected" in stabilized.columns:
        if "detection_conf" in stabilized.columns:
            stabilized["detected"] = np.where(np.isfinite(stabilized["detection_conf"].to_numpy(dtype=float)), 1, 0)
        else:
            stabilized["detected"] = np.where(
                np.isfinite(stabilized.select_dtypes(include=["number"]).to_numpy(dtype=float)).any(axis=1),
                1,
                0,
            )

    return stabilized


def _fill_and_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    filled = _interpolate_missing_series(values)
    if not np.isfinite(filled).any():
        return np.full_like(values, np.nan, dtype=float)

    smoothed = pd.Series(filled, dtype="float64").rolling(window=window, center=True, min_periods=1).mean()
    return smoothed.to_numpy(dtype=float)


def _turning_points(times: np.ndarray, signal: np.ndarray) -> tuple[list[int], list[int]]:
    valid = np.isfinite(times) & np.isfinite(signal)
    if valid.sum() < 5:
        return [], []

    peaks: list[int] = []
    troughs: list[int] = []
    previous_slope = 0.0
    for index in range(1, len(signal)):
        if not (valid[index - 1] and valid[index]):
            continue

        slope = signal[index] - signal[index - 1]
        if abs(slope) < 1e-6:
            continue
        if previous_slope > 0 and slope < 0:
            peaks.append(index - 1)
        elif previous_slope < 0 and slope > 0:
            troughs.append(index - 1)
        previous_slope = slope

    median_dt = float(np.nanmedian(np.diff(times[valid]))) if valid.sum() >= 2 else 0.0
    min_gap = max(1, int(round(0.18 / median_dt))) if median_dt > 1e-6 else 1

    def dedupe(indices: list[int]) -> list[int]:
        kept: list[int] = []
        for candidate in indices:
            if not kept or candidate - kept[-1] >= min_gap:
                kept.append(candidate)
            elif abs(signal[candidate]) > abs(signal[kept[-1]]):
                kept[-1] = candidate
        return kept

    return dedupe(peaks), dedupe(troughs)


def _cycle_stats(times: np.ndarray, values: np.ndarray) -> dict[str, float | int]:
    smoothed = _fill_and_smooth(values)
    peaks, troughs = _turning_points(times, smoothed)

    intervals: list[float] = []
    for points in (peaks, troughs):
        if len(points) >= 2:
            intervals.extend(float(delta) for delta in np.diff(times[points]) if delta > 0)

    cycle_hz = float("nan")
    interval_cv = float("nan")
    if intervals:
        interval_array = np.array(intervals, dtype=float)
        cycle_hz = float(1.0 / np.nanmedian(interval_array))
        mean_interval = float(np.nanmean(interval_array))
        if mean_interval > 1e-6:
            interval_cv = float(np.nanstd(interval_array) / mean_interval)

    valid_smoothed = smoothed[np.isfinite(smoothed)]
    amplitude = (
        float((np.nanpercentile(valid_smoothed, 90) - np.nanpercentile(valid_smoothed, 10)) / 2.0)
        if valid_smoothed.size
        else float("nan")
    )
    return {
        "cycle_hz": cycle_hz,
        "interval_cv": interval_cv,
        "amplitude_norm": amplitude,
        "cycles_detected": int(max(0, len(intervals))),
    }


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")

    left_valid = left[valid]
    right_valid = right[valid]
    if np.nanstd(left_valid) < 1e-6 or np.nanstd(right_valid) < 1e-6:
        return float("nan")
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def _alternation_score(left: np.ndarray, right: np.ndarray) -> float:
    corr = _safe_corr(_fill_and_smooth(left), _fill_and_smooth(right))
    if not np.isfinite(corr):
        return float("nan")
    return float(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))


def _amplitude_balance(value_a: float, value_b: float) -> float:
    if not (np.isfinite(value_a) and np.isfinite(value_b)) or max(value_a, value_b) <= 1e-6:
        return float("nan")
    return float(min(value_a, value_b) / max(value_a, value_b))


def _consistency_score(interval_cvs: list[float]) -> float:
    valid_cvs = [value for value in interval_cvs if np.isfinite(value)]
    if not valid_cvs:
        return float("nan")
    mean_cv = float(np.mean(valid_cvs))
    return float(1.0 / (1.0 + mean_cv))


def _rms(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    centered = valid - np.nanmean(valid)
    return float(np.sqrt(np.nanmean(centered**2)))


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.percentile(valid, percentile))


def _robust_range(values: np.ndarray, lower: float = 10.0, upper: float = 90.0) -> float:
    low = _finite_percentile(values, lower)
    high = _finite_percentile(values, upper)
    if not (np.isfinite(low) and np.isfinite(high)):
        return float("nan")
    return float(high - low)


def _bool_ratio(values: np.ndarray) -> float:
    valid = values[~pd.isna(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid.astype(float)))


def _infer_primary_hindlimb_side(
    left_offload_ratio: float,
    right_offload_ratio: float,
    left_extension_peak_deg: float,
    right_extension_peak_deg: float,
    left_stride_amplitude: float,
    right_stride_amplitude: float,
) -> str:
    left_votes = 0
    right_votes = 0

    if np.isfinite(left_offload_ratio) and np.isfinite(right_offload_ratio):
        if left_offload_ratio >= right_offload_ratio + 0.04:
            left_votes += 1
        elif right_offload_ratio >= left_offload_ratio + 0.04:
            right_votes += 1

    if np.isfinite(left_extension_peak_deg) and np.isfinite(right_extension_peak_deg):
        if left_extension_peak_deg + 6.0 <= right_extension_peak_deg:
            left_votes += 1
        elif right_extension_peak_deg + 6.0 <= left_extension_peak_deg:
            right_votes += 1

    if np.isfinite(left_stride_amplitude) and np.isfinite(right_stride_amplitude):
        if left_stride_amplitude + 0.05 <= right_stride_amplitude:
            left_votes += 1
        elif right_stride_amplitude + 0.05 <= left_stride_amplitude:
            right_votes += 1

    if left_votes == 0 and right_votes == 0:
        return "판정 어려움"
    if left_votes == right_votes:
        return "양측/혼합 패턴"
    return "왼쪽 후지" if left_votes > right_votes else "오른쪽 후지"


def _describe_gait(stats: dict[str, float]) -> str:
    valid_ratio = stats.get("valid_frame_ratio", float("nan"))
    duration = stats.get("analysis_duration_sec", float("nan"))
    if not np.isfinite(valid_ratio) or valid_ratio < 0.4:
        return "유효 프레임 비율이 낮아 보행 특성 해석 신뢰도가 제한적이다."
    if not np.isfinite(duration) or duration < 1.0:
        return "분석 구간이 짧아 패턴보다 순간 자세 변화 위주로 해석하는 편이 안전하다."

    front_alt = stats.get("front_alternation_score", float("nan"))
    rear_alt = stats.get("rear_alternation_score", float("nan"))
    consistency = stats.get("cycle_consistency_score", float("nan"))
    body_sway = stats.get("body_sway_deg_std", float("nan"))

    if (
        np.isfinite(front_alt)
        and np.isfinite(rear_alt)
        and np.isfinite(consistency)
        and np.isfinite(body_sway)
        and front_alt >= 0.65
        and rear_alt >= 0.65
        and consistency >= 0.65
        and body_sway <= 8.0
    ):
        return "좌우 교대와 주기성이 비교적 안정적인 대칭 보행 패턴으로 보인다."
    if (
        np.isfinite(front_alt)
        and np.isfinite(rear_alt)
        and np.isfinite(consistency)
        and (front_alt < 0.4 or rear_alt < 0.4 or consistency < 0.45)
    ):
        return "좌우 교대 또는 주기성이 약해 비대칭/가감속 구간이 포함된 패턴일 수 있다."
    return "일정한 보행 리듬은 보이지만 일부 구간에서 변동성이 남아 있다."


def _status_high_good(value: float, good_min: float, watch_min: float) -> tuple[str, int]:
    if not np.isfinite(value):
        return "판정 불가", 0
    if value >= good_min:
        return "정상 패턴에 가까움", 0
    if value >= watch_min:
        return "경계", 1
    return "주의", 2


def _status_low_good(value: float, good_max: float, watch_max: float) -> tuple[str, int]:
    if not np.isfinite(value):
        return "판정 불가", 0
    if value <= good_max:
        return "정상 패턴에 가까움", 0
    if value <= watch_max:
        return "경계", 1
    return "주의", 2


def _overall_gait_assessment(
    gait_stats: dict[str, float],
    rows: list[dict[str, str]],
) -> tuple[str, str, list[str]]:
    def unique(items: list[str]) -> list[str]:
        return list(dict.fromkeys(items))

    valid_ratio = gait_stats.get("valid_frame_ratio", float("nan"))
    duration = gait_stats.get("analysis_duration_sec", float("nan"))
    if not np.isfinite(valid_ratio) or valid_ratio < 0.5 or not np.isfinite(duration) or duration < 1.0:
        reasons = [
            "유효 프레임 비율이 낮거나 분석 구간이 짧아 판정 신뢰도가 제한된다.",
        ]
        return "판정 유보", "영상 길이 또는 검출 품질이 부족해 건강/이상 상태를 강하게 판단하기 어렵다.", reasons

    concern_reasons = [row["Interpretation"] for row in rows if row["Status"] == "주의"]
    watch_reasons = [row["Interpretation"] for row in rows if row["Status"] == "경계"]

    if len(concern_reasons) >= 2:
        return "주의", "정상 보행에서 기대되는 대칭성/안정성에서 벗어난 지표가 여러 개 보인다.", unique(concern_reasons)[:3]
    if len(concern_reasons) == 1 or len(watch_reasons) >= 2:
        reasons = unique(concern_reasons[:1] + watch_reasons[:2])
        return "경계", "전반적으로 보행은 성립하지만 일부 지표에서 비대칭 또는 변동성이 보인다.", reasons
    return "정상 패턴에 가까움", "현재 지표들은 건강한 보행에서 기대되는 대칭성과 안정성에 대체로 가깝다.", unique(watch_reasons)[:2]


def _interpret_gait_metrics(gait_stats: dict[str, float]) -> tuple[pd.DataFrame, str, list[str], str]:
    rows: list[dict[str, str]] = []

    def add_row(
        metric: str,
        value: str,
        reference: str,
        status: str,
        why: str,
        interpretation: str,
    ) -> None:
        rows.append(
            {
                "Metric": metric,
                "Value": value,
                "Reference": reference,
                "Status": status,
                "Why it matters": why,
                "Interpretation": interpretation,
            }
        )

    valid_ratio = gait_stats.get("valid_frame_ratio", float("nan"))
    status, _ = _status_high_good(valid_ratio, good_min=0.7, watch_min=0.5)
    add_row(
        metric="Analysis confidence",
        value=_format_metric_value(valid_ratio, "score"),
        reference=">= 0.70 권장 (프로젝트 해석 기준)",
        status=status,
        why="유효 프레임 비율이 낮으면 아래 보행 지표도 함께 흔들린다.",
        interpretation="검출된 프레임 비율이 높을수록 보행 해석 신뢰도가 올라간다.",
    )

    analysis_duration = gait_stats.get("analysis_duration_sec", float("nan"))
    duration_status, _ = _status_high_good(analysis_duration, good_min=2.0, watch_min=1.0)
    add_row(
        metric="Analysis duration",
        value=f"{analysis_duration:.2f} s" if np.isfinite(analysis_duration) else "n/a",
        reference=">= 2.0 s 권장 (프로젝트 해석 기준)",
        status=duration_status,
        why="너무 짧은 영상은 정상/이상 패턴보다 순간 동작에 더 크게 좌우된다.",
        interpretation="2초 이상이 되면 여러 step 변화를 볼 수 있어 비교가 조금 더 안정적이다.",
    )

    for label, key, why_text in [
        ("Front alternation score", "front_alternation_score", "건강한 대칭 보행일수록 좌우 전지가 번갈아 움직인다."),
        ("Rear alternation score", "rear_alternation_score", "건강한 대칭 보행일수록 좌우 후지가 번갈아 움직인다."),
    ]:
        value = gait_stats.get(key, float("nan"))
        status, _ = _status_high_good(value, good_min=0.75, watch_min=0.55)
        add_row(
            metric=label,
            value=_format_metric_value(value, "score"),
            reference="0.75-1.00 기대 범위 (대칭 보행 heuristic)",
            status=status,
            why=why_text,
            interpretation="값이 낮을수록 좌우 교대가 약하거나 한쪽 보상 동작이 섞였을 가능성이 있다.",
        )

    consistency = gait_stats.get("cycle_consistency_score", float("nan"))
    status, _ = _status_high_good(consistency, good_min=0.7, watch_min=0.5)
    add_row(
        metric="Cycle consistency score",
        value=_format_metric_value(consistency, "score"),
        reference="0.70-1.00 기대 범위 (리듬 안정 heuristic)",
        status=status,
        why="건강한 보행은 step 간격이 급격히 흔들리지 않는 편이다.",
        interpretation="값이 낮으면 속도 변화, 불안정성, 보상 동작 때문에 리듬이 불규칙할 수 있다.",
    )

    for label, key, why_text in [
        ("Front amplitude balance", "front_amplitude_balance", "좌우 전지 보폭 크기가 비슷하면 하중 분산이 더 균형적일 가능성이 크다."),
        ("Rear amplitude balance", "rear_amplitude_balance", "좌우 후지 보폭 크기가 비슷하면 추진 동작이 더 대칭적일 가능성이 크다."),
    ]:
        value = gait_stats.get(key, float("nan"))
        status, _ = _status_high_good(value, good_min=0.9, watch_min=0.8)
        add_row(
            metric=label,
            value=_format_metric_value(value, "score"),
            reference="0.90-1.00 기대 범위 (좌우 보폭 균형 heuristic)",
            status=status,
            why=why_text,
            interpretation="값이 낮을수록 좌우 중 한쪽 보폭이 더 작거나 보상성 움직임이 섞였을 수 있다.",
        )

    body_sway = gait_stats.get("body_sway_deg_std", float("nan"))
    status, _ = _status_low_good(body_sway, good_max=5.0, watch_max=8.0)
    add_row(
        metric="Body sway (deg std)",
        value=_format_metric_value(body_sway, "deg"),
        reference="<= 5 deg 기대, 5-8 deg 경계, > 8 deg 주의",
        status=status,
        why="몸통 축 흔들림이 커지면 안정성 저하나 비대칭 보상이 동반될 수 있다.",
        interpretation="값이 높을수록 직진성이나 체간 안정성이 떨어졌을 가능성이 있다.",
    )

    head_bob = gait_stats.get("head_bob_rms_norm", float("nan"))
    status, _ = _status_low_good(head_bob, good_max=0.08, watch_max=0.15)
    add_row(
        metric="Head bob RMS",
        value=_format_metric_value(head_bob, "norm"),
        reference="<= 0.08 기대, 0.08-0.15 경계, > 0.15 주의",
        status=status,
        why="머리의 vertical bob/asymmetry 증가는 전지 통증 회피나 보상 동작과 함께 관찰되곤 한다.",
        interpretation="값이 높으면 머리 흔들림이 커져 전지 쪽 보상 가능성을 의심해볼 수 있다.",
    )

    tail_swing = gait_stats.get("tail_swing_rms_norm", float("nan"))
    add_row(
        metric="Tail swing RMS",
        value=_format_metric_value(tail_swing, "norm"),
        reference="절대 정상 범위 없음, 보조 지표",
        status="참고용",
        why="꼬리 움직임은 보행 외에도 감정, 품종, 자세 습관의 영향을 크게 받는다.",
        interpretation="꼬리 흔들림은 다른 지표를 보완하는 참고 정보로만 해석하는 편이 안전하다.",
    )

    for label, key in [
        ("Estimated step rate", "estimated_step_rate_hz"),
        ("Estimated stride frequency", "estimated_stride_frequency_hz"),
    ]:
        add_row(
            metric=label,
            value=_format_metric_value(gait_stats.get(key, float("nan")), "hz"),
            reference="체격/속도/보행 유형 의존, 절대 정상 범위 없음",
            status="참고용",
            why="cadence는 작은 개체, 빠른 속도, trot/walk 여부에 따라 크게 달라진다.",
            interpretation="이 값은 같은 개체를 비슷한 속도와 촬영 조건에서 반복 비교할 때 더 유용하다.",
        )

    overall_status, overall_summary, reasons = _overall_gait_assessment(gait_stats, rows)
    note = (
        "해석 기준은 프로젝트 heuristic이며, 건강한 개의 보행은 좌우 대칭성이 높고 리듬이 일정하다는 점, "
        "그리고 head bob/pelvic asymmetry가 lameness 평가에 활용된다는 수의 보행학 원칙을 바탕으로 잡았다. "
        "문헌에서도 healthy dog gait는 symmetry index가 작고 step:stride 비율이 대체로 50% 부근에 모이는 대칭 패턴으로 설명된다. "
        "다만 stride frequency는 체격, 품종, 속도에 크게 좌우되므로 절대 정상 범위로 해석하지 않는다."
    )
    return pd.DataFrame(rows), overall_status, reasons, f"{overall_status}: {overall_summary}\n{note}"


def _build_gait_summary(
    trend_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], str, str, list[str], str]:
    if trend_df.empty:
        empty = pd.DataFrame(columns=["Metric", "Value", "Reference", "Status", "Why it matters", "Interpretation"])
        return (
            empty,
            {},
            "분석 가능한 프레임이 없어 보행 특성을 계산하지 못했다.",
            "판정 유보",
            ["분석 가능한 프레임이 부족하다."],
            "영상 품질 또는 검출 결과가 부족해 정상 범위 비교를 수행하지 못했다.",
        )

    times = trend_df["time_sec"].to_numpy(dtype=float)
    paw_stats: dict[str, dict[str, float | int]] = {}
    interval_cvs: list[float] = []
    cycle_freqs: list[float] = []

    for point_name in PAW_KEYPOINTS:
        column = f"{point_name}_phase_norm"
        stats = _cycle_stats(times, trend_df[column].to_numpy(dtype=float))
        paw_stats[point_name] = stats
        if np.isfinite(stats["interval_cv"]):
            interval_cvs.append(float(stats["interval_cv"]))
        if np.isfinite(stats["cycle_hz"]):
            cycle_freqs.append(float(stats["cycle_hz"]))

    gait_stats: dict[str, float] = {
        "analysis_duration_sec": float(times[-1] - times[0]) if len(times) >= 2 else 0.0,
        "valid_frame_ratio": float(trend_df["detected"].mean()) if "detected" in trend_df else float("nan"),
        "estimated_stride_frequency_hz": float(np.mean(cycle_freqs)) if cycle_freqs else float("nan"),
        "estimated_step_rate_hz": float(np.mean(cycle_freqs) * 2.0) if cycle_freqs else float("nan"),
        "front_alternation_score": _alternation_score(
            trend_df["front_left_paw_phase_norm"].to_numpy(dtype=float),
            trend_df["front_right_paw_phase_norm"].to_numpy(dtype=float),
        ),
        "rear_alternation_score": _alternation_score(
            trend_df["rear_left_paw_phase_norm"].to_numpy(dtype=float),
            trend_df["rear_right_paw_phase_norm"].to_numpy(dtype=float),
        ),
        "front_amplitude_balance": _amplitude_balance(
            float(paw_stats["front_left_paw"]["amplitude_norm"]),
            float(paw_stats["front_right_paw"]["amplitude_norm"]),
        ),
        "rear_amplitude_balance": _amplitude_balance(
            float(paw_stats["rear_left_paw"]["amplitude_norm"]),
            float(paw_stats["rear_right_paw"]["amplitude_norm"]),
        ),
        "cycle_consistency_score": _consistency_score(interval_cvs),
        "body_sway_deg_std": float(np.nanstd(trend_df["body_axis_angle_deg"].to_numpy(dtype=float))),
        "head_bob_rms_norm": _rms(trend_df["head_height_norm"].to_numpy(dtype=float)),
        "tail_swing_rms_norm": _rms(trend_df["tail_swing_norm"].to_numpy(dtype=float)),
    }

    interpretation_df, overall_status, reasons, interpretation_note = _interpret_gait_metrics(gait_stats)
    note = _describe_gait(gait_stats)
    return interpretation_df, gait_stats, note, overall_status, reasons, interpretation_note


def _build_patellar_luxation_summary(
    trend_df: pd.DataFrame,
    gait_stats: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float], str, str, list[str], str, str]:
    columns = {
        "left_phase": "rear_left_paw_phase_norm",
        "right_phase": "rear_right_paw_phase_norm",
        "left_drop": "rear_left_paw_drop_norm",
        "right_drop": "rear_right_paw_drop_norm",
        "left_proximal": "rear_left_proximal_angle_deg",
        "right_proximal": "rear_right_proximal_angle_deg",
    }
    if any(column not in trend_df.columns for column in columns.values()):
        empty = pd.DataFrame(columns=["Metric", "Value", "Reference", "Status", "Why it matters", "Interpretation"])
        return (
            empty,
            {},
            "후지 관절/발 위치 지표를 충분히 계산하지 못해 슬개골 탈구 스크리닝을 만들지 못했다.",
            "판정 유보",
            ["후지 keypoint 품질이 부족하다."],
            "판정 어려움",
            (
                "슬개골 탈구 관련 스크리닝은 rear paw/관절 keypoint가 안정적으로 추적될 때만 의미가 있다. "
                "현재 영상에서는 필요한 후지 지표가 충분하지 않았다."
            ),
        )

    left_phase = trend_df[columns["left_phase"]].to_numpy(dtype=float)
    right_phase = trend_df[columns["right_phase"]].to_numpy(dtype=float)
    left_drop = trend_df[columns["left_drop"]].to_numpy(dtype=float)
    right_drop = trend_df[columns["right_drop"]].to_numpy(dtype=float)
    left_proximal = trend_df[columns["left_proximal"]].to_numpy(dtype=float)
    right_proximal = trend_df[columns["right_proximal"]].to_numpy(dtype=float)

    left_stride = _cycle_stats(trend_df["time_sec"].to_numpy(dtype=float), left_phase)
    right_stride = _cycle_stats(trend_df["time_sec"].to_numpy(dtype=float), right_phase)
    left_stride_amplitude = float(left_stride["amplitude_norm"])
    right_stride_amplitude = float(right_stride["amplitude_norm"])

    left_extension_peak = _finite_percentile(left_proximal, 90.0)
    right_extension_peak = _finite_percentile(right_proximal, 90.0)
    left_proximal_rom = _robust_range(left_proximal)
    right_proximal_rom = _robust_range(right_proximal)

    rear_phase_diff_rms = _rms(left_phase - right_phase)
    rear_drop_diff_rms = _rms(left_drop - right_drop)
    rear_stride_balance = gait_stats.get("rear_amplitude_balance", float("nan"))
    rear_proximal_rom_balance = _amplitude_balance(left_proximal_rom, right_proximal_rom)
    rear_extension_diff_deg = _safe_abs_difference(left_extension_peak, right_extension_peak)

    valid_offload = (
        np.isfinite(left_drop)
        & np.isfinite(right_drop)
        & np.isfinite(left_proximal)
        & np.isfinite(right_proximal)
    )
    left_offload = valid_offload & (left_drop + 0.05 < right_drop) & (left_proximal + 6.0 < right_proximal)
    right_offload = valid_offload & (right_drop + 0.05 < left_drop) & (right_proximal + 6.0 < left_proximal)
    rear_offloading_ratio = _bool_ratio(left_offload | right_offload)
    left_offload_ratio = _bool_ratio(left_offload)
    right_offload_ratio = _bool_ratio(right_offload)
    primary_side = _infer_primary_hindlimb_side(
        left_offload_ratio=left_offload_ratio,
        right_offload_ratio=right_offload_ratio,
        left_extension_peak_deg=left_extension_peak,
        right_extension_peak_deg=right_extension_peak,
        left_stride_amplitude=left_stride_amplitude,
        right_stride_amplitude=right_stride_amplitude,
    )

    mpl_stats: dict[str, float] = {
        "rear_offloading_ratio": rear_offloading_ratio,
        "rear_left_offload_ratio": left_offload_ratio,
        "rear_right_offload_ratio": right_offload_ratio,
        "rear_paw_drop_asymmetry_rms_norm": rear_drop_diff_rms,
        "rear_phase_asymmetry_rms_norm": rear_phase_diff_rms,
        "rear_stride_balance": rear_stride_balance,
        "rear_proximal_rom_balance": rear_proximal_rom_balance,
        "rear_proximal_extension_diff_deg": rear_extension_diff_deg,
        "rear_left_proximal_rom_deg": left_proximal_rom,
        "rear_right_proximal_rom_deg": right_proximal_rom,
        "rear_left_extension_peak_deg": left_extension_peak,
        "rear_right_extension_peak_deg": right_extension_peak,
    }

    rows: list[dict[str, str]] = []

    def add_row(
        metric: str,
        value: str,
        reference: str,
        status: str,
        why: str,
        interpretation: str,
    ) -> None:
        rows.append(
            {
                "Metric": metric,
                "Value": value,
                "Reference": reference,
                "Status": status,
                "Why it matters": why,
                "Interpretation": interpretation,
            }
        )

    offloading_status, _ = _status_low_good(rear_offloading_ratio, good_max=0.05, watch_max=0.12)
    add_row(
        metric="Rear offloading / skip ratio",
        value=_format_metric_value(rear_offloading_ratio, "score"),
        reference="<= 0.05 기대, 0.05-0.12 경계, > 0.12 주의 (프로젝트 screening 기준)",
        status=offloading_status,
        why="슬개골 탈구에서는 후지를 잠깐 들고 가거나 skip 하듯 체중 부하를 피하는 패턴이 흔하다.",
        interpretation="값이 높을수록 한쪽 후지를 반복해서 덜 딛거나 flexed carry 패턴이 섞였을 가능성이 있다.",
    )

    drop_status, _ = _status_low_good(rear_drop_diff_rms, good_max=0.05, watch_max=0.08)
    add_row(
        metric="Rear paw drop asymmetry RMS",
        value=_format_metric_value(rear_drop_diff_rms, "norm"),
        reference="<= 0.05 기대, 0.05-0.08 경계, > 0.08 주의 (프로젝트 screening 기준)",
        status=drop_status,
        why="체중 부하가 비대칭이면 한쪽 후지가 영상에서 더 자주 높게 유지되는 경향이 있다.",
        interpretation="값이 높을수록 후지 weight-bearing 높이가 좌우 비대칭일 가능성이 크다.",
    )

    stride_status, _ = _status_high_good(rear_stride_balance, good_min=0.9, watch_min=0.8)
    add_row(
        metric="Rear stride excursion balance",
        value=_format_metric_value(rear_stride_balance, "score"),
        reference="0.90-1.00 기대, 0.80-0.90 경계, < 0.80 주의 (프로젝트 screening 기준)",
        status=stride_status,
        why="슬개골 탈구/후지 lameness는 affected limb stride excursion 감소로 이어질 수 있다.",
        interpretation="값이 낮을수록 좌우 후지 전진량 또는 stride amplitude 차이가 커졌을 수 있다.",
    )

    rom_status, _ = _status_high_good(rear_proximal_rom_balance, good_min=0.9, watch_min=0.8)
    add_row(
        metric="Rear proximal joint ROM balance",
        value=_format_metric_value(rear_proximal_rom_balance, "score"),
        reference="0.90-1.00 기대, 0.80-0.90 경계, < 0.80 주의 (프로젝트 screening 기준)",
        status=rom_status,
        why="슬개골 탈구는 stifle extension/flexion ROM 감소와 연결되어 후지 proximal chain ROM 비대칭을 만들 수 있다.",
        interpretation="값이 낮을수록 한쪽 후지의 flexion-extension 가동 범위가 더 제한되었을 수 있다.",
    )

    extension_status, _ = _status_low_good(rear_extension_diff_deg, good_max=6.0, watch_max=12.0)
    add_row(
        metric="Rear proximal extension difference",
        value=_format_metric_value(rear_extension_diff_deg, "deg"),
        reference="<= 6 deg 기대, 6-12 deg 경계, > 12 deg 주의 (프로젝트 screening 기준)",
        status=extension_status,
        why="Grade III MPL에서는 stifle extension 감소가 보고되어, 한쪽 extension peak deficit은 중요한 단서가 된다.",
        interpretation="값이 높을수록 한쪽 후지가 extension end-range에 덜 도달하는 패턴일 수 있다.",
    )

    phase_status, _ = _status_low_good(rear_phase_diff_rms, good_max=0.12, watch_max=0.2)
    add_row(
        metric="Rear phase asymmetry RMS",
        value=_format_metric_value(rear_phase_diff_rms, "norm"),
        reference="<= 0.12 기대, 0.12-0.20 경계, > 0.20 주의 (프로젝트 screening 기준)",
        status=phase_status,
        why="슬개골 탈구가 있으면 후지 교대 타이밍이 흐트러지거나 한쪽만 짧게 쓰는 패턴이 생길 수 있다.",
        interpretation="값이 높을수록 좌우 후지의 전진/후퇴 timing 차이가 커졌을 수 있다.",
    )

    concern_reasons = [row["Interpretation"] for row in rows if row["Status"] == "주의"]
    watch_reasons = [row["Interpretation"] for row in rows if row["Status"] == "경계"]
    unique_reasons = list(dict.fromkeys(concern_reasons + watch_reasons))

    valid_ratio = gait_stats.get("valid_frame_ratio", float("nan"))
    duration = gait_stats.get("analysis_duration_sec", float("nan"))
    if not np.isfinite(valid_ratio) or valid_ratio < 0.5 or not np.isfinite(duration) or duration < 1.0:
        status = "판정 유보"
        reasons = ["후지 시계열이 충분하지 않아 슬개골 탈구 스크리닝 신뢰도가 제한된다."]
    elif len(concern_reasons) >= 2 or (
        np.isfinite(rear_offloading_ratio)
        and rear_offloading_ratio > 0.12
        and (
            (np.isfinite(rear_stride_balance) and rear_stride_balance < 0.85)
            or (np.isfinite(rear_extension_diff_deg) and rear_extension_diff_deg > 6.0)
        )
    ):
        status = "주의"
        reasons = unique_reasons[:4]
    elif len(concern_reasons) == 1 or len(watch_reasons) >= 2:
        status = "경계"
        reasons = unique_reasons[:3]
    else:
        status = "정상 패턴에 가까움"
        reasons = unique_reasons[:2]

    note = (
        f"{status}: "
        "후지 offloading/skip, stride excursion 비대칭, 그리고 후지 proximal joint extension/ROM surrogate를 함께 본 스크리닝 결과다. "
        f"주된 의심 측은 `{primary_side}`로 추정된다."
    )
    evidence_markdown = "\n".join(
        [
            "- [DiGiovanni et al., 2023, *J Am Vet Med Assoc*](https://pubmed.ncbi.nlm.nih.gov/37782523/): objective stance analysis로 patellar luxation dogs의 weight-bearing 변화를 추적했고, 수술 후 개선도 확인했다.",
            "- [Chayatup et al., 2025, *BMC Veterinary Research*](https://pubmed.ncbi.nlm.nih.gov/40393162/): grade III MPL dogs에서 stifle ROM, 특히 extension 제한과 compensatory hip/tarsal motion 변화를 보고했다.",
            "- [Lehmann et al., 2021, *BMC Veterinary Research*](https://pmc.ncbi.nlm.nih.gov/articles/PMC8137626/): MPL dogs에서 stance/toe-off 주변 patellar position과 hindlimb orientation 변화가 정상군과 달랐다.",
            "- [Di Dona et al., 2018, *Veterinary Medicine: Research and Reports*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6026879/): intermittent skipping, flexed carry, crouched posture를 MPL의 전형적 임상 보행 징후로 정리했다.",
            "",
            "현재 프로젝트는 2D monocular video와 dog-pose keypoint만 사용하므로 force-plate나 radiograph 기반 확진은 할 수 없다. "
            "대신 위 문헌에서 반복적으로 언급되는 `후지 offloading`, `후지 extension/ROM 제한`, `후지 비대칭`을 pose-based surrogate로 스크리닝한다.",
        ]
    )
    return pd.DataFrame(rows), mpl_stats, note, status, reasons, primary_side, evidence_markdown


def _format_metric_value(value: float | int | None, kind: str) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    numeric = float(value)
    if kind == "deg":
        return f"{numeric:.1f} deg"
    if kind == "hz":
        return f"{numeric:.2f} Hz"
    if kind == "score":
        return f"{numeric:.2f}"
    if kind == "norm":
        return f"{numeric:.3f}"
    if kind == "px":
        return f"{numeric:.1f} px"
    if kind == "count":
        return str(int(round(numeric)))
    return f"{numeric:.3f}"


def frame_metrics_table(metrics: dict[str, float | int]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for key, label in FRAME_METRIC_LABELS.items():
        value = metrics.get(key)
        if key == "visible_keypoints":
            formatted = _format_metric_value(value, "count")
        elif "angle" in key:
            formatted = _format_metric_value(value, "deg")
        elif key == "body_length_px":
            formatted = _format_metric_value(value, "px")
        elif "conf" in key:
            formatted = _format_metric_value(value, "score")
        else:
            formatted = _format_metric_value(value, "norm")

        if formatted == "n/a":
            continue
        rows.append({"Metric": label, "Value": formatted})
    return pd.DataFrame(rows)


def analyze_video(
    video_path: str | Path,
    model: Any,
    conf_threshold: float,
    keypoint_threshold: float,
    image_size: int,
    analysis_fps: float,
    max_frames: int,
    progress_callback: Callable[[int, int, int, float], None] | None = None,
) -> VideoAnalysis:
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        requested_analysis_fps = float(analysis_fps)
        if source_fps <= 1e-6:
            source_fps = max(requested_analysis_fps, 30.0) if requested_analysis_fps > 1e-6 else 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_duration_sec = float(total_frames / source_fps) if total_frames > 0 else 0.0
        effective_analysis_fps = source_fps if requested_analysis_fps <= 1e-6 else min(requested_analysis_fps, source_fps)
        frame_step = max(1, int(round(source_fps / max(effective_analysis_fps, 0.5))))
        effective_max_frames = int(max_frames) if int(max_frames) > 0 else (total_frames if total_frames > 0 else 10**9)
        estimated_total = (
            min(effective_max_frames, max(1, int(np.ceil(total_frames / frame_step))))
            if total_frames > 0
            else max(1, min(effective_max_frames, 1))
        )
        preview_video_fps = float(source_fps / frame_step)

        buffered_frames: list[_BufferedFrame] = []
        frame_index = 0
        analyzed_count = 0

        while analyzed_count < effective_max_frames:
            success, frame_bgr = capture.read()
            if not success:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            results = predict_image(model, frame_bgr, conf=conf_threshold, imgsz=image_size)
            result = results[0]
            metrics = _frame_metrics_from_result(
                result,
                frame_index=frame_index,
                time_sec=frame_index / source_fps,
                keypoint_conf=keypoint_threshold,
            )
            rendered_bgr = draw_pose_result(frame_bgr, result, keypoint_conf=keypoint_threshold)
            buffered_frames.append(
                _BufferedFrame(
                    frame_index=frame_index,
                    time_sec=float(metrics["time_sec"]),
                    pose_preview_bgr=_ensure_even_frame_size(_resize_for_preview(rendered_bgr)),
                    metrics=metrics,
                    records=result_to_records(result, keypoint_conf=keypoint_threshold),
                )
            )
            analyzed_count += 1
            if progress_callback is not None:
                progress_callback(analyzed_count, estimated_total, frame_index, float(metrics["time_sec"]))
            frame_index += 1
    finally:
        capture.release()

    raw_trend_df = pd.DataFrame([frame.metrics for frame in buffered_frames])
    trend_df = _stabilize_trend_metrics(raw_trend_df, analyzed_fps=preview_video_fps)

    frame_analyses: list[FrameAnalysis] = []
    playback_frames: list[np.ndarray] = []
    for buffered_frame, metrics in zip(buffered_frames, trend_df.to_dict(orient="records")):
        annotated_bgr = _draw_metric_box(buffered_frame.pose_preview_bgr, metrics)
        playback_frames.append(annotated_bgr)
        frame_analyses.append(
            FrameAnalysis(
                frame_index=buffered_frame.frame_index,
                time_sec=buffered_frame.time_sec,
                overlay_jpeg=_encode_jpeg(annotated_bgr),
                metrics=metrics,
                records=buffered_frame.records,
            )
        )

    playback_video_bytes = _encode_video_bytes(playback_frames, fps=preview_video_fps)
    gait_summary, gait_stats, gait_note, gait_status, gait_reasons, interpretation_note = _build_gait_summary(trend_df)
    (
        mpl_summary,
        mpl_stats,
        mpl_note,
        mpl_status,
        mpl_reasons,
        mpl_primary_side,
        mpl_evidence_markdown,
    ) = _build_patellar_luxation_summary(trend_df, gait_stats)
    return VideoAnalysis(
        source_fps=source_fps,
        analyzed_fps=preview_video_fps,
        total_frames=total_frames,
        total_duration_sec=total_duration_sec,
        frame_step=frame_step,
        analyzed_frames=frame_analyses,
        playback_video_bytes=playback_video_bytes,
        trend_df=trend_df,
        gait_summary=gait_summary,
        gait_stats=gait_stats,
        gait_note=gait_note,
        gait_status=gait_status,
        gait_reasons=gait_reasons,
        interpretation_note=interpretation_note,
        mpl_summary=mpl_summary,
        mpl_stats=mpl_stats,
        mpl_note=mpl_note,
        mpl_status=mpl_status,
        mpl_reasons=mpl_reasons,
        mpl_primary_side=mpl_primary_side,
        mpl_evidence_markdown=mpl_evidence_markdown,
    )
