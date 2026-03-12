from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from dog_pose_mvp.gait import TREND_LABELS, analyze_video, frame_metrics_table
from dog_pose_mvp.visualization import (
    build_preview_result,
    draw_pose_result,
    load_image,
    load_model,
    predict_image,
    resolve_default_model_path,
    result_to_records,
)

st.set_page_config(page_title="Dog Pose MVP", layout="wide")

st.title("Dog Pose Estimation MVP")
st.caption("단일 이미지 pose overlay와 비디오 기반 보행 시계열 분석을 함께 확인할 수 있다.")


@st.cache_resource(show_spinner="YOLO 모델을 로딩 중입니다...")
def get_model(path: str):
    return load_model(path)


def _discover_best_model_paths() -> list[str]:
    return sorted(str(path) for path in Path("runs").glob("**/best.pt"))


def _format_model_option(path: str) -> str:
    candidate = Path(path)
    try:
        return str(candidate.relative_to(Path.cwd()))
    except ValueError:
        return str(candidate)


def render_image_demo(
    model_path: str,
    conf_threshold: float,
    keypoint_threshold: float,
    image_size: int,
) -> None:
    st.subheader("Image Pose Demo")
    sample_candidates = sorted(Path("/Users/hossay/Documents/datasets/dog-pose/images/val").glob("*.jpg"))[:20]

    controls_left, controls_right = st.columns([1.2, 1.0])
    with controls_left:
        uploaded_file = st.file_uploader(
            "강아지 이미지를 업로드하세요.",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_uploader",
        )
    with controls_right:
        sample_path = st.selectbox(
            "Validation sample",
            options=["None", *[str(path) for path in sample_candidates]],
            index=0,
            key="sample_path",
        )
        preview_mode = st.radio(
            "Visualization source",
            options=["Model inference", "Dataset label preview"],
            index=0,
            disabled=sample_path == "None",
            key="preview_mode",
        )

    st.caption(
        "Validation sample을 고르면 정답 라벨 미리보기 또는 모델 추론 결과를 바로 비교할 수 있다."
    )

    if not uploaded_file and sample_path == "None":
        st.info("이미지를 업로드하거나 validation sample을 선택하면 skeleton overlay를 확인할 수 있다.")
        return

    image = Image.open(uploaded_file) if uploaded_file else Image.open(sample_path)
    image_bgr = load_image(image)
    try:
        if sample_path != "None" and preview_mode == "Dataset label preview":
            result = build_preview_result(sample_path, image.size)
        else:
            model = get_model(model_path)
            results = predict_image(model, image_bgr, conf=conf_threshold, imgsz=image_size)
            result = results[0]
        rendered_bgr = draw_pose_result(image_bgr, result, keypoint_conf=keypoint_threshold)
        records = result_to_records(result, keypoint_conf=keypoint_threshold)
    except Exception as exc:  # pragma: no cover - Streamlit interaction path
        st.error(str(exc))
        return

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("Skeleton Overlay")
        st.image(rendered_bgr[:, :, ::-1], channels="RGB", use_container_width=True)

    with right:
        st.subheader("Joint Table")
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("현재 threshold에서 표시할 keypoint가 없다.")


def _format_metric(value: float | None, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}{suffix}"


def _analysis_cache_key(
    video_bytes: bytes,
    model_path: str,
    conf_threshold: float,
    keypoint_threshold: float,
    image_size: int,
    analysis_fps: float,
    max_frames: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"overlay-video-export-v5-browser-h264-mpl-screening")
    digest.update(video_bytes)
    digest.update(
        f"{model_path}|{conf_threshold:.3f}|{keypoint_threshold:.3f}|{image_size}|{analysis_fps:.3f}|{max_frames}".encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def render_video_demo(
    model_path: str,
    conf_threshold: float,
    keypoint_threshold: float,
    image_size: int,
) -> None:
    st.subheader("Video Gait Analysis")
    uploaded_video = st.file_uploader(
        "보행 영상을 업로드하세요.",
        type=["mp4", "mov", "avi", "m4v", "webm"],
        key="video_uploader",
    )
    st.caption(
        "기본은 전체 프레임을 분석하고, 필요할 때만 downsampling 하도록 했다. "
        "가려짐으로 비는 구간은 interpolation 기반 시계열 보정으로 이어 붙여 metric missing 을 없앤다. "
        "또한 후지 offloading, extension deficit, ROM 비대칭을 이용한 슬개골 탈구 관련 screening 지표를 함께 계산한다."
    )

    if uploaded_video is None:
        st.info("영상 파일을 업로드하면 프레임별 오버레이와 시간 추이 그래프를 계산한다.")
        return

    video_bytes = uploaded_video.getvalue()
    st.video(video_bytes)

    option_left, option_right = st.columns(2)
    with option_left:
        analysis_fps = st.number_input(
            "Analysis FPS (0 = 전체 프레임)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="0이면 원본 FPS를 그대로 사용해 downsampling 없이 전체 프레임을 분석한다.",
            key="analysis_fps",
        )
    with option_right:
        max_frames = st.number_input(
            "Max analyzed frames (0 = 전체)",
            min_value=0,
            value=0,
            step=30,
            help="0이면 프레임 수 제한 없이 영상 끝까지 분석한다.",
            key="max_frames",
        )

    cache_key = _analysis_cache_key(
        video_bytes=video_bytes,
        model_path=model_path,
        conf_threshold=conf_threshold,
        keypoint_threshold=keypoint_threshold,
        image_size=image_size,
        analysis_fps=float(analysis_fps),
        max_frames=int(max_frames),
    )

    run_analysis = st.button("영상 보행 분석 실행", type="primary", key="run_video_analysis")
    cached_payload = st.session_state.get("video_analysis")

    if run_analysis:
        progress = st.progress(0.0, text="영상 프레임 분석과 pose 추론을 시작한다.")
        progress_note = st.empty()
        temp_path: Path | None = None

        def update_progress(done: int, total: int, frame_index: int, time_sec: float) -> None:
            ratio = min(done / max(total, 1), 1.0)
            progress.progress(
                ratio,
                text=f"{done}/{total} 프레임 분석 중 · 원본 frame {frame_index} · {time_sec:.2f}s",
            )
            progress_note.caption("각 분석 프레임에서 스켈레톤, body axis, paw phase 지표를 함께 계산하고 있다.")

        try:
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(video_bytes)
                temp_path = Path(temp_file.name)

            model = get_model(model_path)
            analysis = analyze_video(
                video_path=temp_path,
                model=model,
                conf_threshold=conf_threshold,
                keypoint_threshold=keypoint_threshold,
                image_size=image_size,
                analysis_fps=float(analysis_fps),
                max_frames=int(max_frames),
                progress_callback=update_progress,
            )
        except Exception as exc:  # pragma: no cover - Streamlit interaction path
            progress_note.empty()
            st.error(str(exc))
        else:
            st.session_state["video_analysis"] = {"key": cache_key, "result": analysis}
            progress.progress(1.0, text="영상 분석이 완료되었다.")
            progress_note.empty()
            cached_payload = st.session_state["video_analysis"]
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    if not cached_payload or cached_payload.get("key") != cache_key:
        st.info("현재 설정으로 분석 결과를 보려면 `영상 보행 분석 실행` 버튼을 눌러주세요.")
        return

    analysis = cached_payload["result"]
    analyzed_frames = analysis.analyzed_frames
    if not analyzed_frames:
        st.warning("분석 가능한 프레임이 없어 결과를 만들지 못했다.")
        return

    summary_cols = st.columns(6)
    summary_cols[0].metric("원본 FPS", _format_metric(analysis.source_fps))
    summary_cols[1].metric("분석 FPS", _format_metric(analysis.analyzed_fps))
    summary_cols[2].metric("분석 프레임", f"{len(analyzed_frames)}")
    summary_cols[3].metric("원본 길이", _format_metric(analysis.total_duration_sec, "s"))
    summary_cols[4].metric(
        "추정 step rate",
        _format_metric(analysis.gait_stats.get("estimated_step_rate_hz"), " Hz"),
    )
    summary_cols[5].metric(
        "주기 일관성",
        _format_metric(analysis.gait_stats.get("cycle_consistency_score")),
    )

    st.subheader("결과 오버레이 playback")
    if analysis.playback_video_bytes:
        playback_left, playback_right = st.columns([1.25, 0.75], gap="large")
        with playback_left:
            st.video(analysis.playback_video_bytes, format="video/mp4")
        with playback_right:
            st.caption(
                "분석된 오버레이 프레임들을 시간 순서대로 저장한 결과 비디오다."
            )
            st.download_button(
                "오버레이 비디오 다운로드",
                data=analysis.playback_video_bytes,
                file_name=f"{Path(uploaded_video.name).stem}-overlay-preview.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
    else:
        st.info("현재 환경에서는 결과 preview 비디오 인코딩을 만들지 못해 프레임 이미지 보기만 제공한다.")

    st.subheader("프레임 오버레이 검사")
    selected_index = st.select_slider(
        "검토할 프레임",
        options=list(range(len(analyzed_frames))),
        value=0,
        format_func=lambda idx: f"frame {analyzed_frames[idx].frame_index} · {analyzed_frames[idx].time_sec:.2f}s",
    )
    selected_frame = analyzed_frames[selected_index]

    frame_left, frame_right = st.columns([1.15, 0.85], gap="large")
    with frame_left:
        st.image(
            selected_frame.overlay_jpeg,
            caption=f"frame {selected_frame.frame_index} · {selected_frame.time_sec:.2f}s",
            use_container_width=True,
        )

    with frame_right:
        st.write("선택 프레임 요약")
        st.dataframe(
            frame_metrics_table(selected_frame.metrics),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("요약 metric은 interpolation/smoothing 을 거친 값이고, 아래 keypoint 표는 원본 프레임 검출 결과다.")
        if selected_frame.records:
            st.write("선택 프레임 keypoints")
            st.dataframe(
                pd.DataFrame(selected_frame.records),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("선택 프레임에서 threshold를 넘는 keypoint가 없다.")

    st.subheader("대표 프레임 미리보기")
    gallery_step = max(1, len(analyzed_frames) // 8)
    gallery_frames = analyzed_frames[::gallery_step]
    if gallery_frames[-1] is not analyzed_frames[-1]:
        gallery_frames.append(analyzed_frames[-1])
    st.image(
        [frame.overlay_jpeg for frame in gallery_frames],
        caption=[f"frame {frame.frame_index} · {frame.time_sec:.2f}s" for frame in gallery_frames],
        width=180,
    )

    st.subheader("시간 추이 그래프")
    trend_df = analysis.trend_df.copy()
    trend_df = trend_df.set_index("time_sec")

    paw_phase_cols = [
        "front_left_paw_phase_norm",
        "front_right_paw_phase_norm",
        "rear_left_paw_phase_norm",
        "rear_right_paw_phase_norm",
    ]
    posture_cols = [
        "body_axis_angle_deg",
        "head_height_norm",
        "tail_swing_norm",
        "front_stride_gap_norm",
        "rear_stride_gap_norm",
    ]
    phase_chart = trend_df[[column for column in paw_phase_cols if column in trend_df.columns]].rename(
        columns=TREND_LABELS
    )
    posture_chart = trend_df[[column for column in posture_cols if column in trend_df.columns]].rename(
        columns=TREND_LABELS
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.write("Paw phase 추이")
        st.line_chart(phase_chart, use_container_width=True, height=300)
    with chart_right:
        st.write("Body / posture 추이")
        st.line_chart(posture_chart, use_container_width=True, height=300)
    st.caption("차트와 프레임 메트릭은 interpolation과 smoothing 을 거친 연속 시계열 값이다.")

    st.subheader("시계열 기반 보행 특성")
    overview_cols = st.columns(3)
    with overview_cols[0]:
        st.metric("일반 보행 해석", analysis.gait_status)
    with overview_cols[1]:
        st.metric("슬개골 탈구 screening", analysis.mpl_status)
    with overview_cols[2]:
        st.metric(
            "해석 신뢰도",
            _format_metric(analysis.gait_stats.get("valid_frame_ratio")),
        )

    st.caption(analysis.gait_note)
    if analysis.gait_reasons:
        st.write("왜 이렇게 보나")
        for reason in analysis.gait_reasons:
            st.write(f"- {reason}")

    st.dataframe(analysis.gait_summary, use_container_width=True, hide_index=True)

    st.subheader("슬개골 탈구 관련 스크리닝")
    mpl_cols = st.columns(3)
    mpl_cols[0].metric("스크리닝 상태", analysis.mpl_status)
    mpl_cols[1].metric("주된 의심 측", analysis.mpl_primary_side)
    mpl_cols[2].metric(
        "Rear offloading ratio",
        _format_metric(analysis.mpl_stats.get("rear_offloading_ratio")),
    )
    st.caption(analysis.mpl_note)
    if analysis.mpl_reasons:
        st.write("왜 슬개골 탈구를 의심하나")
        for reason in analysis.mpl_reasons:
            st.write(f"- {reason}")
    st.dataframe(analysis.mpl_summary, use_container_width=True, hide_index=True)

    with st.expander("정상 범위와 해석 기준"):
        st.write(analysis.interpretation_note)

    with st.expander("슬개골 탈구 문헌 근거"):
        st.markdown(analysis.mpl_evidence_markdown)

    with st.expander("프레임별 안정화 지표 보기"):
        st.dataframe(analysis.trend_df, use_container_width=True, hide_index=True)


with st.sidebar:
    st.header("Inference Settings")
    default_model_path = resolve_default_model_path()
    best_model_paths = _discover_best_model_paths()
    if not best_model_paths:
        model_path = st.text_input("Model checkpoint", value=default_model_path)
        st.caption("`runs` 아래에서 선택 가능한 `best.pt`가 없어 경로 직접 입력 모드로 표시한다.")
    else:
        default_index = best_model_paths.index(default_model_path) if default_model_path in best_model_paths else 0
        model_path = st.selectbox(
            "Model checkpoint",
            options=best_model_paths,
            index=default_index,
            format_func=_format_model_option,
            help="`runs` 아래에서 찾은 `best.pt` 체크포인트만 선택한다.",
        )
    conf_threshold = st.slider("Detection confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    keypoint_threshold = st.slider("Skeleton confidence", min_value=0.05, max_value=0.95, value=0.35, step=0.05)
    image_size = st.select_slider("Inference image size", options=[480, 640, 768, 960], value=640)
    st.markdown(
        "\n".join(
            [
                "- 기본 체크포인트는 `runs/pose/latest_run.json`에 기록된 최신 dog-pose 가중치를 우선 탐색한다.",
                "- `yolo26n-pose.pt`는 학습 시작용 warm-start 체크포인트이며, dog skeleton 데모용 24-keypoint 출력은 보장하지 않는다.",
                "- 비디오 분석은 기본적으로 전체 프레임을 추론하며, 필요하면 `Analysis FPS`나 `Max analyzed frames`로 줄일 수 있다.",
                "- 슬개골 탈구 screening은 후지 offloading, ROM, extension surrogate를 이용한 pose-based heuristic이며 확진을 대체하지 않는다.",
                "- 보행 특성 값은 자세 시계열 기반 heuristic이며 수의학적 진단을 대체하지 않는다.",
            ]
        )
    )


image_tab, video_tab = st.tabs(["Image Demo", "Video Gait Analysis"])

with image_tab:
    render_image_demo(
        model_path=model_path,
        conf_threshold=conf_threshold,
        keypoint_threshold=keypoint_threshold,
        image_size=image_size,
    )

with video_tab:
    render_video_demo(
        model_path=model_path,
        conf_threshold=conf_threshold,
        keypoint_threshold=keypoint_threshold,
        image_size=image_size,
    )
