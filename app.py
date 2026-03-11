from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

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
st.caption("Ultralytics dog-pose 모델 결과를 스켈레톤 오버레이로 시각화한다.")

with st.sidebar:
    st.header("Inference Settings")
    model_path = st.text_input("Model checkpoint", value=resolve_default_model_path())
    conf_threshold = st.slider("Detection confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    keypoint_threshold = st.slider("Skeleton confidence", min_value=0.05, max_value=0.95, value=0.35, step=0.05)
    image_size = st.select_slider("Inference image size", options=[480, 640, 768, 960], value=640)
    sample_candidates = sorted(Path("/Users/hossay/Documents/datasets/dog-pose/images/val").glob("*.jpg"))[:20]
    sample_path = st.selectbox(
        "Validation sample",
        options=["None", *[str(path) for path in sample_candidates]],
        index=0,
    )
    preview_mode = st.radio(
        "Visualization source",
        options=["Model inference", "Dataset label preview"],
        index=0,
        disabled=sample_path == "None",
    )
    st.markdown(
        "\n".join(
            [
                "- 기본 체크포인트는 `runs/pose/latest_run.json`에 기록된 최신 dog-pose 가중치를 우선 탐색한다.",
                "- `yolo26n-pose.pt`는 학습 시작용 warm-start 체크포인트이며, dog skeleton 데모용 24-keypoint 출력은 보장하지 않는다.",
                "- `Dataset label preview`는 validation sample에 포함된 정답 keypoint를 바로 skeleton으로 보여준다.",
                "- 업로드 없이도 validation sample을 바로 열어볼 수 있다.",
            ]
        )
    )


@st.cache_resource(show_spinner="YOLO 모델을 로딩 중입니다...")
def get_model(path: str):
    return load_model(path)


uploaded_file = st.file_uploader("강아지 이미지를 업로드하세요.", type=["jpg", "jpeg", "png", "webp"])

if not uploaded_file and sample_path == "None":
    st.info("이미지를 업로드하거나 sidebar에서 validation sample을 선택하면 skeleton overlay를 확인할 수 있다.")
    st.stop()

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
    st.stop()

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
