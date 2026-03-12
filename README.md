# Dog Pose MVP

Ultralytics `dog-pose` dataset usage를 기준으로 Apple Silicon `mps`에서 1 epoch 학습을 수행하고, 학습된 pose 모델을 Streamlit MVP로 시연하는 프로젝트다.

참고 문서:

- Ultralytics Dog-Pose dataset docs: <https://docs.ultralytics.com/datasets/pose/dog-pose/>
- Dataset YAML: <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dog-pose.yaml>

## Setup

```bash
uv python install 3.10.19
uv sync --python 3.10.19 --managed-python
```

## 1 epoch train on MPS

```bash
uv run python scripts/train_dog_pose.py
```

`batch=32`로 고정해서 실행하려면:

```bash
uv run python scripts/train_dog_pose.py --batch 32 --name dog-pose-mps-1epoch-b32 --exist-ok
```

빠른 smoke test용 5-step 학습:

```bash
uv run python scripts/train_dog_pose.py --batch 32 --train-steps 5 --skip-val --no-plots --name dog-pose-mps-5step-b32 --exist-ok
```

기본값:

- `model=yolo26n-pose.pt`
- `data=dog-pose.yaml`
- `epochs=1`
- `device=mps`
- `batch=auto` (`mps` 메모리를 실제 probe 해서 가능한 최대 batch 선택)
- 출력 경로: `runs/pose/dog-pose-mps-1epoch-maxbatch`

## Run Streamlit demo

```bash
uv run streamlit run app.py
```

사이드바에서 체크포인트 경로와 confidence threshold를 조정할 수 있다.

주의:

- `yolo26n-pose.pt`는 학습 시작용 warm-start 체크포인트다.
- dog skeleton 데모는 24-keypoint dog-pose 체크포인트가 필요하며, 학습 완료 후 `runs/pose/latest_run.json`에 최신 경로가 기록된다.

## Run FastAPI backend

```bash
uv run dog-pose-api
```

개발 중 reload가 필요하면:

```bash
uv run uvicorn dog_pose_mvp.api.main:app --reload
```

기본 주소는 `http://127.0.0.1:8000`이고 Swagger UI는 `http://127.0.0.1:8000/docs`에서 확인할 수 있다.

환경 변수:

- `DOG_POSE_MODEL_PATH`: 사용할 체크포인트 경로. 설정하지 않으면 `runs/pose/dog-pose-colab-longrun/weights/best.pt`를 기본값으로 사용한다.
- `DOG_POSE_API_HOST`: 바인딩 호스트 (기본 `0.0.0.0`)
- `DOG_POSE_API_PORT`: 포트 (기본 `8000`)
- `DOG_POSE_CORS_ALLOW_ORIGINS`: 쉼표 구분 CORS origin 목록 (기본 `*`)

### Model weight 설정 가이드

FastAPI 백엔드는 기본적으로 Streamlit 데모에 지정된 모델 경로인 `runs/pose/dog-pose-colab-longrun/weights/best.pt`를 사용한다.

서버를 기본 설정으로 실행하려면 학습된 dog-pose weight 파일을 아래 경로에 위치시켜야 한다.

```bash
runs/pose/dog-pose-colab-longrun/weights/best.pt
```

예를 들어 weight 파일을 복사하려면:

```bash
mkdir -p runs/pose/dog-pose-colab-longrun/weights
cp /path/to/your/best.pt runs/pose/dog-pose-colab-longrun/weights/best.pt
```

특정 weight를 명시적으로 고정하고 싶으면 서버 실행 전에 환경 변수를 설정한다.

```bash
export DOG_POSE_MODEL_PATH="/absolute/path/to/best.pt"
uv run dog-pose-api
```

현재 API가 어떤 weight를 보고 있는지 확인하려면:

```bash
curl http://127.0.0.1:8000/api/v1/model
```

`exists=true`와 `resolved_model_path`를 보면 실제 로드 대상 체크포인트를 바로 확인할 수 있다.

단일 이미지 추론 예시:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/inference/image \
  -F "file=@/path/to/dog.jpg" \
  -F "conf_threshold=0.25" \
  -F "keypoint_threshold=0.35" \
  -F "image_size=640" \
  -F "include_overlay=true"
```

비디오 보행 분석 예시:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/inference/video \
  -F "file=@/path/to/walk.mp4" \
  -F "analysis_fps=10" \
  -F "max_frames=120" \
  -F "include_trend_data=true"
```

Postman에서 재생 영상을 바로 확인하고 싶으면 두 가지 방법을 쓸 수 있다.

- JSON 응답이 필요하면 `include_playback_video=true`로 요청해서 `playback_video_base64`와 `playback_video_data_url`를 함께 받는다.
- Postman Preview에서 바로 렌더링하려면 `response_format=html`을 추가하면 `<video>` 태그가 포함된 HTML 응답을 받는다.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/inference/video \
  -F "file=@/path/to/walk.mp4" \
  -F "response_format=html"
```

## Colab GPU training

장기 학습은 Colab GPU에서 돌리고, 체크포인트는 Google Drive에 저장하는 구성이 더 현실적이다.

- Notebook: `notebooks/colab_dog_pose_training.ipynb`
- Colab script: `scripts/train_dog_pose_colab.py`

핵심 원칙:

- `RUN_NAME`을 고정하면 Colab 세션이 끊겨도 `last.pt` 기준으로 resume할 수 있다.
- `TARGET_TOTAL_EPOCHS`는 목표 총 epoch 수다. 더 오래 학습하려면 이 값을 더 크게 바꾼 뒤 같은 training cell을 다시 실행하면 된다.
- Colab에서는 로컬 macOS lockfile에 묶이지 않도록 `uv sync` 대신 `uv pip install --system -e .` 경로를 사용한다.
