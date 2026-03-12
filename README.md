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

## API Specification

Base URL:

```text
http://127.0.0.1:8000
```

문서 엔드포인트:

- Swagger UI: `/docs`
- OpenAPI JSON: `/openapi.json`

공통 사항:

- `POST /api/v1/inference/image`와 `POST /api/v1/inference/video`는 `multipart/form-data` 요청이다.
- 파일 필드 이름은 둘 다 `file`이다.
- 에러 응답은 공통적으로 `{"error": {"code": "...", "message": "...", "details": {...}}}` 형식을 사용한다.

### Endpoint Summary

| Method | Path | Description |
|------|------|-------------|
| `GET` | `/` | API 메타데이터 반환 |
| `GET` | `/api/v1/health` | 서버 상태와 기본 모델 경로 확인 |
| `GET` | `/api/v1/model` | 현재 해석되는 모델 경로 확인 |
| `POST` | `/api/v1/inference/image` | 단일 이미지 pose 추론 |
| `POST` | `/api/v1/inference/video` | 비디오 보행 분석 |

### `GET /`

설명:

- API 이름, 버전, 문서 경로를 반환한다.

응답 예시:

```json
{
  "name": "Dog Pose FastAPI",
  "version": "0.1.0",
  "docs": "/docs",
  "openapi": "/openapi.json"
}
```

### `GET /api/v1/health`

설명:

- 서버가 정상 동작하는지 확인한다.
- 현재 기본 체크포인트 경로와 파일 존재 여부를 함께 반환한다.

응답 예시:

```json
{
  "status": "ok",
  "app_name": "dog-pose-fastapi",
  "version": "0.1.0",
  "default_model_path": "/absolute/path/to/runs/pose/dog-pose-colab-longrun/weights/best.pt",
  "default_model_exists": true
}
```

### `GET /api/v1/model`

Query parameter:

- `model_path`: 선택. 특정 체크포인트 경로를 넘기면 그 경로를 기준으로 확인한다.

응답 예시:

```json
{
  "requested_model_path": null,
  "resolved_model_path": "/absolute/path/to/runs/pose/dog-pose-colab-longrun/weights/best.pt",
  "exists": true,
  "is_default": true
}
```

### `POST /api/v1/inference/image`

Form fields:

- `file`: 필수. `jpg`, `jpeg`, `png`, `webp` 이미지 파일
- `model_path`: 선택. 체크포인트 경로 override
- `conf_threshold`: 선택. detection confidence threshold, 기본 `0.25`
- `keypoint_threshold`: 선택. skeleton keypoint threshold, 기본 `0.35`
- `image_size`: 선택. 추론 입력 크기, 기본 `640`
- `include_overlay`: 선택. 오버레이 이미지 포함 여부, 기본 `true`

응답 주요 필드:

- `model_path`: 실제 추론에 사용한 모델 경로
- `image_width`, `image_height`: 입력 이미지 크기
- `detections`: 감지된 개체 목록
- `overlay_image_base64`: 오버레이 JPEG base64
- `overlay_media_type`: 오버레이 MIME type

`detections[]` 주요 필드:

- `instance`: detection index
- `confidence`: detection confidence
- `box`: bounding box (`x1`, `y1`, `x2`, `y2`)
- `visible_keypoints`: threshold 이상 keypoint 수
- `keypoints`: 관절별 좌표와 confidence

응답 예시:

```json
{
  "model_path": "/absolute/path/to/runs/pose/dog-pose-colab-longrun/weights/best.pt",
  "image_width": 1280,
  "image_height": 720,
  "confidence_threshold": 0.25,
  "keypoint_threshold": 0.35,
  "image_size": 640,
  "detections": [
    {
      "instance": 0,
      "confidence": 0.94,
      "box": {
        "x1": 120.4,
        "y1": 80.2,
        "x2": 980.1,
        "y2": 640.7
      },
      "visible_keypoints": 24,
      "keypoints": [
        {
          "joint": "nose",
          "x": 544.8,
          "y": 161.2,
          "confidence": 0.98,
          "visible": true
        }
      ]
    }
  ],
  "overlay_image_base64": "....",
  "overlay_media_type": "image/jpeg"
}
```

### `POST /api/v1/inference/video`

Form fields:

- `file`: 필수. 보행 비디오 파일
- `model_path`: 선택. 체크포인트 경로 override
- `conf_threshold`: 선택. detection confidence threshold, 기본 `0.25`
- `keypoint_threshold`: 선택. skeleton keypoint threshold, 기본 `0.35`
- `image_size`: 선택. 추론 입력 크기, 기본 `640`
- `analysis_fps`: 선택. 분석 FPS, `0`이면 원본 FPS 사용
- `max_frames`: 선택. 최대 분석 프레임 수, `0`이면 전체
- `include_frame_previews`: 선택. 프레임별 오버레이 JPEG 포함 여부
- `include_playback_video`: 선택. 재생용 분석 비디오 base64 포함 여부
- `include_trend_data`: 선택. 안정화된 시계열 데이터 포함 여부
- `response_format`: 선택. `json` 또는 `html`, 기본 `json`

응답 주요 필드:

- `source_fps`, `analyzed_fps`, `total_frames`, `total_duration_sec`, `frame_step`
- `analyzed_frames`: 프레임별 metric, keypoint record, optional preview image
- `trend`: 안정화된 시계열 metric 배열
- `gait_summary`, `gait_stats`, `gait_note`, `gait_status`
- `mpl_summary`, `mpl_stats`, `mpl_note`, `mpl_status`, `mpl_primary_side`
- `playback_video_base64`: 분석 playback mp4 base64
- `playback_video_media_type`: 보통 `video/mp4`
- `playback_video_data_url`: 브라우저나 클라이언트에서 바로 사용할 수 있는 `data:` URL

JSON 응답 예시:

```json
{
  "model_path": "/absolute/path/to/runs/pose/dog-pose-colab-longrun/weights/best.pt",
  "source_fps": 29.97,
  "analyzed_fps": 10.0,
  "total_frames": 180,
  "total_duration_sec": 6.01,
  "frame_step": 3,
  "analyzed_frames": [
    {
      "frame_index": 0,
      "time_sec": 0.0,
      "metrics": {
        "detected": 1,
        "body_axis_angle_deg": 4.5
      },
      "records": []
    }
  ],
  "trend": [
    {
      "time_sec": 0.0,
      "body_axis_angle_deg": 4.5
    }
  ],
  "gait_note": "stable gait",
  "gait_status": "정상 패턴에 가까움",
  "mpl_note": "low concern",
  "mpl_status": "정상 패턴에 가까움",
  "playback_video_base64": "....",
  "playback_video_media_type": "video/mp4",
  "playback_video_data_url": "data:video/mp4;base64,...."
}
```

HTML 응답:

- `response_format=html`로 요청하면 `<video>` 태그가 포함된 HTML 문서를 반환한다.
- 간단한 미리보기 페이지가 필요할 때 사용할 수 있다.

### Error Response

에러 예시:

```json
{
  "error": {
    "code": "model_not_found",
    "message": "Model checkpoint was not found: /absolute/path/to/best.pt",
    "details": {
      "resolved_model_path": "/absolute/path/to/best.pt"
    }
  }
}
```

## Colab GPU training

장기 학습은 Colab GPU에서 돌리고, 체크포인트는 Google Drive에 저장하는 구성이 더 현실적이다.

- Notebook: `notebooks/colab_dog_pose_training.ipynb`
- Colab script: `scripts/train_dog_pose_colab.py`

핵심 원칙:

- `RUN_NAME`을 고정하면 Colab 세션이 끊겨도 `last.pt` 기준으로 resume할 수 있다.
- `TARGET_TOTAL_EPOCHS`는 목표 총 epoch 수다. 더 오래 학습하려면 이 값을 더 크게 바꾼 뒤 같은 training cell을 다시 실행하면 된다.
- Colab에서는 로컬 macOS lockfile에 묶이지 않도록 `uv sync` 대신 `uv pip install --system -e .` 경로를 사용한다.
