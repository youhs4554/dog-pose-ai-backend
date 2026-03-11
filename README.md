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

## Colab GPU training

장기 학습은 Colab GPU에서 돌리고, 체크포인트는 Google Drive에 저장하는 구성이 더 현실적이다.

- Notebook: `notebooks/colab_dog_pose_training.ipynb`
- Colab script: `scripts/train_dog_pose_colab.py`

핵심 원칙:

- `RUN_NAME`을 고정하면 Colab 세션이 끊겨도 `last.pt` 기준으로 resume할 수 있다.
- `TARGET_TOTAL_EPOCHS`는 목표 총 epoch 수다. 더 오래 학습하려면 이 값을 더 크게 바꾼 뒤 같은 training cell을 다시 실행하면 된다.
- Colab에서는 로컬 macOS lockfile에 묶이지 않도록 `uv sync` 대신 `uv pip install --system -e .` 경로를 사용한다.
