from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO, settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or resume dog pose on Colab GPU.")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Warm-start checkpoint for a new run.")
    parser.add_argument("--data", default="dog-pose.yaml", help="Ultralytics dataset config.")
    parser.add_argument("--epochs", type=int, default=100, help="Target total epochs for this run.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument(
        "--batch",
        default="16",
        help="Batch size as an integer or CUDA memory fraction between 0 and 1, e.g. 16 or 0.70.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--device", default="0", help="CUDA device id, e.g. 0.")
    parser.add_argument("--project", default="runs/pose", help="Output directory root.")
    parser.add_argument("--name", default="dog-pose-colab", help="Run name under project/.")
    parser.add_argument("--save-period", type=int, default=5, help="Save a checkpoint every N epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        help="Early-stop patience. Defaults to max(epochs, 1000) to avoid stopping before the target epoch.",
    )
    parser.add_argument(
        "--cache",
        default="disk",
        help="Ultralytics cache mode: false, ram, or disk. Default keeps repeated Colab runs responsive.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from project/name/weights/last.pt if it exists.")
    parser.add_argument("--resume-path", help="Explicit checkpoint path to resume from.")
    parser.add_argument("--datasets-dir", help="Optional Ultralytics datasets_dir override, e.g. /content/datasets.")
    parser.add_argument("--no-val", action="store_true", help="Skip validation during training.")
    parser.add_argument("--no-plots", action="store_true", help="Disable train/val plots.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing run directory.")
    return parser.parse_args()


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Select a GPU runtime in Colab before training.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")


def resolve_batch(raw_value: str) -> int | float:
    value = str(raw_value).strip()
    if not value:
        raise SystemExit("--batch cannot be empty.")
    if "." in value:
        parsed = float(value)
        if not 0 < parsed <= 1:
            raise SystemExit("Fractional --batch must be between 0 and 1.")
        return parsed
    parsed = int(value)
    if parsed < 1:
        raise SystemExit("Integer --batch must be positive.")
    return parsed


def resolve_cache(raw_value: str) -> bool | str:
    value = str(raw_value).strip().lower()
    if value in {"false", "0", "off", "none"}:
        return False
    if value in {"true", "1", "on"}:
        return True
    if value in {"ram", "disk"}:
        return value
    raise SystemExit("--cache must be one of false, ram, disk, or true.")


def configure_ultralytics(datasets_dir: str | None) -> None:
    if not datasets_dir:
        return
    dataset_path = Path(datasets_dir).expanduser().resolve()
    dataset_path.mkdir(parents=True, exist_ok=True)
    settings.update({"datasets_dir": str(dataset_path)})
    print(f"Ultralytics datasets_dir -> {dataset_path}")


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    candidates: list[Path] = []
    if args.resume_path:
        candidates.append(Path(args.resume_path).expanduser())
    candidates.append(Path(args.project) / args.name / "weights" / "last.pt")

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


def save_summary(run_dir: Path, batch_size: int | float, resumed_from: str | None) -> Path:
    weights_dir = run_dir / "weights"
    payload = {
        "run_dir": str(run_dir.resolve()),
        "best": str((weights_dir / "best.pt").resolve()),
        "last": str((weights_dir / "last.pt").resolve()),
        "batch": batch_size,
        "resumed_from": resumed_from,
    }
    summary_path = run_dir.parent / "latest_run.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def train_kwargs(args: argparse.Namespace, batch_size: int | float, cache_mode: bool | str, patience: int) -> dict[str, Any]:
    return {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": batch_size,
        "workers": args.workers,
        "device": args.device,
        "save_period": args.save_period,
        "patience": patience,
        "cache": cache_mode,
        "val": not args.no_val,
        "plots": not args.no_plots,
        "project": str(Path(args.project).expanduser().resolve()),
        "name": args.name,
        "exist_ok": args.exist_ok,
    }


def main() -> None:
    args = parse_args()
    ensure_cuda_available()

    batch_size = resolve_batch(args.batch)
    cache_mode = resolve_cache(args.cache)
    patience = args.patience if args.patience is not None else max(args.epochs, 1000)
    configure_ultralytics(args.datasets_dir)

    Path(args.project).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    common_args = train_kwargs(args, batch_size, cache_mode, patience)
    resume_checkpoint = resolve_resume_checkpoint(args) if args.resume else None

    if resume_checkpoint:
        print(f"Resuming from {resume_checkpoint} to reach {args.epochs} total epochs.")
        model = YOLO(str(resume_checkpoint))
        results = model.train(resume=True, **common_args)
        resumed_from = str(resume_checkpoint)
    else:
        print(f"Starting a new run from {args.model} for {args.epochs} epochs.")
        model = YOLO(args.model)
        results = model.train(data=args.data, **common_args)
        resumed_from = None

    run_dir = Path(results.save_dir)
    summary_path = save_summary(run_dir, batch_size, resumed_from)
    print(f"Training finished. Run directory: {run_dir.resolve()}")
    print(f"Best checkpoint: {(run_dir / 'weights' / 'best.pt').resolve()}")
    print(f"Latest summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
