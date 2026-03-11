from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from ultralytics.data.utils import check_det_dataset
from ultralytics import YOLO
from ultralytics.utils.torch_utils import autocast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dog pose model on Apple MPS.")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Pretrained pose checkpoint.")
    parser.add_argument("--data", default="dog-pose.yaml", help="Ultralytics dataset config.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument(
        "--batch",
        default="auto",
        help="Training batch size. Use an integer or 'auto' to probe the largest MPS-safe batch.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--device", default="mps", help="Training device. Fixed to mps by default.")
    parser.add_argument("--project", default="runs/pose", help="Ultralytics project directory.")
    parser.add_argument("--name", default="dog-pose-mps-1epoch-maxbatch", help="Ultralytics run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing run directory.")
    parser.add_argument("--fraction", type=float, help="Fraction of the training split to use.")
    parser.add_argument(
        "--train-steps",
        type=int,
        help="Approximate number of optimizer steps per epoch to target by shrinking the training split.",
    )
    parser.add_argument("--skip-val", action="store_true", help="Skip validation to shorten smoke runs.")
    parser.add_argument("--no-plots", action="store_true", help="Disable train/val plots to save time.")
    parser.add_argument("--probe-start", type=int, default=4, help="Starting batch size for MPS probing.")
    parser.add_argument("--probe-max", type=int, default=128, help="Upper bound for MPS batch probing.")
    parser.add_argument(
        "--probe-fraction",
        type=float,
        default=0.02,
        help="Dataset fraction used during MPS batch probing.",
    )
    parser.add_argument(
        "--probe-timeout",
        type=int,
        default=120,
        help="Maximum seconds allowed for a single MPS probe candidate.",
    )
    parser.add_argument("--probe-once-batch", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def ensure_mps_available() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine. This script requires device='mps'.")


def save_summary(run_dir: Path, batch_size: int, train_fraction: float) -> None:
    summary_path = Path("runs/pose/latest_run.json")
    weights_dir = run_dir / "weights"
    payload = {
        "run_dir": str(run_dir.resolve()),
        "best": str((weights_dir / "best.pt").resolve()),
        "last": str((weights_dir / "last.pt").resolve()),
        "batch_size": batch_size,
        "train_fraction": train_fraction,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_mps_memory() -> None:
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def is_mps_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def run_probe_once(args: argparse.Namespace, batch_size: int) -> bool:
    clear_mps_memory()
    with tempfile.TemporaryDirectory(prefix=f"dog-pose-probe-b{batch_size}-") as probe_dir:
        try:
            base_model = YOLO(args.model)
            overrides = {
                "model": args.model,
                "data": args.data,
                "task": "pose",
                "imgsz": args.imgsz,
                "batch": batch_size,
                "epochs": 1,
                "workers": min(args.workers, 2),
                "device": args.device,
                "project": probe_dir,
                "name": "probe",
                "exist_ok": True,
                "amp": False,
                "save": False,
                "plots": False,
                "fraction": args.probe_fraction,
                "warmup_epochs": 0,
                "verbose": False,
            }
            trainer = base_model._smart_load("trainer")(overrides=overrides, _callbacks=base_model.callbacks)
            trainer.model = trainer.get_model(
                weights=base_model.model if base_model.ckpt else None,
                cfg=base_model.model.yaml,
            )
            trainer._setup_train()
            trainer._model_train()
            trainer.optimizer.zero_grad()
            train_iter = iter(trainer.train_loader)
            for _ in range(2):
                batch = next(train_iter)
                batch = trainer.preprocess_batch(batch)
                with autocast(trainer.amp):
                    loss, trainer.loss_items = trainer.model(batch)
                    loss = loss.sum()
                trainer.scaler.scale(loss).backward()
            del batch, loss, trainer, base_model
            return True
        except Exception as exc:
            if is_mps_oom(exc):
                print(f"MPS probe failed at batch={batch_size}: {exc}")
                return False
            raise
        finally:
            clear_mps_memory()


def try_probe_batch(args: argparse.Namespace, batch_size: int) -> bool:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-once-batch",
        str(batch_size),
        "--model",
        args.model,
        "--data",
        args.data,
        "--imgsz",
        str(args.imgsz),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
        "--probe-fraction",
        str(args.probe_fraction),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=args.probe_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"MPS probe timed out at batch={batch_size} after {args.probe_timeout}s")
        return False

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    return result.returncode == 0


def resolve_batch_size(args: argparse.Namespace) -> int:
    if str(args.batch).lower() != "auto":
        batch_size = int(args.batch)
        if batch_size < 1:
            raise SystemExit("Batch size must be a positive integer or 'auto'.")
        return batch_size

    low = 0
    high = max(args.probe_start, 1)
    while high <= args.probe_max and try_probe_batch(args, high):
        low = high
        high *= 2

    if low == 0:
        raise SystemExit(
            f"Even batch={args.probe_start} exceeded available MPS memory. Lower --imgsz or start batch smaller."
        )

    high = min(high - 1, args.probe_max)
    left, right = low, high
    while left < right:
        mid = (left + right + 1) // 2
        if try_probe_batch(args, mid):
            left = mid
        else:
            right = mid - 1

    print(f"Selected maximum MPS-safe batch size: {left}")
    return left


def count_training_images(train_path: str) -> int:
    train_dir = Path(train_path)
    image_suffixes = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
    if train_dir.is_dir():
        return sum(1 for path in train_dir.rglob("*") if path.suffix.lower() in image_suffixes)
    if train_dir.is_file():
        return sum(1 for line in train_dir.read_text(encoding="utf-8").splitlines() if line.strip())
    raise FileNotFoundError(f"Could not find training split at {train_path}")


def resolve_train_fraction(args: argparse.Namespace, batch_size: int) -> float:
    if args.fraction is not None and args.train_steps is not None:
        raise SystemExit("Use either --fraction or --train-steps, not both.")

    if args.fraction is not None:
        if not 0 < args.fraction <= 1:
            raise SystemExit("--fraction must be between 0 and 1.")
        return args.fraction

    if args.train_steps is None:
        return 1.0

    if args.train_steps < 1:
        raise SystemExit("--train-steps must be a positive integer.")

    data_config = check_det_dataset(args.data)
    train_image_count = count_training_images(data_config["train"])
    target_image_count = min(train_image_count, args.train_steps * batch_size)
    train_fraction = target_image_count / train_image_count
    effective_steps = math.ceil(target_image_count / batch_size)
    print(
        "Resolved smoke-run subset: "
        f"{target_image_count}/{train_image_count} training images "
        f"({train_fraction:.6f} fraction), about {effective_steps} steps/epoch."
    )
    return train_fraction


def main() -> None:
    args = parse_args()
    if args.device != "mps":
        raise SystemExit("This project is configured for Apple Silicon training. Use --device mps.")

    ensure_mps_available()

    if args.probe_once_batch:
        success = run_probe_once(args, args.probe_once_batch)
        raise SystemExit(0 if success else 2)

    project_dir = str(Path(args.project).resolve())
    batch_size = resolve_batch_size(args)
    train_fraction = resolve_train_fraction(args, batch_size)

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch_size,
        workers=args.workers,
        device=args.device,
        amp=False,
        fraction=train_fraction,
        val=not args.skip_val,
        plots=not args.no_plots,
        project=project_dir,
        name=args.name,
        exist_ok=args.exist_ok,
    )

    run_dir = Path(results.save_dir)
    save_summary(run_dir, batch_size, train_fraction)
    print(f"Training finished. Run directory: {run_dir.resolve()}")
    print(f"Best checkpoint: {(run_dir / 'weights' / 'best.pt').resolve()}")


if __name__ == "__main__":
    main()
