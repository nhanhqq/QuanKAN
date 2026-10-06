"""Command-line interface for reproducible paper-aligned LOSO runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..losses import LossCoefficients
from .loso import run_loso

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "resources" / "default.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train QuanKAN with source-only validation and strict LOSO testing."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset", choices=("seed", "seediv", "seedv", "seedvii"), required=True
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--locs-path")
    parser.add_argument("--sessions", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--quantum-device")
    parser.add_argument("--noise-std", type=float)
    parser.add_argument("--gain-jitter", type=float)
    parser.add_argument("--channel-drop", type=float)
    parser.add_argument("--time-mask", type=float)
    parser.add_argument("--band-drop", type=float)
    parser.add_argument("--augmentation-probability", type=float)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--frontend-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--minimum-lr", type=float)
    parser.add_argument("--spatial-tokens", type=int)
    parser.add_argument("--embedding-dim", type=int)
    return parser


def load_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as file:
        config = json.load(file)

    for name, value in config.items():
        if (
            name not in ("loss_coefficients", "loss_weight_initialization")
            and getattr(args, name, None) is None
        ):
            setattr(args, name, value)
    if args.sessions is None:
        args.sessions = [1, 2, 3]
    initial = config.get("loss_weight_initialization", config.get("loss_coefficients"))
    args.loss_weight_initialization = LossCoefficients(**initial)
    if args.locs_path == "channel_62_pos.locs":
        args.locs_path = str(DEFAULT_CONFIG.parent / "channel_62_pos.locs")
    from .contract import validate_settings

    validate_settings(args)
    return args


def main() -> None:
    args = load_args()
    summary = run_loso(args)
    print(json.dumps(summary, indent=2))
