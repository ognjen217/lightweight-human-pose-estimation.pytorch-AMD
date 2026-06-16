#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import accuracy_validation as base


def _preparse_smart_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--fused-pruned-heatmap-mode",
        choices=["full-res", "smart-full-res"],
        default="full-res",
    )
    parser.add_argument("--smart-proposals", type=int, default=64)
    parser.add_argument("--smart-local-radius", type=int, default=8)
    parser.add_argument("--smart-lowres-nms-radius", type=int, default=1)
    return parser.parse_known_args(argv)


def parse_args() -> argparse.Namespace:
    smart_args, remaining = _preparse_smart_args(sys.argv[1:])
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + remaining
        args = base.parse_args()
    finally:
        sys.argv = old_argv

    args.fused_pruned_heatmap_mode = smart_args.fused_pruned_heatmap_mode
    args.smart_proposals = int(smart_args.smart_proposals)
    args.smart_local_radius = int(smart_args.smart_local_radius)
    args.smart_lowres_nms_radius = int(smart_args.smart_lowres_nms_radius)
    return args


if __name__ == "__main__":
    base.validate_accuracy(parse_args())
