#!/usr/bin/env python3

from pathlib import Path
import argparse
import migraphx

from modules.migraphx_fused_postprocess_pruned_compiler import (
    append_pruning_tail,
    pruned_head_name,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-onnx", required=True)
    ap.add_argument("--output-dir", default="models/fused_postprocess_pruned_cache")

    ap.add_argument("--in-h", type=int, default=68)
    ap.add_argument("--in-w", type=int, default=121)
    ap.add_argument("--full-h", type=int, default=1080)
    ap.add_argument("--full-w", type=int, default=1920)

    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--limb-topm", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--nms-radius", type=int, default=6)
    ap.add_argument("--nms-impl", default="separable")
    ap.add_argument("--heatmap-cubic-a", type=float, default=-0.75)
    ap.add_argument("--paf-cubic-a", type=float, default=-0.75)
    ap.add_argument("--points-per-limb", type=int, default=8)
    ap.add_argument("--min-paf-score", type=float, default=0.05)
    ap.add_argument("--success-ratio-thr", type=float, default=0.8)
    ap.add_argument("--min-pair-score", type=float, default=0.0)

    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-onnx", action="store_true")
    ap.add_argument("--exhaustive-tune", action="store_true")
    args = ap.parse_args()

    base_onnx = Path(args.base_onnx)
    if not base_onnx.exists():
        raise FileNotFoundError(base_onnx)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = pruned_head_name(
        args.in_h,
        args.in_w,
        args.full_h,
        args.full_w,
        topk=args.topk,
        limb_topm=args.limb_topm,
        threshold=args.threshold,
        nms_radius=args.nms_radius,
        nms_impl=args.nms_impl,
        heatmap_cubic_a=args.heatmap_cubic_a,
        points_per_limb=args.points_per_limb,
        min_paf_score=args.min_paf_score,
        success_ratio_thr=args.success_ratio_thr,
        paf_cubic_a=args.paf_cubic_a,
        min_pair_score=args.min_pair_score,
    )

    pruned_onnx = out_dir / f"{name}.onnx"
    pruned_mxr = out_dir / f"{name}.mxr"

    if pruned_mxr.exists() and not args.force:
        print(f"[pruned-direct] exists, skipping: {pruned_mxr}")
        return

    print("=" * 120)
    print("[pruned-direct] base ONNX:")
    print(f"  {base_onnx}")
    print("[pruned-direct] output ONNX:")
    print(f"  {pruned_onnx}")
    print("[pruned-direct] output MXR:")
    print(f"  {pruned_mxr}")
    print("=" * 120)

    print(f"[pruned-direct] appending TopM pruning tail: K={args.topk}, M={args.limb_topm}")
    append_pruning_tail(
        base_onnx,
        pruned_onnx,
        topk=args.topk,
        limb_topm=args.limb_topm,
        min_pair_score=args.min_pair_score,
    )

    print(f"[pruned-direct] compiling MIGraphX GPU target: {pruned_onnx.name} -> {pruned_mxr.name}")
    program = migraphx.parse_onnx(str(pruned_onnx))
    program.compile(
        migraphx.get_target("gpu"),
        exhaustive_tune=bool(args.exhaustive_tune),
    )
    migraphx.save(program, str(pruned_mxr))

    if not args.keep_onnx:
        # Možeš ovo ostaviti, ali ja bih za sada čuvao ONNX radi debug-a.
        pass

    print(f"[pruned-direct] saved: {pruned_mxr}")


if __name__ == "__main__":
    main()
