#!/usr/bin/env python3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modules.migraphx_fused_postprocess_pruned_compiler import (
    append_pruning_tail,
    pruned_head_name,
)

base_onnx = Path(
    "models/fused_postprocess_cache/"
    "fused_cubic_topk_fullres_paf_68x121_to_1080x1920_k20_thr0p1_r6_separable_ham0p75_p8_min0p05_sr0p8_pam0p75.onnx"
)

out_dir = Path("models/fused_postprocess_pruned_cache")
out_dir.mkdir(parents=True, exist_ok=True)

name = pruned_head_name(
    68,
    121,
    1080,
    1920,
    topk=20,
    limb_topm=20,
    threshold=0.1,
    nms_radius=6,
    nms_impl="separable",
    heatmap_cubic_a=-0.75,
    points_per_limb=8,
    min_paf_score=0.05,
    success_ratio_thr=0.8,
    paf_cubic_a=-0.75,
    min_pair_score=0.0,
)

pruned_onnx = out_dir / f"{name}.onnx"

if not base_onnx.exists():
    raise FileNotFoundError(base_onnx)

print(f"[make-pruned-onnx] base:   {base_onnx}")
print(f"[make-pruned-onnx] output: {pruned_onnx}")

append_pruning_tail(
    base_onnx,
    pruned_onnx,
    topk=20,
    limb_topm=20,
    min_pair_score=0.0,
)

print(f"[make-pruned-onnx] saved: {pruned_onnx}")
