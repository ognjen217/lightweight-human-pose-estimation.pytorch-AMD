#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_ROOT:-$HOME/lightweight-human-pose-estimation.pytorch-AMD}"
source venv/bin/activate

export PYTHONPATH=/opt/rocm-7.2.4/lib:.
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1

mkdir -p outputs models/fused_postprocess_pruned_batchaware models/merged_pose_fused_pruned_batchaware

python -m py_compile \
  tools/export_batchaware_fused_pruned_postprocess.py \
  tools/compile_merged_pose_batchaware_fused_pruned.py

POST_ONNX=models/fused_postprocess_pruned_batchaware/fused_pruned_batchaware_b2_68x121_to_1080x1920.onnx
MERGED_ONNX=models/merged_pose_fused_pruned_batchaware/pose_fused_pruned_batchaware_b2_1080x1920.onnx

python tools/export_batchaware_fused_pruned_postprocess.py \
  --onnx "$POST_ONNX" \
  --batch-size 2 \
  --in-h 68 \
  --in-w 121 \
  --full-h 1080 \
  --full-w 1920 \
  --topk 20 \
  --limb-topm 20 \
  --threshold 0.1 \
  --nms-radius 6 \
  --nms-impl separable \
  --heatmap-cubic-a -0.75 \
  --paf-cubic-a -0.75 \
  --points-per-limb 8 \
  --min-paf-score 0.05 \
  --success-ratio-thr 0.8 \
  --min-pair-score 0.0 \
  2>&1 | tee outputs/export_batchaware_fused_pruned_b2_onnx.log

python tools/compile_merged_pose_batchaware_fused_pruned.py \
  --pose-onnx models/fp16_refinment1.onnx \
  --post-onnx "$POST_ONNX" \
  --batch-size 2 \
  --output-onnx "$MERGED_ONNX" \
  --merge-only \
  2>&1 | tee outputs/merge_batchaware_pose_fused_pruned_b2_onnx.log

ls -lh "$POST_ONNX" "$MERGED_ONNX"
