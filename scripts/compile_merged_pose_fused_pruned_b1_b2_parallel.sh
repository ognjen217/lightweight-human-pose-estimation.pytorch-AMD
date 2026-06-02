#!/usr/bin/env bash
set -euo pipefail

# Compile B=1 and B=2 merged pose+fused-pruned MXR in parallel.
#
# Run from repo root:
#   chmod +x scripts/compile_merged_pose_fused_pruned_b1_b2_parallel.sh
#   ./scripts/compile_merged_pose_fused_pruned_b1_b2_parallel.sh
#
# Override defaults:
#   POSE_ONNX=models/fp16_refinment1.onnx \
#   FUSED_PRUNED_ONNX=models/fused_postprocess_pruned_cache/your_pruned.onnx \
#   OUT_DIR=models/merged_pose_fused_pruned \
#   ./scripts/compile_merged_pose_fused_pruned_b1_b2_parallel.sh

POSE_ONNX="${POSE_ONNX:-models/fp16_refinment1.onnx}"
FUSED_PRUNED_ONNX="${FUSED_PRUNED_ONNX:-models/fused_postprocess_pruned_cache/fused_cubic_topk_fullres_paf_pruned_68x121_to_1080x1920_k20_m20_thr0p1_r6_separable_ham0p75_p8_min0p05_sr0p8_pam0p75_mp0p0.onnx}"
OUT_DIR="${OUT_DIR:-models/merged_pose_fused_pruned}"

mkdir -p "$OUT_DIR" outputs

echo "POSE_ONNX=$POSE_ONNX"
echo "FUSED_PRUNED_ONNX=$FUSED_PRUNED_ONNX"
echo "OUT_DIR=$OUT_DIR"

PYTHONPATH=. python tools/inspect_onnx_io.py "$POSE_ONNX" "$FUSED_PRUNED_ONNX" | tee "$OUT_DIR/inspect_pose_and_fused_pruned.txt"

compile_one() {
  local B="$1"
  local LOG="$OUT_DIR/compile_b${B}.log"
  echo "[start] compile B=$B -> $LOG"
  PYTHONPATH=. python tools/compile_merged_pose_fused_pruned.py \
    --pose-onnx "$POSE_ONNX" \
    --fused-pruned-onnx "$FUSED_PRUNED_ONNX" \
    --batch-size "$B" \
    --output-dir "$OUT_DIR" \
    --output-prefix pose_fused_pruned \
    2>&1 | tee "$LOG"
  echo "[done] compile B=$B"
}

compile_one 1 &
PID1=$!

compile_one 2 &
PID2=$!

wait "$PID1"
wait "$PID2"

echo
echo "Compiled files:"
find "$OUT_DIR" -maxdepth 1 \( -name "*.mxr" -o -name "*.onnx" -o -name "*.json" -o -name "*.log" \) -print | sort
