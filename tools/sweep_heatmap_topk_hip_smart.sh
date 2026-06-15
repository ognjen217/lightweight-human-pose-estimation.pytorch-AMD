#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MXR1="${MXR1:-models/split_pose_adapter/pose_adapter_b4_1080x1920.mxr}"
VIDEO="${VIDEO:-cctv_1280x720_24fps_3.mp4}"
OUT_DIR="${OUT_DIR:-outputs/split_pipeline_compare/smart_sweep}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-1}"
FRAME_INDEX="${FRAME_INDEX:-0}"
RUN_FRAME_STRIDE="${RUN_FRAME_STRIDE:-24}"
BATCH_FRAME_STRIDE="${BATCH_FRAME_STRIDE:-1}"
THREADS_LIST="${THREADS_LIST:-32 64 128}"
PROPOSALS_LIST="${PROPOSALS_LIST:-32 64 96}"
LOCAL_RADIUS_LIST="${LOCAL_RADIUS_LIST:-4 8 12}"
LOWRES_NMS_RADIUS="${LOWRES_NMS_RADIUS:-1}"

mkdir -p "${OUT_DIR}"

echo "[sweep] root=${ROOT_DIR}"
echo "[sweep] out_dir=${OUT_DIR}"
echo "[sweep] threads=${THREADS_LIST}"
echo "[sweep] proposals=${PROPOSALS_LIST}"
echo "[sweep] local_radius=${LOCAL_RADIUS_LIST}"

# First sweep reducer thread count with the current best smart params.
for threads in ${THREADS_LIST}; do
  echo "[sweep] build smart threads=${threads}"
  HIP_TOPK_SMART_THREADS="${threads}" bash tools/build_heatmap_topk_hip.sh
  tag="threads${threads}_sp64_lr8_lnms${LOWRES_NMS_RADIUS}"
  python tools/profile_heatmap_topk_hip_smart_real_frame.py \
    --mxr1 "${MXR1}" \
    --video "${VIDEO}" \
    --batch-size "${BATCH_SIZE}" \
    --runs "${RUNS}" \
    --warmup "${WARMUP}" \
    --frame-index "${FRAME_INDEX}" \
    --run-frame-stride "${RUN_FRAME_STRIDE}" \
    --batch-frame-stride "${BATCH_FRAME_STRIDE}" \
    --smart-proposals 64 \
    --smart-local-radius 8 \
    --smart-lowres-nms-radius "${LOWRES_NMS_RADIUS}" \
    --json "${OUT_DIR}/profile_${tag}.json" \
    --markdown "${OUT_DIR}/profile_${tag}.md"
done

# Then keep the best known thread count by default and sweep proposal/radius pairs.
BEST_THREADS="${BEST_THREADS:-64}"
echo "[sweep] build best smart threads=${BEST_THREADS} for proposal/local-radius sweep"
HIP_TOPK_SMART_THREADS="${BEST_THREADS}" bash tools/build_heatmap_topk_hip.sh

for proposals in ${PROPOSALS_LIST}; do
  for local_radius in ${LOCAL_RADIUS_LIST}; do
    tag="threads${BEST_THREADS}_sp${proposals}_lr${local_radius}_lnms${LOWRES_NMS_RADIUS}"
    echo "[sweep] profile ${tag}"
    python tools/profile_heatmap_topk_hip_smart_real_frame.py \
      --mxr1 "${MXR1}" \
      --video "${VIDEO}" \
      --batch-size "${BATCH_SIZE}" \
      --runs "${RUNS}" \
      --warmup "${WARMUP}" \
      --frame-index "${FRAME_INDEX}" \
      --run-frame-stride "${RUN_FRAME_STRIDE}" \
      --batch-frame-stride "${BATCH_FRAME_STRIDE}" \
      --smart-proposals "${proposals}" \
      --smart-local-radius "${local_radius}" \
      --smart-lowres-nms-radius "${LOWRES_NMS_RADIUS}" \
      --json "${OUT_DIR}/profile_${tag}.json" \
      --markdown "${OUT_DIR}/profile_${tag}.md"
  done
done

python - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("profile_*.json")):
    data = json.loads(path.read_text())
    avg = data.get("profile_ms_avg", {})
    ctx = data.get("context_avg", {})
    rows.append({
        "file": path.name,
        "backend_total_ms": avg.get("total_ms"),
        "device_total_ms": avg.get("device_total_ms"),
        "proposal_ms": avg.get("resize_ms"),
        "refine_ms": avg.get("vertical_ms"),
        "final_topk_ms": avg.get("topk_ms"),
        "valid_topk": ctx.get("valid_topk_count"),
    })
rows.sort(key=lambda r: (float("inf") if r["backend_total_ms"] is None else r["backend_total_ms"]))
summary_json = out_dir / "smart_sweep_summary.json"
summary_json.write_text(json.dumps(rows, indent=2))
summary_md = out_dir / "smart_sweep_summary.md"
lines = [
    "# Smart HIP sweep summary", "",
    "| rank | file | total ms | device ms | proposal ms | refine ms | final topk ms | valid topk |",
    "|---:|---|---:|---:|---:|---:|---:|---:|",
]
for i, r in enumerate(rows, 1):
    lines.append(
        f"| {i} | `{r['file']}` | {r['backend_total_ms']:.4f} | {r['device_total_ms']:.4f} | "
        f"{r['proposal_ms']:.4f} | {r['refine_ms']:.4f} | {r['final_topk_ms']:.4f} | {r['valid_topk']:.2f} |"
    )
summary_md.write_text("\n".join(lines) + "\n")
print(f"[sweep] wrote {summary_json}")
print(f"[sweep] wrote {summary_md}")
PY

echo "[sweep] done"
