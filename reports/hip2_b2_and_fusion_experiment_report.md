# HIP2 PAF Backend, B2 Latency Tuning, and HIP Fusion Experiments

**Continuation report after:** `Split HIP Smart Stream Integration and MXR2 Profiling Report`  
**Project:** AMD ROCm / MIGraphX realtime multi-camera pose-estimation pipeline  
**Date:** 2026-06-16  
**Primary target:** 10-camera realtime stream simulation, 24 FPS sources, latest-frame buffering

---

## Executive Summary

The previous report ended with a clear bottleneck diagnosis: the split stream was running successfully, but the MXR2 PAF-pruning graph dominated postprocessing because its full-resolution PAF sampling lowered into a large number of small MIGraphX gather kernels. The recommended next direction was to replace or restructure the MXR2 PAF scoring path while preserving full-resolution PAF sampling.

This phase executed that recommendation. A custom HIP2 PAF-pruning backend was implemented, validated against MXR2, integrated into the realtime simulator, and then tuned across batching and scheduling configurations. The main outcome is that the custom HIP2 backend successfully replaced the MXR2 bottleneck and produced a new best realtime 10-camera configuration.

The best confirmed configuration is:

```text
MXR1:          pose_adapter_b2_1080x1920.mxr
Variant:       split_hip2_host_smart
PAF backend:   custom HIP2 host backend
Batch size:    B2 for MXR1 and HIP2 postprocess
Timeout:       2 ms for MXR1 batching and split postprocess batching
Post workers:  2
Backpressure:  soft
Duration:      130 s
```

Final confirmed result:

| Metric | Value |
|---|---:|
| Processed frames | **9,982** |
| Wall time | **131.48 s** |
| Aggregate FPS | **75.92** |
| FPS / camera | **7.59** |
| Avg E2E latency | **88.46 ms** |
| P95 E2E latency | **109.42 ms** |
| Avg postprocess | **14.53 ms** |
| P95 postprocess | **17.15 ms** |
| Avg inference-to-post queue | **1.51 ms** |
| Inference real batch size | **2.00 / 2** |
| Shared map misses | **0** |

Compared to the previous report’s best MXR2-based split stream result:

| Configuration | Aggregate FPS | Avg post | Avg E2E | P95 E2E |
|---|---:|---:|---:|---:|
| Previous best: MXR2 split stream, B4/t8 | 50.44 | 45.09 ms | 194.60 ms | 217.44 ms |
| New best: HIP2 split stream, B2/t2 | **75.92** | **14.53 ms** | **88.46 ms** | **109.42 ms** |

This is a major improvement: aggregate FPS increased by about **50.5%**, average E2E latency decreased by about **54.5%**, and P95 E2E latency decreased by about **49.7%** relative to the previous MXR2-based best stream configuration.

Several additional experiments were also performed:

1. **B4 HIP2 stream tuning** showed that custom HIP2 immediately improved stream throughput and latency compared to MXR2.
2. **B2 MXR1 compilation and stream testing** showed that smaller static batches are much better for realtime latency than B4, while preserving throughput.
3. **Batch-level handoff** reduced inference-to-post queueing but worsened end-to-end latency and throughput, proving that the queue itself was not the main problem.
4. **HIP1+HIP2 fused backends** were implemented in two forms, including a low-level fused V2, but both regressed performance. The current evidence shows that host-level HIP fusion creates a longer GPU critical section and worsens overlap with MIGraphX inference.

The recommended production/reporting configuration is therefore the unfused **B2/t2 `split_hip2_host_smart`** pipeline.

---

## 1. Context From the Previous Report

The previous report validated the split stream path:

```text
MXR1: image -> heatmaps + PAFs
HIP smart heatmap backend: heatmaps -> top_scores + top_indices
MXR2: PAFs + top_scores + top_indices -> pruned limb-pair tensors
CPU tail: final pose assembly
```

The best previous 10-camera result used the MXR2 PAF-pruning graph:

| Metric | Previous best value |
|---|---:|
| Variant | `split_hip_host_smart` |
| Batch size | B4 |
| Timeout | 8 ms |
| Aggregate FPS | 50.44 |
| FPS / camera | 5.04 |
| Avg postprocess | 45.09 ms |
| Avg E2E | 194.60 ms |
| P95 E2E | 217.44 ms |

PFTrace showed that MXR2 was dominated by a gather-heavy PAF sampling path:

```text
MXR2 gather_kernel.kd launches: 219,100
Captured MXR2/postprocess batch windows: 313
Average gather launches per MXR2 batch: ~700
```

The previous recommendation was to move away from the gather-heavy MXR2 graph and implement a custom full-resolution PAF scoring path in HIP. This phase follows exactly that recommendation.

---

## 2. HIP2 PAF-Pruning Backend

### 2.1 Objective

The goal was to replace the MXR2 PAF-pruning graph with a native HIP backend while keeping the same output contract:

```text
inputs:
  pafs        [B, 38, 68, 121]
  top_scores  [B, 18, K]
  top_indices [B, 18, K]

outputs:
  limb_top_pair_a_idx [B, 19, M]
  limb_top_pair_b_idx [B, 19, M]
  limb_top_pair_score [B, 19, M]
  limb_top_pair_valid [B, 19, M]
```

The backend preserves the same semantics as MXR2:

```text
for each limb:
  decode candidate keypoint coordinates from top_indices
  evaluate K x K candidate pairs
  sample full-resolution PAF field along each limb
  compute affinity and success ratio
  keep TopM valid limb pairs
```

### 2.2 Implemented Components

The following implementation work was added:

| Area | Description |
|---|---|
| C++ HIP backend | Native PAF pair scoring and pruning kernels |
| Python wrapper | ctypes-based host wrapper for HIP2 backend |
| Build script | Shell/CMake build flow for the HIP shared library |
| Validation tool | MXR2 vs HIP2 comparison over deterministic inputs |
| Stream integration | `split_hip2_host_smart` realtime simulator path |

The HIP2 path replaced the MXR2 PAF-pruning stage while keeping the HIP smart heatmap stage and CPU assembly tail.

### 2.3 Isolated Validation Against MXR2

HIP2 was validated against the existing MXR2 graph using deterministic comparison runs.

| Metric | Result |
|---|---:|
| Runs | 3 |
| Batch size | 4 |
| Strict output pass | Yes |
| Semantic output pass | Yes |
| Exact output match | Yes |
| Max absolute output difference | 0 |

Average isolated timings:

| Stage | Avg time |
|---|---:|
| MXR1 | 25.75 ms |
| HIP smart heatmap TopK | 9.34 ms |
| MXR2 PAF pruning | 40.92 ms |
| HIP2 PAF pruning | **8.08 ms** |

This gave an average MXR2-to-HIP2 speedup of about:

```text
40.92 ms / 8.08 ms = 5.06x
```

The steady-state runs after the first warmup-like run were even better, around 6x faster. More importantly, the output contract was preserved exactly.

---

## 3. HIP2 Stream Integration

### 3.1 Stream Pipeline

The stream pipeline after HIP2 integration became:

```text
camera preprocess
  -> shared input slot
  -> MXR1 split pose adapter
  -> shared heatmap/PAF map slot
  -> HIP smart heatmap TopK
  -> HIP2 PAF pruning
  -> CPU pose assembly
  -> output row / optional rendering
```

The active stream variant became:

```text
--variant split_hip2_host_smart
--split-paf-backend hip_host
```

Compared to the previous MXR2 stream, this removes the gather-heavy MIGraphX MXR2 graph from the postprocess stage.

### 3.2 B4 Baseline After HIP2 Integration

The first major stream integration target used the existing B4 MXR1 model and B4 HIP2 postprocess batches.

| Configuration | FPS | FPS/cam | Avg post | P95 post | Avg E2E | P95 E2E | Queue infer→post |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4/t8/P4/soft | 65.14 | 6.51 | 21.36 ms | 48.53 ms | 135.87 ms | 167.94 ms | 8.15 ms |

This was already a strong improvement over the previous MXR2-based best result:

| Metric | MXR2 previous best | HIP2 B4/t8/P4 | Improvement |
|---|---:|---:|---:|
| Aggregate FPS | 50.44 | 65.14 | +29.1% |
| Avg postprocess | 45.09 ms | 21.36 ms | -52.6% |
| Avg E2E | 194.60 ms | 135.87 ms | -30.2% |
| P95 E2E | 217.44 ms | 167.94 ms | -22.8% |

The custom HIP2 backend therefore solved the original MXR2 bottleneck at stream level.

---

## 4. B4 Worker and Timeout Tuning

After HIP2 was integrated, the next question was how many postprocess workers and what batching timeout should be used.

### 4.1 Postprocess Worker Count at B4/t8

| Config | FPS | Avg post | P95 post | Queue infer→post | Avg E2E | P95 E2E |
|---|---:|---:|---:|---:|---:|---:|
| B4/t8/P1/soft | 66.61 | **11.87 ms** | **15.30 ms** | 21.88 ms | 143.74 ms | 166.47 ms |
| B4/t8/P2/soft | **72.38** | 12.30 ms | 28.11 ms | **4.89 ms** | **124.12 ms** | **154.07 ms** |
| B4/t8/P4/soft | 65.14 | 21.36 ms | 48.53 ms | 8.15 ms | 135.87 ms | 167.94 ms |

Interpretation:

- P1 gave the cleanest per-worker postprocess timings, but created too much postprocess queueing.
- P4 fragmented postprocess batches and made P95 postprocess much worse.
- P2 was the best B4 scheduling tradeoff.

### 4.2 B4 Timeout and Backpressure Tuning

Additional B4/P2 tests were run with shorter timeout and backpressure-off variants.

| Config | FPS | Avg post | P95 post | Queue infer→post | Avg E2E | P95 E2E |
|---|---:|---:|---:|---:|---:|---:|
| B4/t8/P2/soft | 72.38 | 12.30 ms | 28.11 ms | 4.89 ms | 124.12 ms | 154.07 ms |
| B4/t4/P2/soft | 74.47 | 12.14 ms | 23.13 ms | **3.40 ms** | **120.53 ms** | **149.78 ms** |
| B4/t6/P2/backpressure-off | **75.55** | **11.74 ms** | **23.06 ms** | 4.25 ms | 121.13 ms | 152.41 ms |

The best B4 latency configuration was:

```text
B4/t4/P2/soft
```

The best B4 throughput configuration was:

```text
B4/t6/P2/backpressure-off
```

However, B4 still had relatively high E2E latency because the real batch wall time remained large.

---

## 5. B2 MXR1 Compilation and Latency Tuning

### 5.1 Motivation

After B4 tuning, the pipeline was much faster than the previous MXR2 version, but E2E latency remained around 120 ms. The hypothesis was that B4 was a good throughput batch size but not an ideal realtime latency batch size.

The next test was therefore to compile and evaluate a B2 MXR1 split pose adapter:

```text
models/split_pose_adapter/pose_adapter_b2_1080x1920.mxr
```

The exhaustive-tune compile path was attempted but was too slow for practical iteration. The B2 model was therefore compiled without exhaustive tuning and used for stream tests.

### 5.2 B2/t4/P2/soft

| Metric | Value |
|---|---:|
| Processed frames | 4,612 |
| Aggregate FPS | 75.06 |
| FPS / camera | 7.51 |
| Avg postprocess | 14.50 ms |
| P95 postprocess | 17.17 ms |
| Queue infer→post | 1.64 ms |
| Avg E2E | 88.57 ms |
| P95 E2E | 109.95 ms |
| Inference real batch size | 2.00 / 2 |
| Shared map misses | 0 |

B2 immediately produced a large E2E latency reduction while preserving throughput.

### 5.3 B2/t2/P2/soft

Reducing both MXR1 and split-postprocess timeouts from 4 ms to 2 ms gave a small additional improvement.

| Metric | B2/t4/P2/soft | B2/t2/P2/soft |
|---|---:|---:|
| Aggregate FPS | 75.06 | **75.66** |
| Avg E2E | 88.57 ms | **88.22 ms** |
| P95 E2E | 109.95 ms | **109.87 ms** |
| Avg postprocess | **14.50 ms** | 14.55 ms |
| Queue infer→post | 1.64 ms | **1.61 ms** |

The difference was small but consistently positive. B2/t2 became the preferred latency configuration.

### 5.4 Final 130s B2 Confirmation Run

The B2/t2/P2/soft configuration was then validated with a longer 130-second run.

| Metric | Final B2/t2/P2/soft 130s result |
|---|---:|
| Processed frames | **9,982** |
| Wall time | **131.48 s** |
| Aggregate FPS | **75.92** |
| FPS / camera | **7.59** |
| Avg preprocess | 5.33 ms |
| Avg queue pre→infer | 22.31 ms |
| Avg inference | 6.45 ms/frame |
| Avg decode | 0.60 ms/frame |
| Avg queue infer→post | **1.51 ms** |
| Avg postprocess | **14.53 ms** |
| P95 postprocess | **17.15 ms** |
| Avg E2E | **88.46 ms** |
| P95 E2E | **109.42 ms** |
| Inference batches | 4,991 |
| Avg real batch size | **2.00 / 2** |
| Shared map misses | **0** |
| Replaced before post | **0** |
| Stale pre-batch records | **0** |

This confirmed that B2/t2 was stable and not a short-run artifact.

### 5.5 B2 vs B4 Summary

| Config | FPS | Avg E2E | P95 E2E | Avg post | Queue infer→post |
|---|---:|---:|---:|---:|---:|
| B4/t4/P2/soft | 74.47 | 120.53 ms | 149.78 ms | **12.14 ms** | 3.40 ms |
| B4/t6/P2/backpressure-off | 75.55 | 121.13 ms | 152.41 ms | **11.74 ms** | 4.25 ms |
| B2/t2/P2/soft, 60s | 75.66 | 88.22 ms | 109.87 ms | 14.55 ms | 1.61 ms |
| B2/t2/P2/soft, 130s | **75.92** | **88.46 ms** | **109.42 ms** | 14.53 ms | **1.51 ms** |

B4 has slightly lower amortized postprocess cost per frame, but B2 has much better realtime latency because it reduces batch wall time and keeps the postprocess queue almost empty.

The key conclusion is:

```text
B4 is a throughput-oriented batch size.
B2 is the realtime latency sweet spot for the current 10-camera stream.
```

---

## 6. Batch-Level Handoff Experiment

### 6.1 Motivation

The B4 results still showed some inference-to-postprocess queueing and postprocess re-batching. A new experimental mode was added to preserve MXR1 batches across the inference-to-postprocess boundary.

New mode:

```text
split_hip2_batch_handoff
```

Instead of:

```text
MXR1 B4 -> split into per-frame queue items -> postprocess rebuilds a batch
```

it used:

```text
MXR1 B4 -> one batch packet -> postprocess consumes the same batch
```

### 6.2 Result

| Metric | B4/t4/P2 batch handoff |
|---|---:|
| Processed frames | 3,585 |
| Aggregate FPS | 57.68 |
| Avg postprocess | 11.40 ms |
| P95 postprocess | 20.03 ms |
| Queue infer→post | **1.49 ms** |
| Avg E2E | 140.49 ms |
| P95 E2E | 189.54 ms |

### 6.3 Interpretation

The batch-handoff experiment successfully reduced inference-to-postprocess queueing to about 1.5 ms, but total throughput and E2E latency became much worse.

This negative result is important because it shows that the original queue between inference and postprocess was not the primary bottleneck. Reducing that queue alone did not improve the pipeline. Instead, the batch-handoff path changed scheduling in a way that reduced throughput and increased end-to-end latency.

Conclusion:

```text
Batch handoff is not a useful optimization for the current pipeline.
The best path remains normal per-frame handoff with B2/t2 scheduling.
```

---

## 7. HIP1 + HIP2 Fusion Experiments

After finding the B2 latency sweet spot, another direction was tested: fusing the HIP smart heatmap stage and the HIP2 PAF-pruning stage into a single larger HIP backend.

The motivation was to remove this intermediate host boundary:

```text
HIP1 heatmap TopK -> top_scores/top_indices on host -> HIP2 PAF pruning
```

The intended fused path was:

```text
heatmaps + PAFs
  -> HIP smart heatmap TopK
  -> top_scores/top_indices remain on GPU
  -> HIP2 PAF pruning
  -> final small tensors copied to host
```

### 7.1 Fused V1: Host-Level Fusion

The first implementation added a single shared-library call around HIP smart TopK and the existing HIP2 backend.

New backend:

```text
--split-paf-backend hip_fused_host
```

Result:

| Metric | Fused V1 B2/t2/P2/soft, 20s |
|---|---:|
| Processed frames | 1,156 |
| Aggregate FPS | 53.88 |
| FPS / camera | 5.39 |
| Avg postprocess | 31.21 ms |
| P95 postprocess | 34.62 ms |
| Queue infer→post | 62.74 ms |
| Avg E2E | 187.92 ms |
| P95 E2E | 230.24 ms |

This was a major regression. The likely cause was that V1 still called the existing HIP2 entrypoint internally, and that entrypoint performed its own allocation and synchronization.

### 7.2 Fused V2: Low-Level Fusion

To test whether the V1 regression was caused by the internal HIP2 call, a second fused implementation was built:

```text
cpp/split_hip_fused/split_hip_fused_v2.cpp
```

V2 directly launches the following kernels inside one stream:

```text
lowres_proposal_kernel
refine_proposals_kernel
final_topk_kernel
fused_score_pairs_kernel
fused_prune_pairs_kernel
final D2H copies
```

This removed the internal `paf_prune_hip_run(...)` call from the fused path.

Result:

| Metric | Fused V1 | Fused V2 |
|---|---:|---:|
| Processed frames | 1,156 | 1,178 |
| Aggregate FPS | 53.88 | **54.85** |
| Avg postprocess | 31.21 ms | **30.76 ms** |
| P95 postprocess | 34.62 ms | **34.10 ms** |
| Queue infer→post | 62.74 ms | **60.74 ms** |
| Avg E2E | 187.92 ms | **183.81 ms** |
| P95 E2E | 230.24 ms | **223.47 ms** |

V2 improved slightly over V1, but remained far worse than the unfused B2/t2 baseline.

### 7.3 Fusion Experiment Conclusion

The fusion experiments produced a useful negative result:

```text
HIP1+HIP2 host-level fusion is not the right optimization boundary.
```

Even after low-level fusion removed the internal HIP2 call and synchronization, postprocess remained around 30 ms and inference-to-postprocess queueing stayed around 60 ms. This shows that the main issue is not only the intermediate TopK host transfer.

The likely explanation is GPU scheduling and overlap:

```text
Unfused path:
  shorter HIP calls
  better interleaving with MIGraphX inference
  lower postprocess queue

Fused path:
  one longer GPU critical section
  worse overlap with MIGraphX inference
  postprocess falls behind inference
  inference-to-post queue grows
```

This is also visible in inference timing. In the fused runs, average inference increased to about 8.9-9.0 ms/frame, compared to about 6.45 ms/frame in the best unfused B2 run.

Conclusion:

```text
Do not use the fused HIP backend for final benchmarks.
Keep it in the repository as an experimental branch/path, but do not continue tuning it unless the next design also removes per-call allocation and introduces persistent device buffers.
```

---

## 8. Consolidated Results

### 8.1 Main Stream Configurations

| Phase / Config | FPS | Avg post | Queue infer→post | Avg E2E | P95 E2E | Status |
|---|---:|---:|---:|---:|---:|---|
| Previous MXR2 B4/t8 | 50.44 | 45.09 ms | n/a | 194.60 ms | 217.44 ms | Superseded |
| HIP2 B4/t8/P4 | 65.14 | 21.36 ms | 8.15 ms | 135.87 ms | 167.94 ms | Improved baseline |
| HIP2 B4/t8/P2 | 72.38 | 12.30 ms | 4.89 ms | 124.12 ms | 154.07 ms | Better B4 scheduling |
| HIP2 B4/t4/P2 | 74.47 | **12.14 ms** | 3.40 ms | 120.53 ms | 149.78 ms | Best B4 latency |
| HIP2 B4/t6/P2/backpressure-off | 75.55 | **11.74 ms** | 4.25 ms | 121.13 ms | 152.41 ms | Best B4 throughput |
| HIP2 B2/t4/P2 | 75.06 | 14.50 ms | 1.64 ms | 88.57 ms | 109.95 ms | B2 latency win |
| HIP2 B2/t2/P2, 60s | 75.66 | 14.55 ms | 1.61 ms | 88.22 ms | 109.87 ms | Best short run |
| HIP2 B2/t2/P2, 130s | **75.92** | 14.53 ms | **1.51 ms** | **88.46 ms** | **109.42 ms** | **Final best** |
| Batch handoff B4/t4/P2 | 57.68 | 11.40 ms | 1.49 ms | 140.49 ms | 189.54 ms | Rejected |
| Fused V1 B2/t2/P2 | 53.88 | 31.21 ms | 62.74 ms | 187.92 ms | 230.24 ms | Rejected |
| Fused V2 B2/t2/P2 | 54.85 | 30.76 ms | 60.74 ms | 183.81 ms | 223.47 ms | Rejected |

### 8.2 Best Current Command

The current best command is:

```bash
python simulate_camera_stream.py \
  --model models/split_pose_adapter/pose_adapter_b2_1080x1920.mxr \
  --variant split_hip2_host_smart \
  --migraphx-batch-size 2 \
  --migraphx-batch-timeout-ms 2 \
  --num-cameras 10 \
  --frames-per-camera 0 \
  --duration-s 130 \
  --realtime \
  --camera-fps 24 \
  --buffer-mode latest \
  --backpressure-mode soft \
  --infer-workers 1 \
  --post-workers 2 \
  --shared-input-slots 10 \
  --shared-map-slots 16 \
  --shared-dtype float32 \
  --shared-input-dtype float32 \
  --split-mxr2-batch-size 2 \
  --split-batch-timeout-ms 2 \
  --split-paf-backend hip_host \
  --smart-proposals 32 \
  --smart-local-radius 4 \
  --smart-lowres-nms-radius 1 \
  --max-keypoints 20 \
  --threshold 0.1 \
  --pin-cpus \
  --pin-camera-base 0 \
  --pin-inference-base 10 \
  --pin-post-base 12 \
  --detailed-csv outputs/split_hip2_best_b2_t2_p2_soft_130s_detailed.csv \
  --summary-json outputs/split_hip2_best_b2_t2_p2_soft_130s_summary.json
```

---

## 9. Technical Interpretation

### 9.1 What Worked

#### Custom HIP2 PAF pruning

This was the most important successful optimization. It directly addressed the MXR2 gather-kernel bottleneck from the previous report and preserved output correctness.

Result:

```text
MXR2 PAF pruning: ~40.92 ms isolated
HIP2 PAF pruning: ~8.08 ms isolated
```

At stream level, replacing MXR2 with HIP2 moved the pipeline from about 50 FPS and 195 ms E2E to about 76 FPS and 88 ms E2E after B2 tuning.

#### B2 static MXR1 model

B2 was the second major success. It did not massively improve throughput, but it dramatically improved realtime latency.

Reason:

```text
B4 reduces amortized per-frame cost.
B2 reduces real batch wall time.
Realtime E2E latency is dominated by wall-time and scheduling, not only amortized cost.
```

#### P2 postprocess worker count

Two postprocess workers consistently provided the best balance. One worker kept cleaner batches but created queueing. Four workers fragmented batches and increased jitter. Two workers kept queueing low while preserving stable postprocess behavior.

### 9.2 What Did Not Work

#### Batch-level handoff

Batch-level handoff reduced one queue metric but worsened the full system. It showed that the queue itself was not the bottleneck.

#### HIP1+HIP2 fusion

Both fused V1 and fused V2 regressed performance. The current host-mediated fused backend made postprocess too long and reduced overlap with MIGraphX inference.

This is a useful finding: optimization should not blindly minimize call count. In this pipeline, shorter independent GPU work chunks appear to schedule better than one larger fused postprocess call.

---

## 10. Updated Bottleneck Analysis

After HIP2 and B2 tuning, the previous MXR2 bottleneck is no longer dominant. The current best pipeline has:

```text
Avg queue pre->infer:   ~22.31 ms
Avg inference:           ~6.45 ms/frame amortized
Avg decode:              ~0.60 ms/frame amortized
Avg queue infer->post:   ~1.51 ms
Avg postprocess:        ~14.53 ms
Avg E2E:                ~88.46 ms
```

The remaining latency is not a single obvious postprocess bottleneck. It is a combination of:

1. Natural frame age from 24 FPS latest-frame scheduling.
2. MXR1 B2 inference wall time.
3. HIP smart heatmap + HIP2 postprocess wall time.
4. Multiprocess scheduling and timestamp accounting.
5. CPU assembly tail.

The `queue_pre_to_infer` value around 22 ms should not be interpreted as a pure inefficiency. With 24 FPS inputs, the frame period is about 41.67 ms, and the latest available frame is naturally around half a frame period old on average.

The `queue_infer_to_post` value around 1.5 ms in the final best run is already low. That queue is no longer a meaningful bottleneck.

---

## 11. Recommended Next Steps

### 11.1 Lock the Best Result

The B2/t2/P2/soft result should be treated as the final result for this optimization phase.

Recommended report wording:

```text
The MXR2 gather-heavy PAF pruning graph was replaced with a custom HIP2 PAF-pruning backend. After integrating HIP2 into the realtime 10-camera simulator and tuning batch size and timeout, the best configuration used a B2 split MXR1 model with 2 ms batching timeouts and two postprocess workers. Over a 130-second run, the pipeline processed 9,982 frames at 75.92 aggregate FPS with 88.46 ms average E2E latency and 109.42 ms P95 E2E latency. This reduced average E2E latency by approximately 54.5% relative to the previous MXR2-based split-stream result.
```

### 11.2 Do Not Continue Host-Level HIP Fusion

The fused HIP backend should not be used for final results. It should remain as an experimental implementation and negative-result reference.

If fusion is revisited, the next version must not only fuse kernels. It must also address:

```text
persistent device buffers
persistent HIP stream per postprocess worker
no hipMalloc/hipFree per batch
no stream create/destroy per batch
device-resident handoff from MXR1 outputs if possible
```

### 11.3 Most Valuable Future Direction: Device-Resident Pipeline

The most promising next phase is not HIP1+HIP2 host fusion. It is reducing or removing the CPU/shared-map boundary between MXR1 and HIP postprocess.

Target architecture:

```text
MXR1 output device buffer
  -> HIP smart heatmap directly from device
  -> HIP2 PAF pruning directly from device
  -> copy only compact final tensors to CPU
  -> CPU pose assembly
```

This would remove the large heatmap/PAF host-mediated transfer path and avoid repeatedly uploading MXR1 outputs back to HIP from CPU memory.

Potential tasks:

| Task | Goal |
|---|---|
| Inspect MIGraphX output buffer ownership | Determine whether output tensors can remain device-resident |
| Add device-pointer HIP entrypoints | Allow HIP1/HIP2 to accept device pointers directly |
| Persistent postprocess context | Reuse HIP buffers and stream across batches |
| Profile with PFTrace | Verify whether H2D/D2H and launch gaps decrease |
| Validate accuracy/semantic identity | Ensure final output remains equivalent to current HIP2 backend |

### 11.4 Secondary Future Direction: CPU Assembly Optimization

After device-resident postprocess work, the CPU assembly tail may become a more visible bottleneck. It is not currently the dominant limiter, but it is a candidate for later optimization if E2E latency is pushed below the current 88 ms range.

---

## 12. Final Status Table

| Work item | Status | Outcome |
|---|---|---|
| Replace MXR2 with custom HIP PAF pruning | Done | Successful |
| Validate HIP2 against MXR2 | Done | Exact output match |
| Integrate HIP2 into stream simulator | Done | Successful |
| Tune B4 workers/timeouts | Done | Best B4: B4/t4/P2/soft |
| Compile and test B2 MXR1 | Done | Successful |
| Confirm B2/t2 over 130s | Done | Final best result |
| Batch-level handoff | Done | Rejected |
| HIP1+HIP2 fused V1 | Done | Rejected |
| HIP1+HIP2 fused V2 low-level | Done | Rejected |
| Device-resident MXR1->HIP handoff | Not started | Recommended next phase |
| Persistent HIP postprocess context | Not started | Recommended only if fusion is revisited |

---

## Final Assessment

This phase successfully moved the project beyond the MXR2 bottleneck identified in the previous report. The custom HIP2 PAF-pruning backend replaced a gather-heavy MIGraphX graph with a native HIP implementation, matched MXR2 outputs exactly, and reduced isolated PAF-pruning time by about 5x.

The largest realtime gain came from combining HIP2 with a B2 MXR1 split model. B4 remained competitive for throughput, but B2 was substantially better for realtime latency. The final B2/t2/P2/soft run reached 75.92 aggregate FPS across 10 cameras with 88.46 ms average E2E latency and 109.42 ms P95 E2E latency.

The negative experiments are also useful. Batch-level handoff and HIP1+HIP2 fusion both reduced specific local overheads but worsened complete system behavior. This shows that the current pipeline is sensitive to GPU scheduling and overlap, and that one larger fused host-level GPU call can be worse than two shorter calls.

The current best configuration should therefore be locked as:

```text
split_hip2_host_smart + B2 + 2 ms batching + 2 postprocess workers + soft backpressure
```

The next meaningful optimization step is a device-resident split pipeline, where MXR1 outputs are consumed by HIP postprocess without returning large heatmap/PAF tensors through CPU/shared memory.
