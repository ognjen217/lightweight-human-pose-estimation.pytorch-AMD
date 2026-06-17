# Split HIP Smart Stream Integration and MXR2 Profiling Report

## Executive Summary

This report continues from the previous custom HIP kernel direction report. At that point, the split pipeline had been validated in isolated tests, but the multi-camera simulator integration was still listed as a later phase. The work covered here moves the split path into the realtime 10-camera streaming simulator, fixes the transport issues that initially limited the implementation, tunes the batching parameters, and profiles the resulting pipeline with ROCm PFTrace.

The main outcome is that the split pipeline is now runnable inside the realtime multi-camera simulator as a production-like variant:

```text
MXR1: image -> heatmaps + PAFs
HIP smart backend: heatmaps -> top_scores + top_indices
MXR2: full-resolution PAFs + top_scores + top_indices -> pruned limb-pair tensors
CPU tail: final pose assembly
```

The best measured 10-camera configuration so far is:

| Configuration | Aggregate FPS | FPS / camera | Avg postprocess | Avg E2E | P95 E2E |
|---|---:|---:|---:|---:|---:|
| `split_hip_host_smart`, B4, timeout 8 ms | **50.44** | **5.04** | **45.09 ms** | **194.60 ms** | **217.44 ms** |

This is a major improvement over the first integrated run, but the PFTrace profile shows that the remaining bottleneck is not the HIP heatmap stage. The new bottleneck is MXR2, specifically the full-resolution PAF sampling graph that lowers into a very large number of small MIGraphX gather kernels.

The most important profiling result is:

```text
MXR2 gather_kernel.kd launches: 219,100
Captured MXR2/postprocess batch windows: 313
Average gather launches per MXR2 batch: ~700
```

This changes the optimization direction. The heatmap backend is now fast enough for the stream experiment. Further performance work should focus on reducing or fusing the full-resolution PAF gather path inside MXR2, while preserving full-resolution PAF accuracy.

---

## Context From the Previous Report

The previous report established the split architecture and validated the custom HIP heatmap backend in isolated conditions. It showed that:

| Item | Previous status |
|---|---:|
| MXR1 pose adapter | Implemented |
| MXR2 PAF pruning graph | Implemented |
| HIP heatmap TopK backend | Implemented as dense correctness backend |
| Real-frame semantic validation | Passed |
| Valid TopK index mismatches | 0 |
| Split path vs merged baseline | Slower in isolated test |
| Main measured HIP bottlenecks | Dense full-resolution memory path and TopK scan |
| Multi-camera simulator integration | Later phase |

The previous recommended next step was to continue optimizing the heatmap backend with a fused candidate TopK path. After integrating the pipeline into the simulator, the measured priority changed: the stream-level bottleneck is now MXR2 rather than the HIP heatmap stage.

---

## Objective of This Phase

The goal of this phase was to move from isolated split-pipeline validation into a realtime multi-camera simulation and answer three questions:

| Question | Result |
|---|---|
| Can the split MXR1 -> HIP -> MXR2 pipeline run inside the 10-camera simulator? | Yes |
| Can shared-memory transport be made compatible with the split 18-channel heatmap contract? | Yes |
| What is the dominant bottleneck after simulator integration? | MXR2 full-resolution PAF gather/sampling |

The target setup was the same class of realtime stream used for previous multi-camera experiments:

```text
10 simulated cameras
24 FPS source target
latest-frame buffering
soft backpressure
1 inference worker
4 postprocess workers
B4 static MIGraphX models
shared input and shared heatmap/PAF maps
CPU pinning for camera, inference, and postprocess workers
```

---

## Implemented Integration Work

### Stream Variant

A new stream-level split variant was integrated:

```text
--variant split_hip_host_smart
```

This variant runs the simulator as:

```text
camera preprocess
  -> shared input slot
  -> MXR1 pose adapter inference
  -> shared heatmap/PAF map slot
  -> HIP smart heatmap TopK
  -> MXR2 PAF pruning
  -> CPU pose assembly tail
```

The variant exposes runtime parameters for MXR2 and the HIP smart heatmap backend:

| Parameter | Purpose |
|---|---|
| `--split-mxr2` | MXR2 model path |
| `--split-mxr2-batch-size` | Static MXR2 batch size |
| `--split-batch-timeout-ms` | Timeout used to fill MXR2 postprocess batches |
| `--smart-proposals` | Number of heatmap proposals per keypoint type |
| `--smart-local-radius` | Full-resolution local refinement radius |
| `--smart-lowres-nms-radius` | Low-resolution proposal NMS radius |

### CPU Pinning and Runtime Isolation

The successful stream runs used pinned process groups:

| Process group | CPU cores |
|---|---|
| Cameras | `0-9` |
| Inference worker | `10` |
| Postprocess workers | `12-15` |

This made the profile easier to interpret and prevented scheduler noise from hiding the real GPU bottleneck.

### 18-Channel Heatmap Contract Fix

The split pose adapter returns 18 body-keypoint heatmap channels instead of the legacy 19-channel heatmap tensor that includes the background channel. The original shared map allocation assumed 19 channels:

```text
legacy heatmap shared map: H x W x 19
split heatmap shared map: H x W x 18
```

Before the fix, the stream fell back to queue/copy transport because the decoded split heatmaps did not fit the shared map slots. This was visible in the inference worker statistics:

```text
shared_map_misses = 6132
```

After making shared map allocation variant-aware, the same test dropped to:

```text
shared_map_misses = 12
```

This confirmed that the inference-to-postprocess heatmap/PAF handoff was using shared memory almost all the time.

---

## Stream Benchmark Results

All runs below used the split HIP smart path with:

```text
MXR1: pose_adapter_b4_1080x1920.mxr
MXR2: split_paf_pruning_from_topk_b4_68x121_to_1080x1920_k20_m20_p8_min0p05_sr0p8_pam0p75_mp0p0.mxr
B4 static batch size
10 cameras
latest buffering
soft backpressure
1 inference worker
4 postprocess workers
```

### Summary Table

| Run | Timeout | Shared-map state | Aggregate FPS | FPS / camera | Avg post | Avg E2E | P95 E2E | Inference real batch | Shared misses |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Initial split stream | 4 ms | Broken/fallback | 42.26 | 4.23 | 55.48 ms | 199.12 ms | 223.47 ms | 2.51 / 4 | 6132 |
| Shared-map fixed | 4 ms | Fixed | 43.97 | 4.40 | 51.90 ms | 191.14 ms | 213.90 ms | 2.51 / 4 | 12 |
| Timeout-tuned best | 8 ms | Fixed | **50.44** | **5.04** | **45.09 ms** | 194.60 ms | 217.44 ms | **3.27 / 4** | 14 |
| Timeout too high | 12 ms | Fixed | 48.77 | 4.88 | 46.59 ms | 199.84 ms | 225.84 ms | 3.33 / 4 | 17 |

### Interpretation

The shared-memory fix improved the transport path and reduced queue delay, but the larger improvement came from increasing the batching timeout from 4 ms to 8 ms. The 8 ms timeout improved the average real batch size from about 2.51 to 3.27, which reduced per-frame inference and MXR2 cost.

The 12 ms timeout slightly increased the average inference batch size again, from 3.27 to 3.33, but this did not improve the complete stream. Queueing and end-to-end latency increased, and throughput dropped. This makes 8 ms the current best practical timeout for B4.

---

## Stage-Level Timing

The best B4 t8 stream run showed the following postprocess breakdown:

| Stage | Average per frame |
|---|---:|
| HIP smart heatmap stage | 3.92 ms |
| MXR2 | **38.53 ms** |
| CPU assembly | 2.64 ms |
| Total postprocess | **45.09 ms** |

The HIP smart heatmap stage is no longer the main stream bottleneck. MXR2 dominates the postprocess time.

For comparison:

| Run | HIP smart heatmap | MXR2 | CPU assembly | Total postprocess |
|---|---:|---:|---:|---:|
| Initial t4 | 4.65 ms | 48.01 ms | 2.83 ms | 55.48 ms |
| Shared-map fixed t4 | 4.46 ms | 45.27 ms | 2.17 ms | 51.90 ms |
| Best t8 | **3.92 ms** | **38.53 ms** | 2.64 ms | **45.09 ms** |
| t12 | 4.12 ms | 39.73 ms | 2.74 ms | 46.59 ms |

---

## Batch Behavior

The stream results confirm that the split pipeline is highly sensitive to batch fill. MXR2 batch runtime is relatively stable, so underfilled batches are expensive on a per-frame basis.

### Postprocess Batch Distribution

| Real batch size | Initial t4 | Shared-fix t4 | Best t8 | t12 |
|---:|---:|---:|---:|---:|
| 1 | 1211 | 1419 | 844 | 919 |
| 2 | 2172 | 756 | 1334 | 1414 |
| 3 | 315 | 1209 | 1809 | 1014 |
| 4 | 1892 | 2432 | 2684 | 3104 |

The 8 ms timeout was the best tradeoff: it reduced single-frame batches and increased B3/B4 usage without creating too much queueing latency. The 12 ms timeout pushed more frames into B4, but the extra waiting and scheduling effects reduced total throughput.

---

## System-Level Observations

The best stream runs were GPU-bound:

```text
Average GPU utilization: ~97%
Peak GPU utilization: 100%
GPU idle samples: ~1%
VRAM average: ~2.68 GB
```

CPU utilization was controlled by process pinning. Camera workers occupied the first ten cores, inference was pinned to one core, and postprocess workers were pinned to four separate cores. This confirmed that the remaining performance issue is not a lack of CPU utilization; the GPU workload itself is saturated.

---

## ROCm / PFTrace Profiling

A full `rocprofv3` PFTrace run was executed around the whole B4 t8 stream configuration. As expected, the profiler significantly slowed down throughput, so the profiled run is not used as the final performance number. It is used only for kernel-level structure.

### Runtime During Profiling

| Metric | Profiled value |
|---|---:|
| Aggregate output FPS | 27.80 FPS |
| Avg postprocess | 60.98 ms |
| Avg E2E | 221.36 ms |
| P95 E2E | 264.17 ms |
| Inference average real batch | 2.70 / 4 |
| Shared map misses | 16 |

### PFTrace Kernel-Level Finding

The PFTrace showed that MXR2 is dominated by a very large number of small gather kernels.

| Kernel group | Count | Total GPU time | Average time |
|---|---:|---:|---:|
| MXR2 `gather_kernel.kd` | **219,100** | **2974.80 ms** | 13.58 us |
| HIP `final_topk_kernel` | 313 | 1162.73 ms | 3714.79 us |
| HIP `lowres_proposal_kernel` | 313 | 982.69 ms | 3139.58 us |
| MXR2 large geometry/mask elementwise | 5947 | 621.30 ms | 104.47 us |
| MXR2 repeated mul-chain elementwise | 5947 | 349.43 ms | 58.76 us |

The parsed region contained 313 MXR2/postprocess batch windows. Therefore:

```text
219,100 gather launches / 313 captured MXR2 windows = ~700 gather launches per MXR2 batch
```

This is the strongest diagnostic result of this phase.

---

## Updated Bottleneck Analysis

The previous report correctly identified dense full-resolution heatmap memory traversal as a problem in the early HIP backend. After integrating the smart HIP path into the stream, this heatmap stage is no longer the dominant limiter.

The current bottleneck is:

```text
MXR2 full-resolution PAF candidate sampling
  -> many Gather / elementwise / mask kernels
  -> ~700 gather launches per MXR2 batch
  -> high launch overhead and poor fusion
```

Important negative findings:

| Candidate optimization | Status |
|---|---|
| More shared-memory tuning | Mostly exhausted; shared misses are near zero |
| Larger timeout than 8 ms | Not beneficial in B4 test |
| Further CPU pinning | Useful for stability, not the main bottleneck |
| MXR2 TopK optimization first | Not the main target; MXR2 TopK kernel is small in PFTrace |
| Lower-resolution PAF sampling | Rejected because PAF resolution is tied to accuracy |

---

## ONNX and MXR Compilation Implications

The PFTrace result suggests that the MXR2 issue is not only a raw kernel-performance problem. It is also an ONNX graph structure problem. The current MXR2 graph likely expresses full-resolution PAF sampling as many small gather operations instead of a small number of large vectorized gather operations.

Potential ONNX/MXR optimization directions:

| Direction | Expected value | Notes |
|---|---:|---|
| Static-shape B4 compile | Low to medium | Already using static B4, but a cleaner graph may help compiler passes |
| Constant folding and initializer cleanup | Low to medium | Avoid repeated constant/unsqueeze/cast chains |
| Dtype cleanup | Medium | Avoid repeated fp16/fp32/index casts |
| Vectorized PAF sampling gather | High | Replace many small gathers with fewer large gathers |
| Pre-pruned candidate pairs before PAF sampling | High | Keep full-res PAF but reduce candidates entering gather-heavy scoring |
| Custom HIP full-res PAF scoring | Highest potential | Replaces gather-heavy MIGraphX graph with fused kernels |

The key constraint remains:

```text
PAF resolution must remain full-resolution because it is directly connected to accuracy.
```

Therefore, the optimization goal is not to reduce PAF resolution. The goal is to reduce the number of candidate pairs and the number of kernel launches required to score those candidates.

---

## Recommended Next Phase

### Phase H: MXR2 Graph Optimization and Full-Resolution PAF Scoring

The next phase should focus on MXR2 rather than the heatmap backend.

#### H1: MXR2 ONNX Operator Audit

Count and inspect the current MXR2 ONNX graph:

```text
Gather / GatherElements / GatherND
Concat
Cast
Unsqueeze
Reshape
Clip
ReduceSum
TopK
```

Success criterion:

```text
Produce a node-level explanation of why MXR2 lowers into ~700 gather kernels per batch.
```

#### H2: Clean Static MXR2 Rebuild

Generate a cleaned MXR2 ONNX with:

```text
static B4/K20/M20/P8 shapes
constant folding
initializer-based constants
minimal dtype casts
no repeated index cast chains
```

Success criterion:

```text
Same output contract and accuracy behavior, fewer unnecessary kernels in PFTrace.
```

#### H3: Vectorized Full-Resolution PAF Sampling

Restructure ONNX generation so that PAF sampling is represented as fewer larger gather operations:

```text
current: many small per-limb/per-sample/per-channel gather operations
preferred: one or a small number of large gather operations over flattened full-res PAF tensors
```

Success criterion:

```text
Reduce gather kernel launch count substantially while keeping full-resolution PAF sampling.
```

#### H4: Candidate Pre-Pruning Before PAF Sampling

Keep full-resolution PAFs but reduce the number of candidate pairs entering the expensive PAF scoring path.

Proposed first variants:

| Variant | Candidate count before PAF scoring | PAF resolution | Output contract |
|---|---:|---|---|
| Baseline | K20 x K20 = 400 per limb | Full-res | M20 |
| Pre-pruned N96 | 96 per limb | Full-res | M20 |
| Pre-pruned N64 | 64 per limb | Full-res | M20 |
| K16/M20 control | K16 x K16 = 256 per limb | Full-res | M20 |

Success criterion:

```text
Reduce MXR2 runtime without degrading accuracy beyond an acceptable threshold.
```

#### H5: Custom HIP Full-Resolution PAF Scoring

If ONNX/MXR restructuring still lowers to too many kernels, the next serious option is a native HIP PAF scoring backend:

```text
full-res PAF + top_scores + top_indices
  -> compute candidate pair coordinates
  -> sample full-res PAF along limb
  -> accumulate score
  -> apply validity checks
  -> output compact limb pair tensors
```

This would directly replace the gather-heavy MXR2 graph with fused native kernels.

---

## Updated Status Table

| Phase | Description | Previous status | Current status |
|---|---|---:|---:|
| A | Split architecture concept | Done | Done |
| B | MXR1 / MXR2 graph export | Done | Done |
| C | Host-mediated Python external backend | Done | Done |
| D | Native HIP dense correctness backend | Done | Done |
| D.1 | Random-input HIP validation | Done | Done |
| D.2 | Real-frame HIP validation | Done | Done |
| E1 | Segmented TopK optimization | Rejected | Rejected |
| E2 | HIP stage profiling | Done | Done |
| E3 | Fused heatmap candidate TopK path | Next | Lower priority after stream evidence |
| F | GPU-resident / zero-copy handoff | Later | Partially addressed through shared-memory stream transport |
| G | Multi-camera simulator integration | Later | **Done** |
| G.1 | 18-channel split shared-map support | Not started | **Done** |
| G.2 | B4 timeout tuning | Not started | **Done** |
| G.3 | ROCm PFTrace stream profiling | Not started | **Done** |
| H | MXR2 graph / PAF scoring optimization | Not defined | **Next** |

---

## Final Assessment

The split HIP smart direction has moved from isolated correctness validation into a working realtime multi-camera stream variant. The shared-memory incompatibility caused by the 18-channel split heatmap output was fixed, the B4 batching behavior was tuned, and the best current configuration reaches 50.44 aggregate FPS across 10 cameras.

The most important new result is the PFTrace diagnosis. The remaining bottleneck is MXR2, not the HIP heatmap backend. MXR2 spends a large amount of time in a gather-heavy full-resolution PAF sampling graph, producing roughly 700 gather kernel launches per captured MXR2 batch.

The optimization direction should therefore shift from heatmap TopK to MXR2 graph restructuring. Since PAF resolution must remain full-resolution for accuracy, the next phase should reduce the number of candidate pairs and the number of gather launches, not the spatial resolution of PAF sampling.

The recommended next work item is:

```text
Create and profile a full-resolution MXR2 pre-pruned/vectorized candidate sampling variant.
```

If this does not reduce the gather-kernel storm enough, the follow-up should be a custom HIP full-resolution PAF scoring backend.
