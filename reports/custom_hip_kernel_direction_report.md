# Custom HIP Kernel Direction Report

## Executive Summary

This work evaluated a split pose-estimation pipeline in which the original merged MIGraphX graph is decomposed into two MXR graphs with an external native HIP heatmap TopK backend between them:

```text
MXR1: input image -> heatmaps + PAFs
HIP backend: heatmaps -> top_scores + top_indices
MXR2: PAFs + top_scores + top_indices -> pruned limb-pair tensors
```

The main result is that the custom HIP backend is now semantically validated on real CCTV frames with valid heatmap peaks. The current implementation is still a dense correctness backend, not a final optimized path. It is slower than the merged MXR baseline, but it gives a stable and measurable foundation for the next optimization phase.

The most important diagnostic result is that CPU-GPU transfers are not the bottleneck in the current host-mediated test. The measured bottlenecks are the dense full-resolution TopK scan and the full-resolution resize/NMS memory path.

---

## Objective

The objective of this direction was to determine whether the heatmap postprocessing part of the pose-estimation pipeline can be extracted from the MIGraphX graph and replaced by a custom AMD GPU implementation, while preserving the output contract required by the downstream PAF pruning graph.

The long-term target is:

```text
MXR1 GPU output -> HIP heatmap backend -> MXR2 GPU input
```

with no unnecessary CPU roundtrip and with reduced full-resolution memory traffic.

---

## Pipeline Architecture

| Component | Role | Status |
|---|---|---:|
| Merged baseline MXR | Reference graph containing pose model + fused-pruned postprocessing | Stable baseline |
| MXR1 / pose adapter | Exports heatmaps and PAFs from the pose model | Implemented |
| External heatmap backend | Converts heatmaps into `top_scores` and `top_indices` | Implemented, HIP dense path |
| MXR2 / PAF pruning graph | Consumes PAFs and TopK tensors, outputs pruned limb-pair tensors | Implemented |
| CPU final assembly | Final lightweight pose assembly tail | Existing path |

Current split pipeline:

```text
input image
  -> MXR1
  -> heatmaps + PAFs
  -> HIP host-mediated heatmap TopK
  -> top_scores + top_indices
  -> MXR2
  -> limb_top_pair_* outputs
```

---

## Implemented Artifacts

| Area | Files / Tools | Purpose |
|---|---|---|
| Split graph export | `tools/export_split_pose_adapter.py` | Build MXR1: image to heatmaps/PAFs |
| Split graph export | `tools/export_split_paf_pruning_from_topk.py` | Build MXR2: PAFs + TopK to limb pairs |
| Python external backend | `modules/external_heatmap_topk.py` | Backend registry and Python/CUDA/CPU style prototype path |
| HIP C ABI | `cpp/heatmap_topk_hip/heatmap_topk_hip.h` | Stable C interface for native backend |
| HIP backend | `cpp/heatmap_topk_hip/heatmap_topk_hip.cpp` | Dense correctness implementation |
| HIP profiling backend | `cpp/heatmap_topk_hip/heatmap_topk_hip_profile.cpp` | Per-stage HIP event instrumentation |
| Python ctypes loader | `modules/external_heatmap_topk_hip.py` | Loads `.so`, exposes host and profiling calls |
| Build helper | `tools/build_heatmap_topk_hip.sh` | Reproducible `hipcc` build path |
| Smoke test | `tools/smoke_heatmap_topk_hip.py` | Validates library load, kernel launch, basic output |
| Split comparison | `tools/compare_split_pipeline_hip_host.py` | Random input compare against merged MXR |
| Real-frame comparison | `tools/compare_split_pipeline_hip_host_real_frame.py` | Real CCTV frame compare against merged MXR |
| Stage profiler | `tools/profile_heatmap_topk_hip_real_frame.py` | Internal HIP stage profiling on real frames |

---

## Validation Results

### Random Input Validation

The random-input test established that the external HIP backend obeys the expected tensor contract and that invalid TopK slots can differ without affecting downstream results.

| Metric | Result |
|---|---:|
| Runs | 3 |
| Passed all runs | True |
| Semantic passed all runs | True |
| Strict passed all runs | False |
| Valid TopK count | 0 |
| Valid index mismatches | 0 |
| Invalid index mismatches | 1368 per run |
| Final MXR2 outputs exact | True |

Interpretation: useful contract test, but not sufficient by itself because random input produced no valid heatmap peaks above the threshold.

### Real-Frame Validation

The real-frame test used CCTV frames and the same preprocessing convention as the video validation path.

| Run | Valid TopK | Valid index mismatch | Invalid index mismatch | Semantic match | Final limb outputs exact |
|---:|---:|---:|---:|:---:|:---:|
| 0 | 341 | 0 | 1027 | True | True |
| 1 | 304 | 0 | 1064 | True | True |
| 2 | 380 | 0 | 988 | True | True |
| **Average / summary** | **341.67 avg** | **0** | — | **True** | **True** |

This confirms that the HIP heatmap backend is semantically compatible with the merged MIGraphX heatmap branch on real frames with valid pose peaks.

---

## Performance Results

### Stable Dense HIP Backend vs Merged Baseline

| Stage | Average time, ms |
|---|---:|
| Merged MXR baseline | 96.69 |
| MXR1 | 20.30 |
| HIP external heatmap | 69.12 |
| MXR2 | 44.78 |
| Split total | 134.20 |

Current split path is slower than the merged baseline:

| Comparison | Time, ms | Relative |
|---|---:|---:|
| Merged baseline | 96.69 | 1.00x |
| Split + HIP dense backend | 134.20 | 1.39x slower |

Interpretation: the custom HIP path is correctness-ready, but not performance-ready.

---

## E-Phase Experiments

### E1: Segmented TopK Reduction

The first E-phase attempt tried to parallelize the final TopK scan by splitting each full-resolution heatmap plane into multiple segments and reducing partial TopK results.

| Result | Value |
|---|---:|
| Correctness | Passed |
| Semantic match | True |
| Valid index mismatches | 0 |
| HIP heatmap time before E1 | ~69 ms |
| HIP heatmap time after E1 | ~691 ms |
| Outcome | Rejected |

Reason for rejection: the segmented reduction added extra global memory traffic, intermediate buffers, and another reduction pass. It made runtime roughly 10x worse while preserving correctness.

### E2: HIP Stage Profiling

E2 added HIP-event timing around the current dense backend.

| Stage | Average ms | P95 ms | Share of device time |
|---|---:|---:|---:|
| Host to device copy | 0.09 | 0.16 | 0.1% |
| Full-resolution cubic resize | 13.12 | 13.18 | 19.7% |
| Vertical max pass | 15.75 | 15.96 | 23.7% |
| Horizontal max pass | 9.65 | 10.14 | 14.5% |
| TopK scan/reduction | 27.98 | 28.13 | 42.1% |
| Device to host scores | 0.03 | 0.04 | <0.1% |
| Device to host indices | 0.02 | 0.03 | <0.1% |
| Device total | 66.49 | 67.40 | 100.0% |
| Total HIP profiled time | 66.64 | 67.62 | — |

Key finding: host-device copies are negligible in this test. The main bottlenecks are GPU-side dense memory work and the final full-resolution TopK scan.

---

## Bottleneck Analysis

The current backend performs the following dense path:

```text
heatmaps [B,18,68,121]
  -> full-resolution cubic resize [B,18,1080,1920]
  -> vertical max buffer [B,18,1080,1920]
  -> horizontal max / pooled buffer [B,18,1080,1920]
  -> full-resolution TopK scan
```

For batch size 4, the dense buffers are very large:

| Buffer | Shape | Approx. size |
|---|---|---:|
| resized | `[4,18,1080,1920]` | ~597 MB |
| vertical | `[4,18,1080,1920]` | ~597 MB |
| pooled | `[4,18,1080,1920]` | ~597 MB |
| Total dense intermediates | — | ~1.79 GB |

This explains why optimization must reduce full-resolution materialization and repeated memory traversal, not only adjust the TopK reduction structure.

---

## Current Conclusions

| Question | Answer |
|---|---|
| Can the graph be split between heatmap TopK and PAF pruning? | Yes |
| Can MXR2 consume externally generated TopK tensors? | Yes |
| Does the HIP backend match merged MIGraphX semantics on real frames? | Yes |
| Are valid heatmap peaks correctly preserved? | Yes, valid index mismatches are 0 |
| Is the current split HIP path faster than merged MXR? | No |
| Are CPU-GPU transfers the main bottleneck? | No |
| Is dense TopK a bottleneck? | Yes |
| Is dense resize/NMS memory traffic also a bottleneck? | Yes |
| Was segmented TopK reduction successful? | No, rejected due ~10x slowdown |

---

## Recommended Next Step: E3

The next experiment should preserve the stable dense backend as fallback and add an experimental fused candidate path.

### E3 Goal

Replace this pair:

```text
horizontal_max_kernel
-> topk_from_pooled_kernel
```

with an experimental fused path:

```text
vertical buffer
-> fused horizontal max + candidate extraction
-> compact candidate list
-> small TopK over candidates
```

### Why E3 Makes Sense

Current measured cost:

| Combined region | Current cost |
|---|---:|
| horizontal max | 9.65 ms |
| TopK scan | 27.98 ms |
| Combined | 37.63 ms |

E3 should target this 37.63 ms region. Even a partial reduction could move the backend from ~66.6 ms toward ~45-50 ms.

### E3 Design Constraints

| Constraint | Reason |
|---|---|
| Keep dense backend as fallback | Avoid destabilizing validated correctness path |
| Add experimental entrypoint or mode | Allows A/B testing without breaking existing tests |
| Preserve semantic comparison tooling | Must keep `valid_index_mismatch_count = 0` |
| Measure stage timings separately | Prevent another blind E1-style regression |
| Validate first on real frames | Random input can hide valid-peak errors |

### Expected E3 Success Criteria

| Metric | Target |
|---|---:|
| Semantic passed all runs | True |
| Valid index mismatches | 0 |
| Final MXR2 outputs exact/allclose | True |
| HIP heatmap time | < 66 ms |
| Preferred first target | 45-50 ms |

---

## Status

| Phase | Description | Status |
|---|---|---:|
| A | Split architecture concept | Done |
| B | MXR1 / MXR2 graph export | Done |
| C | Host-mediated Python external backend | Done |
| D | Native HIP dense correctness backend | Done |
| D.1 | Random-input HIP validation | Done |
| D.2 | Real-frame HIP validation | Done |
| E1 | Segmented TopK optimization | Rejected |
| E2 | HIP stage profiling | Done |
| E3 | Fused candidate TopK path | Next |
| F | GPU-resident / zero-copy handoff | Later |
| G | Multi-camera simulator integration | Later |

---

## Final Assessment

The custom HIP kernel direction is technically validated but not yet performance competitive. The strongest outcome so far is not speedup; it is the successful decoupling of the heatmap TopK stage from the MIGraphX graph while preserving downstream MXR2 semantics on real video frames.

The performance evidence now points clearly to the next optimization target: reduce dense full-resolution memory traffic and avoid a separate full-resolution TopK scan. E3 should therefore implement a fused horizontal-max candidate extraction path with a compact TopK reduction over candidates, keeping the current dense backend as a correctness fallback.
