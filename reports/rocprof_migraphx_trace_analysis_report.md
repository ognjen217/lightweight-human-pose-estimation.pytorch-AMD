# ROCprof Trace Analysis Report: B4 MIGraphX Multi-Camera Live-Feed Pipeline

## 1. Purpose of the profiling run

The goal of this profiling pass was to understand the current bottleneck of the optimized multi-camera live-feed execution path after the pipeline was migrated toward a merged/fused/pruned MIGraphX implementation.

The previous optimization work already moved the most expensive postprocessing stages away from the original CPU-heavy path and into the compiled MIGraphX graph. The remaining question was therefore not whether the old Python postprocessing was still slow, but rather where the optimized runtime now spends time:

```text
camera frame production
→ preprocessing
→ latest-frame buffering / shared-memory input handoff
→ MIGraphX batched inference
→ decode / fused-pruned output assembly
→ postprocess accounting
→ output frame statistics
```

The profiling run was designed to answer the following questions:

1. Is the pipeline still limited by CPU preprocessing?
2. Is the latest-frame / batch collector waiting too long before launching inference?
3. Is queue pre→infer latency caused by poor batch filling, or by the inference worker being busy?
4. Is postprocessing still a relevant bottleneck?
5. What does the GPU trace show inside the MIGraphX B4 model execution?
6. Is the current limitation caused by the live-feed orchestration, or by the compiled graph execution itself?

The most important result is that the pipeline is now primarily **MIGraphX graph execution limited**. The main bottleneck is no longer CPU-side postprocessing. Instead, the B4 merged fused/pruned `.mxr` graph executes as a highly fragmented sequence of GPU kernels, with expensive `TopK`, `Gather`, pooling, logical/where and convolution-related kernels, plus a significant amount of launch/runtime gap time between kernels.

---

## 2. Profiling setup

The analyzed configuration used the refactored live-feed simulation entrypoint:

```text
simulate_camera_stream.py
```

The profiled model was the B4 member of the merged pose fused-pruned batch-aware model family:

```text
models/merged_pose_fused_pruned_batchaware/
  pose_fused_pruned_batchaware_b4_1080x1920_k20_m20_thr0p1_r6_separable.mxr
```

The runtime configuration was:

```text
variant:                    mx_merged_pose_fused_pruned
num_cameras:                10
camera_fps:                 24
buffer_mode:                latest
backpressure_mode:          soft
max_pending_age_ms:         300
infer_workers:              1
post_workers:               4
migraphx_batch_size:        4
migraphx_batch_timeout_ms:  0
shared_input_slots:         10
shared_input_dtype:         float16
mp_start_method:            spawn
warmup_s:                   10
collector_coalesce:         true
target input size:          3 × 544 × 968
```

The relevant outputs were:

```text
summary.json
detailed.csv
run.log
2103046_kernel_trace.csv
2103046_hip_api_trace.csv
2103046_hsa_api_trace.csv
```

The run used ROCprof CSV output rather than relying only on the high-level simulator metrics. This made it possible to inspect both the pipeline-level behavior and the GPU/API-level behavior.

One important interpretation detail is that this was a profiled run using `spawn` as the multiprocessing start method. Therefore, the absolute FPS should not be treated as a perfectly clean production benchmark. Profiling overhead and the different process-start method can reduce throughput. However, the relative stage distribution and the GPU kernel-level breakdown are still highly useful for bottleneck identification.

---

## 3. High-level simulation results

The high-level simulation summary shows the following runtime behavior:

| Metric | Value |
|---|---:|
| Total processed post-warmup frames | 2056 |
| Raw processed frames | 2367 |
| Warmup discarded frames | 311 |
| Wall time | 86.27 s |
| Active cameras | 10 |
| Aggregate output FPS | 23.83 FPS |
| Average output FPS per camera | 2.38 FPS/camera |
| Average preprocess time | 4.52 ms |
| Average queue pre→infer time | 21.73 ms |
| Average inference time | 31.80 ms/frame |
| Average decode time | 0.04 ms |
| Average queue infer→post time | 4.08 ms |
| Average postprocess time | 3.50 ms |
| Average E2E latency | 165.81 ms |
| P95 E2E latency | 191.49 ms |
| P95 postprocess time | 8.22 ms |

The most important pipeline-level observation is that the largest stage is inference. Since the configured batch size is B4, the top-level average inference value of approximately **31.8 ms per frame** corresponds to roughly:

```text
31.8 ms/frame × 4 frames ≈ 127 ms per B4 batch
```

This is consistent with the kernel trace, which shows that one B4 batch spans roughly 134 ms during the traced steady-state window.

The other pipeline stages are much smaller:

```text
preprocess:          ~4.5 ms average
queue infer→post:    ~4.1 ms average
postprocess:         ~3.5 ms average
decode:              ~0.04 ms average
```

Therefore, the old CPU-side postprocessing bottleneck has effectively been removed from the critical path. The remaining limitation is now dominated by the B4 MIGraphX execution stage.

---

## 4. Per-camera behavior

The output distribution across cameras was balanced. Each camera produced roughly the same number of post-warmup output frames:

| Camera | Frames | FPS | Avg E2E latency | P95 E2E latency |
|---:|---:|---:|---:|---:|
| 0 | 205 | 2.38 | 157.49 ms | 177.23 ms |
| 1 | 204 | 2.36 | 164.45 ms | 188.49 ms |
| 2 | 206 | 2.39 | 154.91 ms | 182.65 ms |
| 3 | 205 | 2.38 | 167.83 ms | 190.31 ms |
| 4 | 206 | 2.39 | 162.24 ms | 179.39 ms |
| 5 | 207 | 2.40 | 162.97 ms | 188.92 ms |
| 6 | 207 | 2.40 | 170.78 ms | 192.42 ms |
| 7 | 206 | 2.39 | 173.53 ms | 193.29 ms |
| 8 | 205 | 2.38 | 167.86 ms | 191.33 ms |
| 9 | 205 | 2.38 | 176.07 ms | 194.48 ms |

This indicates that the latest-frame pipeline does not strongly starve any individual camera. The system is not processing every input frame, but the output sampling is relatively fair across the 10 streams.

The per-camera FPS of approximately 2.36–2.40 confirms the current practical behavior of the system: it operates as a **low-FPS multi-camera monitoring pipeline**, not as a full 10×24 FPS processing pipeline.

This behavior is expected because latest-frame buffering intentionally replaces older frames when the inference stage cannot keep up with the camera input rate. The aim is to keep processed frames fresh rather than to process every frame.

---

## 5. Camera preprocessing behavior

The average preprocessing time is moderate, but it is not completely uniform across camera sources.

Several cameras show average preprocessing times around 8–10 ms, while others are closer to 5–7 ms:

| Camera group | Observed behavior |
|---|---|
| Cameras 0, 1, 2, 3, 5 | Higher average preprocess time, often around 8–10 ms |
| Cameras 6, 7, 8, 9 | Lower average preprocess time, around 5–7 ms |
| P95 preprocess | Some cameras reach 28–30 ms p95 |

This indicates that preprocessing can still create jitter. However, it does not dominate the end-to-end pipeline because the B4 inference batch time is much larger.

The practical interpretation is:

```text
Preprocessing optimization may reduce CPU load and jitter,
but it will not produce the largest throughput improvement
unless MIGraphX inference time is also reduced.
```

This means preprocessing remains a valid secondary optimization target, but it is no longer the primary bottleneck.

---

## 6. Interpretation of queue pre→infer latency

The average queue pre→infer latency was:

```text
avg_queue_pre_to_infer_ms ≈ 21.73 ms
```

At first glance, this could look like a queueing problem. However, the trace/log behavior shows that this is mostly a symptom of the inference worker being busy with the previous B4 batch.

In earlier trace log lines from the same style of run, the inference worker repeatedly reported:

```text
actual_batch=4
launch=full
wait_fill≈0.06–0.16 ms
infer≈125–137 ms
decode≈0.08–0.14 ms
```

This means the collector is not spending significant time waiting to fill the B4 batch. The B4 batch is almost always full immediately or nearly immediately. Therefore, `queue pre→infer` does **not** primarily mean:

```text
the batch collector is waiting too long to gather four frames
```

It means:

```text
a frame has already been preprocessed and published,
but the inference worker is still executing the previous MIGraphX B4 batch.
```

Therefore, queue pre→infer is mainly a consequence of expensive B4 inference, not an independent root bottleneck.

This distinction matters because it changes the optimization priority. If the batch collector were waiting to fill batches, then tuning timeout/freshness policy would be the main fix. But here the batch is already full, and the dominant delay is the runtime of the previous B4 `model.run()`.

---

## 7. Postprocessing is no longer the main bottleneck

The postprocess stage is small relative to inference:

```text
avg_post_ms:      ~3.50 ms
p95_post_ms:      ~8.22 ms
avg_decode_ms:    ~0.04 ms
```

This confirms that the merged/fused/pruned MIGraphX pipeline achieved its main architectural goal: the old CPU-heavy postprocessing is no longer the dominant stage.

There is still variation in postprocess time depending on the number of detected poses and output complexity, but the stage is no longer large enough to explain the throughput limit.

The practical conclusion is:

```text
Further optimizing CPU postprocess is unlikely to produce a major aggregate FPS improvement.
The next optimization stage should focus on the compiled MIGraphX graph and its GPU execution pattern.
```

---

## 8. ROCprof kernel trace summary

The ROCprof kernel trace contained:

| Kernel trace metric | Value |
|---|---:|
| Kernel dispatch rows | 257,676 |
| Trace window span | ~35.03 s |
| Total active GPU kernel time | ~25.56 s |
| Estimated GPU active ratio | ~72.96% |
| Estimated B4 batch executions in trace | 261 |
| Kernel dispatches per B4 batch | ~987–988 |
| Average trace span per B4 batch | ~134.22 ms |
| Active GPU kernel time per B4 batch | ~97.92 ms |
| Inter-kernel gap / non-kernel time per batch | ~36.30 ms |

This is the most important profiling result.

A single B4 `model.run()` does not execute as a small number of large fused kernels. Instead, it launches almost **one thousand GPU kernel dispatches per batch**.

The approximate per-batch timeline is:

```text
B4 batch span:              ~134 ms
active GPU kernel time:      ~98 ms
inter-kernel gap/runtime:    ~36 ms
```

This means that even during inference, the GPU is not continuously executing kernels. Around 27% of the traced batch window is not active GPU kernel execution. This time likely comes from a combination of:

```text
kernel launch overhead
runtime scheduling overhead
HSA/HIP dispatch overhead
small inter-kernel gaps
synchronization / dependency boundaries
MIGraphX graph fragmentation
```

This makes the key bottleneck more precise:

```text
The bottleneck is not just “inference”.
The bottleneck is fragmented MIGraphX graph execution with many GPU dispatches per B4 batch.
```

---

## 9. Most expensive GPU kernel groups

The top kernel groups by total time per B4 batch are:

| Kernel / group | Count per B4 batch | Time per B4 batch | Interpretation |
|---|---:|---:|---|
| `gather_kernel` | ~707 | ~17.68 ms | Very frequent gather operations; most are tiny, but a few are very expensive |
| `topk_kernel` | 3 | ~17.39 ms | Expensive TopK candidate selection |
| `mlir_convolution_broadcast_add_relu` | 40 | ~17.17 ms | Backbone / convolutional part of the graph |
| `mul_mul_mul_mul_add_add_add_kernel` | 2 | ~16.89 ms | Large fused elementwise operation, likely from fused postprocess math |
| `mloPoolingG` | 2 | ~13.98 ms | Pooling / NMS-like operation |
| `greater_convert_equal_convert_logical_and_where_kernel` | 1 | ~9.42 ms | Logical mask / where operation |
| `mlir_convolution_broadcast_add_relu_add` | 5 | ~2.41 ms | Additional convolution/relu/add work |
| Other convolution/slice/elementwise kernels | many | smaller individually | Contributes to total dispatch count |

These groups account for almost all of the active GPU kernel time per batch.

The important conclusion is that the cost is concentrated in a small number of conceptual graph regions:

```text
Gather
TopK
Pooling
Logical/Where
large fused elementwise kernels
convolution/relu blocks
```

This strongly suggests that the next optimization stage should focus on the graph structure around TopK/Gather/NMS-like processing, not on Python postprocess.

---

## 10. Gather kernel analysis

`gather_kernel` is the most suspicious kernel group because it appears extremely frequently.

The observed gather statistics were:

| Gather metric | Value |
|---|---:|
| Total gather kernels in trace | 184,610 |
| Gather kernels per B4 batch | ~707 |
| Gather time per B4 batch | ~17.68 ms |
| Median gather duration | ~1.76 µs |
| P95 gather duration | ~5.60 µs |
| Gather kernels > 1 ms | 1,044 |
| Large gather kernels per B4 batch | 4 |
| Large gather time per B4 batch | ~15.35 ms |

This is a very important distinction.

Most gather kernels are tiny. However, each batch contains approximately four large gather kernels, and those large gather kernels account for most of the gather cost.

Therefore, the issue is not only the number of gather kernels. The issue is that the graph contains a few large gather operations per batch that are expensive enough to become a major part of the B4 execution time.

This likely comes from the GPU-side postprocess / candidate-selection part of the graph. The fused graph is functionally correct, but the way candidate selection is represented in ONNX/MIGraphX appears to produce expensive indexing operations.

The optimization direction should be:

```text
reduce or restructure large Gather operations
avoid unnecessary dynamic indexing patterns
reduce the tensor size before Gather where possible
replace Gather-heavy postprocess logic with a simpler dense-mask or reduced-candidate approach
```

---

## 11. TopK kernel analysis

`topk_kernel` is another major cost.

The observed TopK behavior was:

| TopK metric | Value |
|---|---:|
| TopK kernels per B4 batch | 3 |
| TopK time per B4 batch | ~17.39 ms |
| Average TopK kernel duration | ~5.80 ms |
| P95 TopK kernel duration | ~16.79 ms |
| Max TopK kernel duration | ~17.65 ms |

This is significant. A small number of TopK calls contribute almost as much time as the entire gather group.

This suggests that the current GPU postprocess graph may be doing TopK over tensors that are still too large, or doing TopK in a shape/layout that is expensive for MIGraphX.

The optimization direction should be:

```text
reduce K if accuracy permits
perform TopK on a smaller candidate set
avoid repeated TopK over large tensors
split or restructure TopK if MIGraphX handles a different layout better
test alternative candidate extraction approaches that avoid expensive TopK
```

This is especially important because the current model name includes:

```text
k20_m20_thr0p1_r6_separable
```

If `K=20` is still more than needed for the target use case, smaller candidate counts could be tested. However, any reduction in K must be validated against accuracy and detection quality, not only speed.

---

## 12. Pooling and logical/where kernels

The trace also shows substantial cost in pooling and logical mask operations:

```text
mloPoolingG:
  ~2 kernels per B4 batch
  ~13.98 ms per B4 batch

greater_convert_equal_convert_logical_and_where_kernel:
  ~1 kernel per B4 batch
  ~9.42 ms per B4 batch
```

These likely correspond to NMS-like filtering and mask construction inside the fused graph.

This is important because it confirms that the GPU-side postprocessing logic is still expensive, only now the expense is visible as GPU kernels rather than Python functions.

This is an expected stage in the optimization process:

```text
Before:
  CPU postprocess was the dominant bottleneck.

Now:
  postprocess is fused into the MIGraphX graph,
  but the graph-level implementation of postprocess contains expensive GPU kernels.
```

Therefore, the next step is not simply to move more work to the GPU, but to make the GPU graph cheaper.

---

## 13. HIP API trace summary

The HIP runtime API trace shows a large number of kernel launch calls.

The most relevant HIP API statistics per B4 batch were:

| HIP API function | Calls per B4 batch | Time per B4 batch |
|---|---:|---:|
| `hipExtModuleLaunchKernel` | ~487 | ~11.61 ms |
| `hipSetDevice` | ~492 | ~0.11 ms |
| `hipFree` | ~3 | ~0.74 ms |
| `hipHostMalloc` | ~3 | ~0.15 ms |
| `hipLaunchKernel` | ~3.5 | ~0.07 ms |
| `hipStreamSynchronize` | ~0.5 | ~0.02 ms |

The key observation is:

```text
The trace contains hundreds of kernel launch calls per B4 batch.
```

This supports the kernel trace conclusion: the MIGraphX graph is fragmented into many GPU dispatches.

The total time spent in `hipExtModuleLaunchKernel` is not as large as the total GPU kernel time, but it is large enough to matter. Around **11.6 ms per batch** is associated with this launch API category.

This means that improving graph fusion and reducing the number of kernels could improve performance in two ways:

1. Reduce active GPU work if redundant or expensive graph operations are removed.
2. Reduce CPU/HIP/HSA launch overhead and inter-kernel gaps by reducing dispatch count.

---

## 14. HSA API trace summary

The HSA trace also confirms a high-dispatch execution pattern.

The most relevant HSA API statistics per B4 batch were:

| HSA API function | Calls per B4 batch | Time per B4 batch |
|---|---:|---:|
| `hsa_signal_store_screlease` | ~990 | ~14.00 ms |
| `hsa_amd_memory_pool_free` | 6 | ~1.44 ms |
| `hsa_amd_memory_pool_allocate` | 6 | ~0.21 ms |
| `hsa_queue_add_write_index_screlease` | ~990 | ~0.12 ms |
| `hsa_queue_load_read_index_scacquire` | ~990 | ~0.09 ms |

The high number of `hsa_signal_store_screlease` calls lines up almost exactly with the number of kernel dispatches per batch. This again indicates that the graph launches many kernels and issues many queue/signal operations.

Memory allocation/free activity is present but not dominant:

```text
hsa memory free:      ~1.44 ms per batch
hsa memory allocate:  ~0.21 ms per batch
hipFree:              ~0.74 ms per batch
hipHostMalloc:         ~0.15 ms per batch
```

This means buffer reuse or preallocation may help slightly, but it is not expected to be the main fix.

A realistic expectation for memory allocation/cache optimization would be:

```text
possibly 1–3 ms saved per B4 batch
```

whereas the total gap/runtime overhead and expensive graph operations are much larger.

---

## 15. What this means for the current optimization hypotheses

### 15.1 Preprocessing is not the primary bottleneck

Preprocessing still has jitter, and some camera sources have higher p95 preprocessing times. However, average preprocessing is much smaller than the B4 inference batch runtime.

Preprocessing optimization remains useful for:

```text
reducing CPU pressure
reducing per-camera jitter
reducing frame replacement pressure
improving stability under more cameras
```

but it is not the main source of the current aggregate FPS limit.

### 15.2 Postprocessing is no longer the bottleneck

The CPU-visible postprocess stage is small. This validates the merged/fused/pruned MIGraphX approach.

However, the GPU trace reveals that part of the postprocessing cost has moved into the graph as expensive `TopK`, `Gather`, pooling, and logical/where kernels.

Therefore, the next postprocessing optimization should happen at the graph level, not in the Python postprocess worker.

### 15.3 B4 timeout is not the problem

The current B4 configuration uses timeout 0 ms and still fills batches effectively. Previous trace log lines showed that almost all launches were full-batch launches, with `wait_fill` close to zero.

Therefore, timeout tuning is not the main optimization direction for this configuration.

### 15.4 Queue pre→infer is mostly a symptom, not the root cause

The average queue pre→infer time is around 21.7 ms. This does not mean that the queue operation itself is slow. It means that frames are ready before the inference worker is available.

Because B4 inference takes around 125–135 ms per batch, new frames naturally wait while the previous batch is running.

Therefore:

```text
queue pre→infer latency is mostly caused by long inference batch runtime.
```

Reducing B4 model execution time should also reduce queue pre→infer latency.

### 15.5 The main bottleneck is MIGraphX graph execution efficiency

This is the central conclusion of the trace.

The graph is functionally successful because it removes the old CPU postprocessing bottleneck. However, its GPU execution is expensive and fragmented:

```text
~988 kernel dispatches per B4 batch
~98 ms active GPU kernel time per batch
~36 ms inter-kernel gap/runtime overhead per batch
expensive TopK/Gather/Pooling/Where operations
```

The next major speedup must come from reducing graph cost and graph fragmentation.

---

## 16. Recommended next experiments

### 16.1 Compare standalone B4 MIGraphX trace against stream B4 trace

This is the highest priority next experiment.

The question is:

```text
Does standalone B4 model.run have the same kernel pattern as the stream pipeline?
```

If standalone B4 has approximately the same number of kernels, active GPU time, and batch span, then the bottleneck is inherent to the compiled `.mxr` graph.

If standalone B4 is much faster, then the stream environment is adding overhead through scheduling, memory handoff, process behavior, or runtime interaction.

The standalone test should measure:

```text
model.run average time
kernel dispatches per run
active GPU kernel time per run
inter-kernel gap per run
top kernels by total time
HIP/HSA calls per run
```

Expected interpretation:

```text
same pattern as stream:
  optimize the graph

stream much slower than standalone:
  optimize runtime integration / worker scheduling / input handoff
```

### 16.2 Build a reusable ROCprof CSV trace analyzer

Manual inspection of ROCprof CSV files is slow. A dedicated script should be added, for example:

```text
tools/analyze_rocprof_trace.py
```

The script should read:

```text
*_kernel_trace.csv
*_hip_api_trace.csv
*_hsa_api_trace.csv
summary.json
```

and produce:

```text
trace span
total kernel time
estimated GPU duty
estimated batch count
kernel dispatches per batch
active GPU time per batch
gap/runtime overhead per batch
top kernels by total time
TopK/Gather/Pooling/Where breakdown
HIP calls per batch
HSA calls per batch
memory allocation/free overhead
```

This would turn ROCprof output into repeatable optimization evidence instead of one-off manual analysis.

### 16.3 Optimize the ONNX/MIGraphX graph around TopK/Gather/NMS-like operations

The most valuable graph-level targets are:

```text
TopK
large Gather operations
Pooling/NMS-like operations
logical/where mask construction
large fused elementwise kernels
```

Potential directions:

```text
reduce K if accuracy permits
test K=10 or other smaller candidate limits
perform TopK after stronger thresholding or smaller candidate selection
avoid Gather-heavy patterns if possible
replace dynamic indexing with dense-mask logic where faster
simplify pooling/NMS graph
check whether reshapes/transposes/gathers create unnecessary layout work
test alternative ONNX formulations and compile to MXR
```

Any candidate graph change must be evaluated with both speed and accuracy/quality metrics.

### 16.4 Test whether multiple inference workers can hide inter-kernel gaps

The trace shows approximately 27% non-kernel time inside the traced GPU execution window. In theory, another inference worker might hide some gaps by keeping the GPU busier.

However, this is risky. Multiple workers can also increase:

```text
VRAM usage
p95 latency
GPU contention
runtime scheduling overhead
queueing instability
```

This should be tested only after standalone B4 trace comparison.

Suggested experiment:

```text
B4, 10 cameras, timeout 0 ms
infer_workers = 1 vs 2
same post_workers
same pinning policy
same shared_input dtype
same duration/warmup
```

Success criteria:

```text
aggregate FPS increases meaningfully
p95 E2E latency does not increase unacceptably
GPU utilization improves
real batch size remains full
no severe replaced_before_post or backpressure spikes
```

---

## 17. Final conclusion

The ROCprof trace shows that the optimized live-feed pipeline has moved beyond the original CPU postprocessing bottleneck. The merged fused/pruned MIGraphX pipeline successfully reduced CPU postprocess time to a small fraction of total latency.

The new bottleneck is the execution efficiency of the compiled MIGraphX graph itself.

In the profiled B4 / 10-camera setup, the system reaches approximately 23.8 aggregate FPS under profiling, with around 2.38 FPS per camera, average E2E latency around 166 ms, and p95 E2E latency around 191 ms. The B4 batch is effectively filled, decode is negligible, and postprocess is small.

The GPU trace shows that each B4 batch is executed through approximately 988 GPU kernel dispatches. A typical batch spends roughly 98 ms in active GPU kernels and around 36 ms in inter-kernel gaps or runtime/launch overhead, resulting in an estimated GPU active ratio of about 73%.

The most expensive GPU regions are `Gather`, `TopK`, convolution/relu blocks, pooling/NMS-like kernels, logical/where mask kernels, and large fused elementwise kernels. This means the next optimization phase should focus on reducing graph fragmentation and simplifying the GPU-side postprocessing representation inside the ONNX/MIGraphX graph.

The key engineering conclusion is:

```text
The pipeline is no longer blocked by Python postprocessing.
It is now blocked by the cost and fragmentation of the compiled MIGraphX graph.
```

Therefore, further speedups should prioritize:

```text
1. standalone B4 trace comparison,
2. automated ROCprof CSV analysis,
3. graph-level reduction of TopK/Gather/Pooling/Where cost,
4. careful testing of multi-inference-worker execution only if GPU gaps can be exploited.
```

This profiling pass gives a much clearer optimization direction: the next major gains must come from improving the compiled graph structure and reducing the number/cost of GPU dispatches per batch, rather than further tuning the already small CPU postprocess stage.
