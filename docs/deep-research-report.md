# Bridge Report: From Stream Simulation Grid Search to Focused MIGraphX Pipeline Optimization


The first grid search compared multiple postprocessing and streaming strategies. It established that live multi-camera performance depends not only on raw inference speed, but also on buffering mode, backpressure, worker topology, queue pressure, and tail latency.

The second grid search narrowed the scope to the merged fused-pruned MIGraphX pipeline. At that stage, the main question changed from:

```text
Which postprocessing variant is best?
```

to:

```text
Why does the merged MIGraphX stream pipeline perform far below isolated model benchmarks?
```

The key finding was that the early B4 streaming collapse was not caused by the B4 `.mxr` model itself. In isolation, the B4 model ran at approximately **85 ms per batch** and **47 FPS effective throughput**. In the initial full stream simulation, the same B4 path appeared to run at approximately **802 ms per batch**, with only **~2.13 aggregate FPS** and **~7% GPU utilization**. This gap showed that the bottleneck was outside the compiled model.

The investigation then focused on two runtime issues:

1. **Camera-to-inference transport overhead** caused by sending large preprocessed tensors through multiprocessing queues.
2. **CPU scheduling and process placement**, which prevented the inference worker from consistently feeding MIGraphX and the GPU.

The final optimized path combined:

```text
shared input transport + CPU pinning + B4 static batch
```

This recovered stream performance to approximately **29–30 aggregate FPS**, reduced average E2E latency to approximately **144 ms**, and increased GPU utilization to approximately **77%**.

The follow-up focused grid validated this conclusion by sweeping B2/B4/B8, shared input on/off, and pinning on/off. The best practical configuration was:

```text
B4 + shared-input + CPU pinning
```

B2 remained the lower-latency and lower-memory alternative. B8 was rejected because it increased latency, VRAM usage, and downstream backpressure without improving delivered throughput.

---


## 1. Starting point from the first grid search

The first grid search showed that live-feed performance is governed by pipeline behavior, not just by single-stage speed.

Important principles from that phase were:

- `latest` buffering is preferred for live streams because freshness matters more than processing every frame.
- Soft backpressure helps avoid stale work accumulation.
- Worker count changes can improve throughput but may increase latency and queue pressure.
- Postprocessing cost must be analyzed together with queue delay and p95 latency.
- GPU acceleration is only useful if the surrounding pipeline can keep the GPU fed.

These principles shaped the later investigation. When the merged fused-pruned MIGraphX model became available, the expectation was that postprocessing would no longer be the dominant bottleneck. The new risk was that the runtime system around MIGraphX would become the limiting factor.

---

## 2. Initial B4 streaming failure

The first major test in this phase used the merged fused-pruned B4 MIGraphX model in a 10-camera stream setup:

```text
variant = mx_merged_pose_fused_pruned
num_cameras = 10
camera_fps = 24
buffer_mode = latest
backpressure = soft
infer_workers = 1
post_workers = 3
migraphx_batch_size = 4
migraphx_batch_timeout_ms = 8
```

The result was a severe collapse:

| Metric | Observed value |
|---|---:|
| Aggregate output FPS | ~2.13 FPS |
| Avg FPS per camera | ~0.21 FPS |
| Avg E2E latency | ~1301 ms |
| P95 E2E latency | ~1557 ms |
| Avg B4 batch inference | ~802 ms |
| Avg pre→infer queue | ~266 ms |
| GPU utilization | ~7% |
| Avg real batch size | ~3.97 / 4 |

At first, this suggested that the B4 `.mxr` graph might be inefficient. However, the high `avg_real_batch_size` showed that the simulator was successfully forming full batches. The batch collector was not the immediate cause.

The failure pattern was more consistent with runtime starvation:

```text
full batches + very slow model.run + very low GPU utilization
```

This indicated that the model was not being executed under conditions similar to the isolated benchmark.

---

## 3. Isolated MIGraphX benchmark

To isolate model performance, the `.mxr` files were benchmarked directly with `model.run()`.

The isolated benchmark used the model-reported input shape:

```text
B x 3 x 544 x 968
```

This confirmed that the model consumes already-resized and preprocessed tensors. The isolated results were:

| Model | Avg batch time | Avg frame time | Effective FPS |
|---|---:|---:|---:|
| B1 | 23.75 ms | 23.75 ms | 42.10 FPS |
| B2 | 45.76 ms | 22.88 ms | 43.71 FPS |
| B4 | 85.09 ms | 21.27 ms | 47.01 FPS |
| B8 | 174.68 ms | 21.83 ms | 45.80 FPS |

This disproved the hypothesis that B4 was inherently slow. The B4 model was approximately **85 ms/batch** in isolation, but approximately **802 ms/batch** in the initial stream test.

The investigation therefore moved from model compilation to stream runtime behavior.

---

## 4. Hypothesis 1: queue transport overhead

The first runtime hypothesis was that camera workers were sending large preprocessed tensors through `multiprocessing.Queue`.

A single preprocessed input tensor has shape:

```text
1 x 3 x 544 x 968
```

At `float32`, that is approximately **6 MB per frame**. With 10 camera workers continuously publishing frames, the system could spend significant CPU time and memory bandwidth copying tensors that would later be dropped by `latest` buffering.

The proposed fix was shared-memory input transport:

```text
camera process:
  preprocess frame
  write tensor into shared memory slot
  enqueue metadata only

inference process:
  dequeue metadata
  read tensor view from shared memory
  assemble MIGraphX batch
```

This introduced:

```text
--shared-input-slots
--shared-input-dtype
```

For 10 cameras, the intended configuration was:

```text
--shared-input-slots 10
--shared-input-dtype float32
```

---

## 5. Shared input result

Shared input significantly improved transport behavior, but did not solve the full problem.

| Metric | Before shared input | Shared input, no pinning |
|---|---:|---:|
| Aggregate FPS | ~2.13 FPS | ~2.34 FPS |
| Avg E2E latency | ~1301 ms | ~1269 ms |
| Avg pre→infer queue | ~266 ms | ~45 ms |
| GPU utilization | ~7.2% | ~7.7% |

This result was important because it separated two effects:

- Shared input **did** reduce pre→infer queue delay.
- Shared input **did not** recover throughput or GPU utilization.

Therefore, queue-copy overhead was real, but not the dominant cause of the B4 collapse.

The remaining issue was most likely CPU scheduling and process placement.

---

## 6. Hypothesis 2: CPU scheduling and process placement

The stream pipeline runs multiple active processes:

```text
10 camera workers
1 inference worker
3 postprocess workers
```

Without explicit CPU affinity, Linux can migrate these processes across cores. That is problematic because camera workers are CPU-heavy, while the inference worker needs stable CPU availability to submit work to MIGraphX and keep the GPU busy.

The proposed solution was CPU pinning:

```text
camera workers:   cores 0–9
inference worker: core 10
post workers:     cores 12–14
```

The relevant flags were:

```bash
--pin-cpus
--pin-camera-base 0
--pin-inference-base 10
--pin-post-base 12
--pin-all-threads
--worker-threads 1
```

This isolated the major process groups and prevented the inference worker from competing unpredictably with camera and postprocess workers.

---

## 7. Shared input + CPU pinning result

Combining shared input and CPU pinning produced the decisive improvement.

| Metric | Shared input, no pinning | Shared input + pinning |
|---|---:|---:|
| Aggregate FPS | ~2.34 FPS | ~29.60 FPS |
| Avg FPS per camera | ~0.23 | ~2.96 |
| Avg E2E latency | ~1269 ms | ~144 ms |
| P95 E2E latency | ~1464 ms | ~170 ms |
| Avg B4 batch inference | ~900+ ms | ~94 ms |
| Avg pre→infer queue | ~45 ms | ~12 ms |
| Avg postprocess | ~21 ms | ~2.7 ms |
| GPU utilization | ~7.7% | ~77.2% |

The streamed B4 inference time became close to the isolated benchmark:

```text
B4 isolated:      ~85 ms/batch
B4 stream pinned: ~94 ms/batch
```

This confirmed the root cause:

```text
The B4 .mxr model was not the bottleneck.
The bottleneck was runtime scheduling and data transport around the model.
```

Shared input reduced transport overhead. CPU pinning allowed the inference worker to consistently feed the GPU.

---

## 8. Batch-size evaluation

After fixing the runtime bottleneck, batch size became the next decision point.

Isolated model benchmarks suggested only modest scaling:

| Batch | Effective FPS | Interpretation |
|---:|---:|---|
| B1 | 42.10 FPS | Baseline |
| B2 | 43.71 FPS | Slight improvement |
| B4 | 47.01 FPS | Best isolated throughput |
| B8 | 45.80 FPS | Stable, but not better than B4 |

The full-stream results confirmed that B4 was the best practical point.

### B4 versus B8

| Metric | B4 shared input + pinning | B8 shared input + pinning |
|---|---:|---:|
| Aggregate FPS | ~29.60 FPS | ~25.14 FPS |
| Avg FPS per camera | ~2.96 | ~2.51 |
| Avg E2E latency | ~144 ms | ~259 ms |
| P95 E2E latency | ~170 ms | ~278 ms |
| Avg batch inference | ~94 ms / B4 | ~193 ms / B8 |
| GPU utilization | ~77% | ~80% |
| VRAM avg | ~3.48 GB | ~6.33 GB |
| Replaced before post | 0 | 1102 |
| Backpressure skips | 29 | 54673 |

B8 successfully filled batches, so the issue was not batching efficiency. The issue was service time and burstiness. Larger batches increased latency, memory usage, and downstream replacement/backpressure without improving delivered throughput.

B8 was therefore rejected for the current single-inference-worker architecture.

---

## 9. B8 transport control run

A B8 pinned run without shared input was also executed. Its summary reported:

```text
shared_input_slots = 0
```

This acted as a transport control.

| Metric | B8 shared input + pinning | B8 queue input + pinning |
|---|---:|---:|
| Aggregate FPS | ~25.14 FPS | ~23.73 FPS |
| Avg E2E latency | ~259 ms | ~287 ms |
| P95 E2E latency | ~278 ms | ~315 ms |
| Avg pre→infer queue | ~16 ms | ~39 ms |
| Avg B8 batch inference | ~193 ms | ~207 ms |
| GPU utilization | ~80% | ~73% |

This confirmed that shared input still improves the pinned pipeline. However, it did not change the main conclusion: B8 remained worse than B4.

---

## 10. Why the focused grid was needed

The manual experiments established the likely cause-and-effect chain, but the results needed to be formalized in a reproducible grid.

The focused grid was designed to answer:

1. How much does CPU pinning matter?
2. How much does shared input matter?
3. Which batch size is best: B2, B4, or B8?
4. Does B4 remain best after controlled ablations?
5. Are the manual results reproducible?

The resulting grid covered:

```text
B2 shared-input pinned
B4 queue-input unpinned
B4 queue-input pinned
B4 shared-input unpinned
B4 shared-input pinned
B8 shared-input pinned
```

This was intentionally a causal grid, not a broad exploratory grid:

- B4 included the full transport/pinning ablation.
- B2 and B8 tested the optimized path.
- All runs used the same live-stream envelope.
- The grid directly tested the hypotheses developed during debugging.

---

## 11. Change-to-effect summary

| Change | Effect | Conclusion |
|---|---|---|
| Isolated benchmark with model-reported input shape | B4 measured ~85 ms/batch | The model was not inherently slow |
| Shared input | Reduced pre→infer queue delay | Queue transport overhead was real |
| Shared input without pinning | Throughput remained ~2 FPS | Transport alone was insufficient |
| CPU pinning | B4 reached ~29–30 FPS and ~77% GPU utilization | Process placement was decisive |
| B4 batch size | Best full-stream throughput and stability | Best current batch size |
| B8 batch size | Higher latency, VRAM, and backpressure | Too bursty for current pipeline |
| Focused grid runner | Encoded the causal hypotheses into repeatable experiments | Converted debugging into reportable evidence |

---

## 12. Recommended configuration

The recommended configuration from the intermediate investigation is:

```text
variant = mx_merged_pose_fused_pruned
model = pose_fused_pruned_batchaware_b4_1080x1920_k20_m20_thr0p1_r6_separable.mxr
num_cameras = 10
camera_fps = 24
buffer_mode = latest
backpressure_mode = soft
max_pending_age_ms = 300
infer_workers = 1
post_workers = 3
migraphx_batch_size = 4
migraphx_batch_timeout_ms = 8
shared_input_slots = 10
shared_input_dtype = float32
pin_cpus = true
pin_camera_base = 0
pin_inference_base = 10
pin_post_base = 12
pin_all_threads = true
worker_threads = 1
```

Rationale:

1. Highest validated full-stream throughput in this investigation.
2. Stream inference time close to isolated model performance.
3. Stable downstream postprocess behavior.
4. Lower latency and VRAM than B8.
5. Strong GPU utilization without excessive queue buildup.
6. Directly reproducible by the focused grid.

---

## 13. Operational interpretation

The optimized B4 configuration does **not** achieve true 10×24 FPS realtime.

The target is:

```text
10 cameras x 24 FPS = 240 aggregate FPS
```

The best observed single-worker B4 configuration reaches approximately:

```text
~29–34 aggregate FPS
```

This corresponds to approximately:

```text
~3 FPS per camera
```

The correct conclusion is therefore:

```text
The merged MIGraphX path, after shared-input transport and CPU pinning,
supports a stable low-FPS live-monitoring mode across 10 cameras with one
inference worker.
```

It does not support full 24 FPS per camera under the tested single-worker setup.

Future work should target:

- multiple inference workers,
- multiple GPUs,
- lower input resolution,
- more aggressive pruning,
- GPU-side preprocessing,
- reduced camera-worker CPU overhead,
- and alternative scheduling models.

---

## 14. Final conclusion

The engineering bridge between the two grid searches is the transition from **postprocessing optimization** to **runtime systems optimization**.

The first grid search identified the live-stream principles:

- latest-frame buffering,
- soft backpressure,
- worker topology,
- queue pressure,
- p95 latency,
- and live output freshness.

The intermediate debugging phase identified the runtime bottlenecks:

- queue-copy overhead,
- CPU scheduling instability,
- GPU starvation,
- and batch-size burstiness.

The second grid search validated the optimized merged-MIGraphX path:

```text
B4 + shared input + CPU pinning
```

This is the current best single-worker configuration for 10-camera low-FPS live monitoring. B2 remains the lower-latency and lower-memory alternative. B8 should not be used unless the downstream pipeline and scheduling model are redesigned to handle larger bursts.