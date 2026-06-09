# Stream Simulation Grid Search — Detailed Performance Report

## 1. Context and goal

This report documents the grid-search results for the 10-camera stream simulation experiments. The purpose of the grid search was to compare postprocessing implementations and software architecture parameters under the same live multi-camera scenario.

The experiments focused on the following questions:

- which postprocessing variant gives the best live-feed latency,
- which variant gives the highest aggregate throughput,
- how worker topology affects queue pressure and end-to-end latency,
- whether soft backpressure and output throttling improve live behavior,
- how GPU NMS hyperparameters affect postprocessing performance,
- and whether CPU `k/threshold` tuning can close the gap to GPU full-resolution NMS.

Low-resolution variants were excluded from this analysis. The report therefore focuses on standard full-resolution CPU postprocessing, CPU optimized variants, GPU full-resolution NMS, and MIGraphX NMS variants.

---

## 2. Dataset and experiment coverage

The plotting bundle contains `265` experiments after filtering, of which `250` are valid runs. The remaining `15` entries are missing/invalid queue-baseline runs and are not used for conclusions.

Valid run count by variant:

| variant                     |   valid_runs |
|:----------------------------|-------------:|
| gpu_nms_fullres_two_process |           58 |
| standard                    |           42 |
| optimized_batch_k20_fast    |           39 |
| cpu_k20_fast_two_process    |           39 |
| migraphx_nms                |           36 |
| migraphx_nms_k20            |           36 |

The generated plot bundle contains the following analysis groups:

- `01_overall_rankings/` — global best/worst rankings and throughput-latency scatter plots
- `02_postprocessing_variants/` — per-variant distributions and common-setup comparisons
- `03_architecture_workers/` — infer/post worker heatmaps and worker-level trends
- `04_backpressure_throttle/` — backpressure/throttle/worker interactions
- `05_buffer_queue_modes/` — latest-buffer vs FIFO queue mode comparisons
- `06_gpu_nms_hparams/` — GPU NMS radius, implementation and dtype sweeps
- `07_cpu_k_threshold/` — CPU max-keypoints and threshold sweeps
- `08_scheduler_pressure/` — counters related to skipped/replaced/throttled frames
- `09_detailed_distributions/` — per-frame distributions from detailed CSV files

---

## 3. Evaluation criteria

For a live multi-camera system, the best configuration is not necessarily the one with the highest aggregate FPS. The main metrics used here are:

- **aggregate output FPS** — total output rate across all camera streams,
- **average E2E latency** — average time from frame capture/preprocess to final pose output,
- **P95 E2E latency** — tail latency, important for perceived live responsiveness,
- **average postprocess time** — direct cost of the postprocessing stage,
- **queue infer→post** — pressure between inference and postprocessing; high values indicate pipeline imbalance.

A balanced score is used only as a ranking helper:

```text
balanced_score = avg_e2e_ms + 0.5 * p95_e2e_ms + 5 * queue_infer_to_post_ms - 20 * aggregate_fps
```

Lower balanced score is better. It intentionally favors low latency and low queue pressure while still rewarding throughput.

---

## 4. High-level conclusion

The central conclusion of the grid search is:

> `gpu_nms_fullres_two_process` is the best live-feed candidate because it provides the lowest E2E latency, lowest P95 latency, and lowest postprocessing time. CPU optimized variants can win on raw aggregate FPS, but they do so with significantly higher latency.

The best overall latency setup was:

```text
variant = gpu_nms_fullres_two_process
buffer_mode = latest
workers = I1/P3
backpressure = soft
max_pending_age_ms = 300
target_output_fps_per_camera = 2
```

Results:

| Metric | Value |
|---|---:|
| Aggregate FPS | 14.15 |
| Avg E2E latency | 185.88 ms |
| P95 E2E latency | 263.03 ms |
| Avg postprocess | 113.84 ms |
| Queue infer→post | 7.43 ms |

A more balanced high-output live candidate was:

```text
variant = gpu_nms_fullres_two_process
workers = I1/P5
backpressure = soft
target_output_fps_per_camera = 3
```

This setup reached 16.84 aggregate FPS with 226.42 ms average E2E and 284.85 ms P95 E2E.

Compared to the best standard baseline, GPU full-resolution NMS improved the best average E2E latency by approximately 6.9×, best postprocess time by approximately 8.4×, and best aggregate FPS by approximately 4.0×.

---

## 5. Variant-level comparison

The table below summarizes both median behavior and best observed behavior for each variant. Median values show robustness across the grid; best values show the strongest observed configuration.

| variant                     |   runs |   median_fps |   best_fps |   median_avg_e2e_ms |   best_avg_e2e_ms |   median_p95_e2e_ms |   best_p95_e2e_ms |   median_post_ms |   best_post_ms |   median_queue_ms |   best_queue_ms |
|:----------------------------|-------:|-------------:|-----------:|--------------------:|------------------:|--------------------:|------------------:|-----------------:|---------------:|------------------:|----------------:|
| gpu_nms_fullres_two_process |     58 |        16.03 |      18.13 |              397.06 |            185.88 |              588.13 |            263.03 |           236.68 |          79.76 |             36.45 |            7.43 |
| cpu_k20_fast_two_process    |     39 |        16.77 |      19.06 |              453.06 |            263.84 |              710.17 |            332.00 |           269.92 |         154.94 |            137.96 |            4.18 |
| optimized_batch_k20_fast    |     39 |        16.88 |      19.46 |              468.59 |            269.69 |              696.94 |            333.25 |           269.22 |         156.25 |            131.36 |            3.98 |
| migraphx_nms                |     36 |        11.58 |      12.65 |              637.87 |            451.53 |              909.18 |            715.99 |           411.18 |         251.59 |            169.27 |          107.73 |
| migraphx_nms_k20            |     36 |        11.72 |      12.87 |              635.20 |            466.60 |              911.14 |            721.69 |           407.49 |         243.13 |            170.40 |           99.20 |
| standard                    |     42 |         4.01 |       4.54 |             1541.72 |           1275.41 |             2283.46 |           1915.44 |          1193.79 |         669.93 |            266.61 |          183.21 |

Interpretation:

- `gpu_nms_fullres_two_process` has the best observed latency and postprocess time.
- `optimized_batch_k20_fast` and `cpu_k20_fast_two_process` achieve the highest raw throughput, but their best-latency configurations are slower than GPU fullres NMS.
- MIGraphX NMS variants are better than the original standard baseline, but do not beat the PyTorch/ROCm GPU fullres NMS implementation in this simulation.
- `standard` remains the slowest variant by a large margin and should only be used as a correctness/reference baseline.

---

## 6. Overall rankings

### 6.1 Highest aggregate throughput

| variant                     | buffer_mode   | workers   | backpressure_mode   |   max_pending_age_ms |   target_output_fps_per_camera |   aggregate_output_fps |   avg_e2e_ms |   p95_e2e_ms |   avg_post_ms |   avg_queue_infer_to_post_ms |
|:----------------------------|:--------------|:----------|:--------------------|---------------------:|-------------------------------:|-----------------------:|-------------:|-------------:|--------------:|-----------------------------:|
| optimized_batch_k20_fast    | latest        | I1/P5     | strict              |               300.00 |                           0.00 |                  19.46 |       506.71 |       733.00 |        247.73 |                       190.81 |
| cpu_k20_fast_two_process    | latest        | I1/P5     | strict              |               300.00 |                           0.00 |                  19.06 |       516.68 |       748.33 |        252.74 |                       195.84 |
| cpu_k20_fast_two_process    | latest        | I2/P6     | strict              |               300.00 |                           0.00 |                  18.39 |       540.79 |       786.12 |        314.21 |                       157.11 |
| optimized_batch_k20_fast    | latest        | I1/P5     | soft                |               300.00 |                           4.00 |                  18.16 |       459.02 |       675.66 |        265.34 |                       126.45 |
| gpu_nms_fullres_two_process | latest        | I2/P6     | strict              |               300.00 |                           0.00 |                  18.13 |       515.10 |       734.67 |        295.09 |                        74.27 |
| optimized_batch_k20_fast    | latest        | I2/P6     | strict              |               300.00 |                           0.00 |                  18.02 |       556.30 |       811.40 |        320.45 |                       165.89 |
| cpu_k20_fast_two_process    | latest        | I2/P5     | strict              |               300.00 |                           0.00 |                  17.99 |       542.94 |       827.61 |        268.43 |                       213.77 |
| optimized_batch_k20_fast    | latest        | I1/P5     | soft                |               200.00 |                           0.00 |                  17.89 |       502.51 |       719.51 |        269.65 |                       163.91 |
| optimized_batch_k20_fast    | latest        | I2/P5     | strict              |               300.00 |                           0.00 |                  17.89 |       542.90 |       804.17 |        269.22 |                       208.27 |
| cpu_k20_fast_two_process    | latest        | I1/P5     | soft                |               300.00 |                           4.00 |                  17.87 |       451.89 |       710.17 |        263.78 |                       119.76 |

The highest raw throughput comes from CPU optimized variants: `optimized_batch_k20_fast` and `cpu_k20_fast_two_process`. The best throughput run reached 19.46 aggregate FPS, but with 506.71 ms average E2E latency and 733.00 ms P95 E2E latency. This is not the best live-feed operating point.

### 6.2 Lowest average E2E latency

| variant                     | buffer_mode   | workers   | backpressure_mode   |   max_pending_age_ms |   target_output_fps_per_camera |   aggregate_output_fps |   avg_e2e_ms |   p95_e2e_ms |   avg_post_ms |   avg_queue_infer_to_post_ms |
|:----------------------------|:--------------|:----------|:--------------------|---------------------:|-------------------------------:|-----------------------:|-------------:|-------------:|--------------:|-----------------------------:|
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               300.00 |                           2.00 |                  14.15 |       185.88 |       263.03 |        113.84 |                         7.43 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           2.00 |                  13.17 |       202.87 |       398.39 |        123.99 |                        12.61 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  16.84 |       226.42 |       284.85 |        141.21 |                         9.13 |
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               300.00 |                           3.00 |                  16.00 |       247.43 |       327.08 |        150.75 |                        14.15 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               350.00 |                           3.00 |                  16.00 |       250.22 |       327.28 |        157.44 |                         7.49 |
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               350.00 |                           3.00 |                  15.95 |       250.86 |       352.87 |        151.69 |                        14.47 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  15.86 |       251.47 |       347.18 |        159.15 |                         9.67 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               500.00 |                           3.00 |                  16.01 |       254.06 |       335.35 |        157.60 |                        13.52 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  15.91 |       254.54 |       338.21 |        160.13 |                        10.11 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  15.84 |       257.51 |       412.08 |        159.07 |                        11.66 |

GPU fullres NMS dominates the latency ranking. The first 10 entries are all `gpu_nms_fullres_two_process`, which confirms that GPU full-resolution NMS is the strongest candidate for live operation.

### 6.3 Lowest P95 E2E latency

| variant                     | buffer_mode   | workers   | backpressure_mode   |   max_pending_age_ms |   target_output_fps_per_camera |   aggregate_output_fps |   avg_e2e_ms |   p95_e2e_ms |   avg_post_ms |   avg_queue_infer_to_post_ms |
|:----------------------------|:--------------|:----------|:--------------------|---------------------:|-------------------------------:|-----------------------:|-------------:|-------------:|--------------:|-----------------------------:|
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               300.00 |                           2.00 |                  14.15 |       185.88 |       263.03 |        113.84 |                         7.43 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  16.84 |       226.42 |       284.85 |        141.21 |                         9.13 |
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               300.00 |                           3.00 |                  16.00 |       247.43 |       327.08 |        150.75 |                        14.15 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               350.00 |                           3.00 |                  16.00 |       250.22 |       327.28 |        157.44 |                         7.49 |
| cpu_k20_fast_two_process    | latest        | I1/P5     | soft                |               300.00 |                           2.00 |                  13.34 |       263.84 |       332.00 |        199.19 |                         4.18 |
| optimized_batch_k20_fast    | latest        | I1/P5     | soft                |               350.00 |                           3.00 |                  16.66 |       272.06 |       333.25 |        205.41 |                         3.98 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               500.00 |                           3.00 |                  16.01 |       254.06 |       335.35 |        157.60 |                        13.52 |
| cpu_k20_fast_two_process    | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  16.60 |       273.93 |       336.27 |        206.63 |                         5.04 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  15.91 |       254.54 |       338.21 |        160.13 |                        10.11 |
| cpu_k20_fast_two_process    | latest        | I1/P5     | soft                |               500.00 |                           3.00 |                  16.61 |       271.54 |       339.66 |        205.66 |                         4.40 |

P95 latency tells the same story: GPU fullres is best at controlling tail latency, while CPU optimized variants appear slightly lower in the ranking but not at the top.

### 6.4 Lowest postprocessing time

| variant                     | buffer_mode   | workers   | backpressure_mode   |   max_pending_age_ms |   target_output_fps_per_camera |   aggregate_output_fps |   avg_e2e_ms |   p95_e2e_ms |   avg_post_ms |   avg_queue_infer_to_post_ms |
|:----------------------------|:--------------|:----------|:--------------------|---------------------:|-------------------------------:|-----------------------:|-------------:|-------------:|--------------:|-----------------------------:|
| gpu_nms_fullres_two_process | latest        | I1/P1     | off                 |               300.00 |                           0.00 |                  11.46 |       348.09 |       735.96 |         79.76 |                       206.98 |
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               300.00 |                           2.00 |                  14.15 |       185.88 |       263.03 |        113.84 |                         7.43 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           2.00 |                  13.17 |       202.87 |       398.39 |        123.99 |                        12.61 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  16.84 |       226.42 |       284.85 |        141.21 |                         9.13 |
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               300.00 |                           3.00 |                  16.00 |       247.43 |       327.08 |        150.75 |                        14.15 |
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               500.00 |                           3.00 |                  16.03 |       258.49 |       364.02 |        150.87 |                        23.03 |
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               350.00 |                           3.00 |                  15.95 |       250.86 |       352.87 |        151.69 |                        14.47 |
| cpu_k20_fast_two_process    | latest        | I1/P1     | off                 |               300.00 |                           0.00 |                   6.25 |       328.12 |       655.55 |        154.94 |                       126.05 |
| optimized_batch_k20_fast    | latest        | I1/P1     | off                 |               300.00 |                           0.00 |                   6.21 |       340.72 |       584.34 |        156.25 |                       136.32 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               350.00 |                           3.00 |                  16.00 |       250.22 |       327.28 |        157.44 |                         7.49 |

The absolute lowest postprocess time was `gpu_nms_fullres_two_process` with `I1/P1`, no backpressure and no throttle. However, this run is not the best E2E setup because queue pressure is higher. The best live setup uses soft backpressure and output throttling, not just minimum isolated postprocess time.

---

## 7. Best balanced configurations

The balanced ranking better captures the live-feed objective because it penalizes both average latency and queue pressure. The top configurations are:

| variant                     | buffer_mode   | workers   | backpressure_mode   |   max_pending_age_ms |   target_output_fps_per_camera |   aggregate_output_fps |   avg_e2e_ms |   p95_e2e_ms |   avg_post_ms |   avg_queue_infer_to_post_ms |   balanced_score |
|:----------------------------|:--------------|:----------|:--------------------|---------------------:|-------------------------------:|-----------------------:|-------------:|-------------:|--------------:|-----------------------------:|-----------------:|
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               300.00 |                           2.00 |                  14.15 |       185.88 |       263.03 |        113.84 |                         7.43 |            71.62 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  16.84 |       226.42 |       284.85 |        141.21 |                         9.13 |            77.71 |
| optimized_batch_k20_fast    | latest        | I1/P5     | soft                |               350.00 |                           3.00 |                  16.66 |       272.06 |       333.25 |        205.41 |                         3.98 |           125.36 |
| cpu_k20_fast_two_process    | latest        | I1/P5     | soft                |               500.00 |                           3.00 |                  16.61 |       271.54 |       339.66 |        205.66 |                         4.40 |           131.23 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               350.00 |                           3.00 |                  16.00 |       250.22 |       327.28 |        157.44 |                         7.49 |           131.30 |
| cpu_k20_fast_two_process    | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  16.60 |       273.93 |       336.27 |        206.63 |                         5.04 |           135.19 |
| optimized_batch_k20_fast    | latest        | I1/P5     | soft                |               500.00 |                           3.00 |                  16.55 |       276.10 |       342.32 |        208.75 |                         4.88 |           140.71 |
| cpu_k20_fast_two_process    | latest        | I1/P5     | soft                |               350.00 |                           3.00 |                  16.55 |       273.85 |       343.80 |        207.23 |                         5.23 |           140.98 |
| optimized_batch_k20_fast    | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  16.66 |       277.07 |       348.08 |        209.15 |                         6.66 |           151.17 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  15.91 |       254.54 |       338.21 |        160.13 |                        10.11 |           155.94 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  15.86 |       251.47 |       347.18 |        159.15 |                         9.67 |           156.19 |
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               300.00 |                           3.00 |                  16.00 |       247.43 |       327.08 |        150.75 |                        14.15 |           161.72 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               500.00 |                           3.00 |                  16.01 |       254.06 |       335.35 |        157.60 |                        13.52 |           169.14 |
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               350.00 |                           3.00 |                  15.95 |       250.86 |       352.87 |        151.69 |                        14.47 |           180.60 |
| cpu_k20_fast_two_process    | latest        | I1/P5     | soft                |               300.00 |                           2.00 |                  13.34 |       263.84 |       332.00 |        199.19 |                         4.18 |           183.98 |

The top two candidates are both GPU fullres configurations. The best low-latency setup is `I1/P3` with throttle 2 FPS/camera. The best higher-output live setup is `I1/P5` with throttle 3 FPS/camera.

---

## 8. Common-setup comparison

To compare postprocessing variants under the same architecture, the following table uses the common latest-buffer setup:

```text
buffer_mode = latest
workers = I1/P5
backpressure = soft
max_pending_age_ms = 300
target_output_fps_per_camera = 3
label = latest_grid
```

| variant                     |   aggregate_output_fps |   avg_e2e_ms |   p95_e2e_ms |   avg_post_ms |   avg_queue_infer_to_post_ms |   balanced_score |
|:----------------------------|-----------------------:|-------------:|-------------:|--------------:|-----------------------------:|-----------------:|
| cpu_k20_fast_two_process    |                  16.60 |       273.93 |       336.27 |        206.63 |                         5.04 |           135.19 |
| optimized_batch_k20_fast    |                  16.66 |       277.07 |       348.08 |        209.15 |                         6.66 |           151.17 |
| gpu_nms_fullres_two_process |                  15.91 |       254.54 |       338.21 |        160.13 |                        10.11 |           155.94 |
| migraphx_nms                |                  11.71 |       630.73 |       893.20 |        410.78 |                       163.49 |          1660.61 |
| migraphx_nms_k20            |                  11.97 |       636.34 |       932.38 |        401.87 |                       175.82 |          1742.11 |
| standard                    |                   3.95 |      1530.12 |      1915.44 |       1217.23 |                       247.85 |          3648.02 |

Under this fixed setup, the CPU optimized variants have very strong throughput and competitive P95 latency, while GPU fullres keeps the lowest average E2E and lowest postprocess time. This confirms that the choice depends on objective: maximum FPS favors optimized CPU, but live latency favors GPU fullres.

---

## 9. Backpressure and throttle analysis

### 9.1 Median behavior by backpressure/throttle mode

| backpressure_mode   |   target_output_fps_per_camera |   runs |   median_fps |   median_avg_e2e_ms |   median_p95_e2e_ms |   median_queue_ms |
|:--------------------|-------------------------------:|-------:|-------------:|--------------------:|--------------------:|------------------:|
| off                 |                           0.00 |     24 |        10.99 |              516.61 |              772.97 |            148.40 |
| soft                |                           0.00 |     48 |        13.08 |              563.71 |              802.21 |            151.94 |
| soft                |                           2.00 |     24 |        12.66 |              494.61 |              705.99 |            104.28 |
| soft                |                           3.00 |    103 |        12.42 |              555.95 |              808.17 |            152.27 |
| soft                |                           4.00 |     24 |        13.19 |              550.46 |              785.69 |            157.50 |
| strict              |                           0.00 |     24 |        13.97 |              719.92 |             1061.86 |            314.64 |

Strict backpressure tends to produce strong throughput in some cases, but its median E2E and P95 latencies are worse. Soft backpressure with moderate throttle is better aligned with live-feed behavior because it allows stale pending work to be overridden and keeps the output fresher.

### 9.2 Best setup per backpressure/throttle group

| backpressure_mode   |   target_output_fps_per_camera | variant                     | workers   |   aggregate_output_fps |   avg_e2e_ms |   p95_e2e_ms |   avg_post_ms |   avg_queue_infer_to_post_ms |   balanced_score |
|:--------------------|-------------------------------:|:----------------------------|:----------|-----------------------:|-------------:|-------------:|--------------:|-----------------------------:|-----------------:|
| off                 |                           0.00 | gpu_nms_fullres_two_process | I1/P5     |                  16.43 |       295.64 |       386.26 |        189.77 |                         9.49 |           207.65 |
| soft                |                           0.00 | gpu_nms_fullres_two_process | I1/P5     |                  16.73 |       285.83 |       402.42 |        179.59 |                        12.26 |           213.81 |
| soft                |                           2.00 | gpu_nms_fullres_two_process | I1/P3     |                  14.15 |       185.88 |       263.03 |        113.84 |                         7.43 |            71.62 |
| soft                |                           3.00 | gpu_nms_fullres_two_process | I1/P5     |                  16.84 |       226.42 |       284.85 |        141.21 |                         9.13 |            77.71 |
| soft                |                           4.00 | gpu_nms_fullres_two_process | I1/P5     |                  16.24 |       274.72 |       412.97 |        171.70 |                        11.79 |           215.44 |
| strict              |                           0.00 | gpu_nms_fullres_two_process | I1/P5     |                  16.67 |       297.42 |       376.88 |        176.00 |                        28.23 |           293.65 |

The best setup for each backpressure/throttle group is always `gpu_nms_fullres_two_process`. This is a strong signal that GPU fullres is the most robust live-feed family even when scheduling policy changes.

---

## 10. Worker architecture analysis

The worker sweep shows that adding workers is not always beneficial. The best latency results occur with a small number of inference workers and a moderate number of postprocess workers.

Median results by worker configuration:

| workers   |   runs |   median_fps |   median_avg_e2e_ms |   median_p95_e2e_ms |   median_post_ms |   median_queue_ms |
|:----------|-------:|-------------:|--------------------:|--------------------:|-----------------:|------------------:|
| I1/P1     |      6 |         5.10 |              399.81 |              763.67 |           199.69 |            162.09 |
| I1/P5     |     74 |        13.01 |              515.83 |              753.40 |           285.51 |            153.05 |
| I1/P3     |     54 |        11.66 |              524.21 |              739.18 |           248.62 |            153.87 |
| I2/P5     |     65 |        12.51 |              612.24 |              831.16 |           384.54 |            150.14 |
| I2/P6     |     48 |        14.20 |              633.25 |              866.84 |           394.09 |            151.80 |

The strongest live candidates are `I1/P3` and `I1/P5`. Higher worker counts can increase aggregate FPS in some configurations, but they often increase GPU contention, queue pressure and E2E latency. This is especially visible for GPU-based variants where MIGraphX inference and PyTorch/ROCm postprocessing share the same GPU resources.

---

## 11. GPU full-resolution NMS hyperparameter analysis

The GPU fullres sweep varied NMS implementation, full-resolution radius and GPU compute dtype.

### 11.1 NMS implementation and radius

| nms_impl   |   nms_radius_fullres |   runs |   median_fps |   best_fps |   median_avg_e2e_ms |   best_avg_e2e_ms |   median_p95_e2e_ms |   median_post_ms |
|:-----------|---------------------:|-------:|-------------:|-----------:|--------------------:|------------------:|--------------------:|-----------------:|
| separable  |                    6 |     38 |        16.34 |      18.13 |              372.54 |            185.88 |              543.23 |           195.85 |
| 2d         |                    3 |      4 |        12.60 |      17.21 |              579.62 |            226.42 |              874.06 |           383.48 |
| separable  |                    3 |      4 |        12.15 |      17.35 |              602.99 |            257.51 |              940.86 |           390.96 |
| separable  |                    8 |      4 |        12.13 |      16.23 |              612.54 |            251.47 |              947.82 |           399.21 |
| 2d         |                    6 |      4 |        10.91 |      14.82 |              700.62 |            270.59 |             1054.01 |           443.80 |
| 2d         |                    8 |      4 |         9.73 |      12.62 |              796.07 |            326.02 |             1189.07 |           487.46 |

The most robust configuration is `separable` with radius 6. It has the best median behavior and produced the absolute best latency setup. Some `2d/r3` configurations achieved excellent best-case performance, but the broader median behavior still favors `separable/r6`.

### 11.2 GPU compute dtype

| gpu_compute_dtype   |   nms_radius_fullres |   runs |   median_fps |   best_fps |   median_avg_e2e_ms |   best_avg_e2e_ms |   median_p95_e2e_ms |   median_post_ms |
|:--------------------|---------------------:|-------:|-------------:|-----------:|--------------------:|------------------:|--------------------:|-----------------:|
| float32             |                    3 |      4 |        17.02 |      17.35 |              301.46 |            226.42 |              450.18 |           188.92 |
| float32             |                    6 |     38 |        16.34 |      18.13 |              367.74 |            185.88 |              538.39 |           184.68 |
| float32             |                    8 |      4 |        14.24 |      16.23 |              371.55 |            251.47 |              590.00 |           229.73 |
| float16             |                    3 |      4 |         8.33 |       8.45 |              823.74 |            805.79 |             1305.88 |           549.84 |
| float16             |                    6 |      4 |         8.12 |       8.40 |              855.51 |            801.64 |             1299.13 |           565.02 |
| float16             |                    8 |      4 |         7.99 |       8.40 |              868.04 |            807.99 |             1324.29 |           574.93 |

Float32 is clearly better in this setup. Float16 GPU postprocessing substantially reduces throughput and increases latency, likely because this workload or backend path does not benefit from half precision and may pay conversion/scheduling costs.

### 11.3 Best GPU fullres configurations

| variant                     | buffer_mode   | workers   | backpressure_mode   |   max_pending_age_ms |   target_output_fps_per_camera |   aggregate_output_fps |   avg_e2e_ms |   p95_e2e_ms |   avg_post_ms |   avg_queue_infer_to_post_ms | nms_impl   |   nms_radius_fullres | gpu_compute_dtype   |   balanced_score |
|:----------------------------|:--------------|:----------|:--------------------|---------------------:|-------------------------------:|-----------------------:|-------------:|-------------:|--------------:|-----------------------------:|:-----------|---------------------:|:--------------------|-----------------:|
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               300.00 |                           2.00 |                  14.15 |       185.88 |       263.03 |        113.84 |                         7.43 | separable  |                    6 | float32             |            71.62 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  16.84 |       226.42 |       284.85 |        141.21 |                         9.13 | 2d         |                    3 | float32             |            77.71 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               350.00 |                           3.00 |                  16.00 |       250.22 |       327.28 |        157.44 |                         7.49 | separable  |                    6 | float32             |           131.30 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  15.91 |       254.54 |       338.21 |        160.13 |                        10.11 | separable  |                    6 | float32             |           155.94 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  15.86 |       251.47 |       347.18 |        159.15 |                         9.67 | separable  |                    8 | float32             |           156.19 |
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               300.00 |                           3.00 |                  16.00 |       247.43 |       327.08 |        150.75 |                        14.15 | separable  |                    6 | float32             |           161.72 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               500.00 |                           3.00 |                  16.01 |       254.06 |       335.35 |        157.60 |                        13.52 | separable  |                    6 | float32             |           169.14 |
| gpu_nms_fullres_two_process | latest        | I1/P3     | soft                |               350.00 |                           3.00 |                  15.95 |       250.86 |       352.87 |        151.69 |                        14.47 | separable  |                    6 | float32             |           180.60 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           2.00 |                  13.17 |       202.87 |       398.39 |        123.99 |                        12.61 | separable  |                    6 | float32             |           201.76 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           3.00 |                  15.84 |       257.51 |       412.08 |        159.07 |                        11.66 | separable  |                    3 | float32             |           205.04 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | off                 |               300.00 |                           0.00 |                  16.43 |       295.64 |       386.26 |        189.77 |                         9.49 | separable  |                    6 | float32             |           207.65 |
| gpu_nms_fullres_two_process | latest        | I1/P5     | soft                |               300.00 |                           0.00 |                  16.73 |       285.83 |       402.42 |        179.59 |                        12.26 | separable  |                    6 | float32             |           213.81 |

Final GPU fullres recommendation:

```text
Primary low-latency: I1/P3, soft BP 300 ms, throttle 2 fps/cam, separable/r6, float32
Balanced live-output: I1/P5, soft BP 300 ms, throttle 3 fps/cam, float32
Robust default hparams: separable NMS, radius 6, gpu_compute_dtype=float32
```

---

## 12. CPU `k/threshold` analysis

The CPU-specific grid varied `max_keypoints` and `threshold` for `standard`, `optimized_batch_k20_fast`, and `cpu_k20_fast_two_process`.

| variant                  |   max_keypoints |   threshold |   aggregate_output_fps |   avg_e2e_ms |   p95_e2e_ms |   avg_post_ms |   avg_queue_infer_to_post_ms |
|:-------------------------|----------------:|------------:|-----------------------:|-------------:|-------------:|--------------:|-----------------------------:|
| cpu_k20_fast_two_process |              10 |        0.10 |                  16.85 |       494.93 |       737.49 |        284.11 |                       138.44 |
| cpu_k20_fast_two_process |              10 |        0.15 |                  16.62 |       453.06 |       718.35 |        274.99 |                       108.79 |
| cpu_k20_fast_two_process |              20 |        0.15 |                  16.77 |       445.61 |       729.10 |        270.72 |                       106.51 |
| optimized_batch_k20_fast |              10 |        0.10 |                  16.88 |       360.46 |       614.36 |        243.24 |                        53.55 |
| optimized_batch_k20_fast |              10 |        0.15 |                  17.12 |       470.99 |       711.35 |        279.77 |                       124.54 |
| optimized_batch_k20_fast |              20 |        0.15 |                  16.69 |       489.33 |       735.12 |        286.58 |                       131.36 |
| standard                 |              10 |        0.10 |                   4.06 |      1498.87 |      2286.74 |       1188.41 |                       242.06 |
| standard                 |              10 |        0.15 |                   4.04 |      1516.37 |      2156.27 |       1193.40 |                       257.91 |
| standard                 |              20 |        0.15 |                   3.92 |      1585.83 |      2544.67 |       1229.74 |                       293.89 |

The CPU optimized variants remain much better than standard. However, CPU `k/threshold` tuning does not close the latency gap to GPU fullres. The best CPU-tuned rows are useful as CPU fallback configurations, but not as the primary live-feed choice.

---

## 13. Buffer and queue-mode analysis

### 13.1 Buffer mode

| buffer_mode   |   runs |   median_fps |   median_avg_e2e_ms |   median_p95_e2e_ms |   median_queue_ms |
|:--------------|-------:|-------------:|--------------------:|--------------------:|------------------:|
| latest        |    247 |        12.62 |              556.30 |              799.71 |            153.73 |
| queue         |      3 |         4.12 |            14556.98 |            22935.86 |           7484.12 |

### 13.2 Queue policy

| queue_policy   |   runs |   median_fps |   median_avg_e2e_ms |   median_p95_e2e_ms |   median_queue_ms |
|:---------------|-------:|-------------:|--------------------:|--------------------:|------------------:|
| block          |      1 |         4.12 |            17866.67 |            18752.93 |           7484.12 |
| drop           |    249 |        12.62 |              556.56 |              800.44 |            153.90 |

The reliable conclusions should be based on `latest` buffer mode. FIFO queue baselines are underrepresented because most queue-baseline entries were missing/invalid in this plotting bundle. The single valid `block` queue result shows extremely poor latency and should not be treated as a recommended mode.

For live monitoring, `latest` buffering is the correct architectural choice because the system should prioritize fresh frames over processing every queued frame.

---

## 14. MIGraphX NMS observations

The MIGraphX NMS variants are not competitive with GPU fullres NMS in this grid. Their median aggregate FPS is around 11.6–11.7 FPS and median E2E latency is around 635–638 ms. They are significantly better than the original standard baseline, but they do not beat the best CPU optimized or PyTorch/ROCm GPU fullres variants.

The likely interpretation is that compiling only the NMS portion is not enough to remove the main postprocessing bottlenecks. The overall pose postprocessing pipeline still includes resize, extraction, grouping and CPU/GPU synchronization overheads. Therefore, MIGraphX NMS should be treated as an experimental path, not as the preferred current implementation.

---

## 15. Final recommendations

### 15.1 Recommended live-feed configuration

For the current 10-camera live simulation objective, the recommended configuration family is:

```text
variant = gpu_nms_fullres_two_process
buffer_mode = latest
backpressure_mode = soft
max_pending_age_ms = 300
gpu_compute_dtype = float32
nms_impl = separable
nms_radius_fullres = 6
```

Recommended worker/throttle candidates:

| Use case | Setup | Reason |
|---|---|---|
| Lowest latency | `I1/P3`, throttle 2 fps/cam | Best observed avg and P95 E2E latency |
| Balanced live output | `I1/P5`, throttle 3 fps/cam | Higher output rate while keeping latency low |
| Conservative default | `I1/P5`, throttle 2–3 fps/cam | Stable, simple and easy to compare against previous runs |
| Max throughput reference | CPU fast / optimized batch strict BP | Useful benchmark, but not preferred for live latency |
