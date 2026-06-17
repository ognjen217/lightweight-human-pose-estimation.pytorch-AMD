# Heatmap TopK Branch Ablation Summary

The heatmap branch ablation was executed on the validated accuracy-preserving configuration: batch size 4, 18 heatmap channels, input heatmap resolution `68x121`, full-resolution target `1080x1920`, `K=20`, threshold `0.1`, NMS radius `6`, separable NMS, and cubic coefficient `a=-0.75`. The purpose of this experiment was not to change model semantics, but to isolate where the runtime cost comes from inside the full-resolution heatmap path. The measured branch corresponds to the sequence: manual cubic resize, full-resolution NMS pooling, peak masking, flattening, and final TopK candidate extraction.

| Parameter | Value |
|---|---:|
| Batch size | 4 |
| Heatmap channels | 18 |
| Input heatmap shape | `68 x 121` |
| Full-resolution shape | `1080 x 1920` |
| TopK | 20 |
| Threshold | 0.1 |
| NMS radius | 6 |
| NMS implementation | separable |
| Cubic coefficient | -0.75 |
| Benchmark runs | 60 |

The results show that the branch is already expensive at the resize stage. The `resize_only` head takes **46.55 ms avg**, while the full `resize_pool_mask_topk` head takes **64.32 ms avg**. This means that manual cubic upsampling alone accounts for roughly **72.4%** of the final heatmap TopK branch runtime. Adding separable pooling increases the average time to **68.51 ms**, while the mask-producing variant reaches **72.85 ms**. However, the latter two variants return full-resolution tensors, so their numbers are not directly additive against the final TopK variant, which returns only small TopK outputs.

| Ablation mode | Stage included | Avg ms | P50 ms | P95 ms |
|---|---|---:|---:|---:|
| `resize_only` | Manual cubic resize only | 46.55 | 46.61 | 47.04 |
| `resize_pool` | Resize + separable NMS pooling | 68.51 | 68.63 | 69.16 |
| `resize_pool_mask` | Resize + pooling + peak mask | 72.85 | 72.92 | 73.52 |
| `resize_pool_mask_topk` | Resize + pooling + mask + TopK | 64.32 | 64.71 | 65.15 |

The important interpretation is that the heatmap branch bottleneck is not primarily caused by PAF vectorization or CPU postprocessing. The dominant cost is the creation and processing of the full-resolution heatmap tensor, especially the manual bicubic resize from `68x121` to `1080x1920`. For B4 and 18 channels, this produces a full-resolution intermediate of `4 * 18 * 1080 * 1920 = 149,299,200` elements. The separable pooling/NMS stage also adds significant cost, approximately **22 ms** when comparing `resize_pool` against `resize_only`, although this estimate must be treated carefully because the output contracts differ.

| Derived observation | Value | Interpretation |
|---|---:|---|
| Full-resolution heatmap elements | 149,299,200 | Very large intermediate tensor |
| Resize-only share of final TopK branch | 72.4% | Resize dominates the measured branch |
| Resize → pooling delta | +21.96 ms | Separable full-res pooling is also expensive |
| Pooling → mask delta | +4.34 ms | Masking adds smaller but visible cost |
| Final TopK mode vs resize-only | +17.77 ms | Not strictly additive due to smaller TopK outputs |

The next optimization target should therefore be the full-resolution heatmap branch, not the PAF branch. The most promising directions are: rewriting the manual cubic resize into a form that MIGraphX lowers more efficiently, reducing full-resolution pooling overhead, avoiding unnecessary materialization of full-resolution mask tensors, and later testing a semantics-preserving TopK rewrite. Since all current accuracy-preserving parameters must remain fixed, the optimization should preserve `1080x1920`, `K=20`, threshold `0.1`, radius `6`, and separable NMS semantics while changing only how the same computation is represented in ONNX/MIGraphX.

| Priority | Optimization direction | Reason |
|---:|---|---|
| 1 | Manual cubic resize rewrite | Largest confirmed cost: 46.55 ms |
| 2 | Separable NMS/pooling rewrite | Adds about 22 ms after resize |
| 3 | Mask materialization avoidance | Full-resolution mask tensors are expensive outputs/intermediates |
| 4 | Semantics-preserving TopK rewrite | Still important, but not isolated cleanly by this test |
