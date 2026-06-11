"""Command-line interface for the stream simulator."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .defaults import DEFAULT_VIDEO_CYCLE
from .runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate 10 live camera streams through preprocess -> MIGraphX -> postprocess pipeline."
    )
    parser.add_argument("--model", default="pose_model1_fp16_ref1.mxr")
    parser.add_argument(
        "--variant",
        default="gpu_nms_fullres_two_process",
        help=(
            "Postprocess variant. Examples: standard, optimized_batch_k20_fast, "
            "lowres_cpu_group, gpu_nms_fullres_two_process, gpu_nms_lowres_two_process, "
            "migraphx-nms, migraphx-nms-k20."
        ),
    )
    parser.add_argument("--videos", nargs="*", default=DEFAULT_VIDEO_CYCLE)
    parser.add_argument("--num-cameras", type=int, default=10)
    parser.add_argument("--frames-per-camera", type=int, default=100, help="0 means run until interrupted/duration.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Optional wall-clock duration per camera. 0 disables duration limit.")
    parser.add_argument("--realtime", action="store_true", help="Throttle each simulated camera to --camera-fps.")
    parser.add_argument("--camera-fps", type=float, default=24.0)
    parser.add_argument("--queue-policy", choices=["drop", "block"], default="drop")
    parser.add_argument("--buffer-mode", choices=["latest", "queue"], default="latest", help="latest keeps one newest-frame slot per camera between stages; queue preserves the original FIFO queues.")
    parser.add_argument(
        "--disable-backpressure",
        action="store_true",
        help="Legacy alias for --backpressure-mode off.",
    )
    parser.add_argument(
        "--backpressure-mode",
        choices=["off", "strict", "soft"],
        default="strict",
        help=(
            "Backpressure policy for --buffer-mode latest. "
            "'off': never skip cameras (max throughput, results may be overwritten). "
            "'strict': skip a camera while its post_pending flag is set (original behaviour). "
            "'soft': skip only while the pending result is fresher than --max-pending-age-ms; "
            "allows re-inference once a result has been sitting too long. "
            "--disable-backpressure is a legacy alias for 'off'."
        ),
    )
    parser.add_argument(
        "--max-pending-age-ms",
        type=float,
        default=300.0,
        help=(
            "Used with --backpressure-mode soft. A camera whose pending postprocess result "
            "is older than this threshold (ms) is eligible for re-inference even though "
            "post_pending is still set. Prevents slow post workers from starving cameras. "
            "Default: 300 ms."
        ),
    )
    parser.add_argument(
        "--target-output-fps-per-camera",
        type=float,
        default=0.0,
        help=(
            "When > 0, the inference scheduler skips cameras that were fully postprocessed "
            "more recently than 1/fps seconds ago. Useful to cap per-camera processing rate "
            "and ensure fair share across cameras in mixed-difficulty scenes. 0 = disabled."
        ),
    )

    parser.add_argument("--infer-workers", type=int, default=1)
    parser.add_argument("--post-workers", type=int, default=1)
    parser.add_argument(
        "--mp-start-method",
        choices=["spawn", "fork", "forkserver"],
        default="spawn",
        help="Multiprocessing start method. Keep spawn for normal runs; fork is useful for rocprofv3 direct profiling to avoid child re-exec profiler registration issues.",
    )
    parser.add_argument(
        "--migraphx-batch-size",
        type=int,
        default=1,
        help=(
            "Batch size used by MIGraphX inference workers. Use 1 for old behavior. "
            "For static batch MXR models, set this to the compiled batch size, e.g. 2/4/8."
        ),
    )
    parser.add_argument(
        "--migraphx-batch-timeout-ms",
        type=float,
        default=0.0,
        help=(
            "Maximum time an inference worker waits to fill a MIGraphX batch. "
            "Use a small value such as 2-8 ms for live simulation."
        ),
    )
    parser.add_argument("--preprocess-queue-size", type=int, default=30)
    parser.add_argument("--postprocess-queue-size", type=int, default=30)

    parser.add_argument("--target-width", type=int, default=968)
    parser.add_argument("--target-height", type=int, default=544)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--shared-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--shared-map-slots",
        type=int,
        default=0,
        help="Latest-mode only: preallocate this many shared-memory heatmap/PAF slots between inference and postprocess. 0 keeps Queue pickle/copy.",
    )
    parser.add_argument(
        "--shared-input-slots",
        type=int,
        default=0,
        help=(
            "Latest-mode only: preallocate this many shared-memory preprocessed input slots "
            "between camera/preprocess and inference. Use --num-cameras so each camera "
            "has one stable slot. 0 keeps old Queue pickle/copy behavior."
        ),
    )
    parser.add_argument(
        "--shared-input-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="dtype used for shared camera->inference input slots. float32 matches current preprocess output.",
    )

    parser.add_argument("--torch-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--max-keypoints", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nms-radius-fullres", type=int, default=6)
    parser.add_argument("--nms-radius-lowres", type=int, default=1)
    parser.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    parser.add_argument("--gpu-compute-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--prealloc-resize-buffers",
        action="store_true",
        help="Reuse persistent cv2.resize dst buffers inside each postprocess worker when supported by OpenCV.",
    )
    parser.add_argument(
        "--gpu-nms-batch-size",
        type=int,
        default=1,
        help="Latest-mode gpu_nms_fullres_two_process only: batch this many frames per post worker for Torch max_pool NMS. 1 disables batching.",
    )
    parser.add_argument(
        "--gpu-nms-batch-timeout-ms",
        type=float,
        default=0.0,
        help="Maximum wait to fill a gpu_nms batch before running it. Keep small, e.g. 2-8 ms, for live feeds.",
    )
    parser.add_argument(
        "--migraphx-nms-mxr",
        default="",
        help="Optional explicit compiled MIGraphX NMS .mxr path for migraphx-nms variants.",
    )
    parser.add_argument(
        "--migraphx-nms-cache-dir",
        default="models/nms_fullres_cache",
        help="Directory containing heatmap_nms_head_<H>x<W>.mxr files.",
    )
    parser.add_argument(
        "--compile-migraphx-nms",
        action="store_true",
        help="Compile the stream-resolution MIGraphX NMS head before starting the stream.",
    )
    parser.add_argument("--force-compile-migraphx-nms", action="store_true")
    parser.add_argument("--keep-migraphx-nms-onnx", action="store_true")
    parser.add_argument("--exhaustive-tune-migraphx-nms", action="store_true")

    parser.add_argument(
        "--grid-video",
        default="",
        help=(
            "Optional output path for a single security-monitor-style grid video. "
            "When set, postprocessed frames are drawn and concatenated into one video."
        ),
    )
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--grid-cell-width", type=int, default=480)
    parser.add_argument("--grid-cell-height", type=int, default=270)
    parser.add_argument("--grid-video-fps", type=float, default=10.0)
    parser.add_argument("--grid-video-codec", default="mp4v")
    parser.add_argument("--grid-queue-size", type=int, default=256)

    parser.add_argument("--pin-cpus", action="store_true", help="Pin each camera, inference worker, and postprocess worker to distinct CPU cores.")
    parser.add_argument("--pin-camera-base", type=int, default=0, help="First CPU core for camera workers when --pin-cpus is set.")
    parser.add_argument("--pin-inference-base", type=int, default=10, help="First CPU core for inference workers when --pin-cpus is set.")
    parser.add_argument("--pin-post-base", type=int, default=12, help="First CPU core for postprocess workers when --pin-cpus is set.")
    parser.add_argument("--pin-all-threads", action="store_true", help="Also pin existing native threads under /proc/<pid>/task for each worker after startup.")
    parser.add_argument("--worker-threads", type=int, default=1, help="Set OpenCV/OpenMP/OpenBLAS/NumExpr/PyTorch CPU thread pools per worker. Default: 1.")
    parser.add_argument("--warmup-s", type=float, default=0.0, help="Discard output rows whose postprocess completion is within this many seconds of the first output row.")
    parser.add_argument("--warmup-output-frames", type=int, default=0, help="Discard this many additional earliest output rows before computing the summary.")

    parser.add_argument(
        "--profile-system",
        action="store_true",
        help="Collect parent-side per-PID CPU/memory, affinity, per-core CPU, GPU busy, and VRAM stats.",
    )
    parser.add_argument(
        "--profile-interval-s",
        type=float,
        default=0.1,
        help="Sampling interval for --profile-system. Default: 0.1 s.",
    )
    parser.add_argument(
        "--report-affinity",
        action="store_true",
        help="Print worker CPU affinity after all child processes are started.",
    )
    parser.add_argument(
        "--roctx",
        action="store_true",
        help="Emit ROCTx ranges around preprocess, MIGraphX load/run/decode, and postprocess work for rocprofv3 marker traces.",
    )
    parser.add_argument(
        "--trace-log-every",
        type=int,
        default=0,
        help="Print per-worker timing trace every N processed frames/batches. 0 disables these verbose logs.",
    )
    parser.add_argument(
        "--allow-ptrace-attach",
        action="store_true",
        help="Let same-user tools such as rocprofv3 --attach attach to worker processes on Yama ptrace_scope systems.",
    )



    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--detailed-csv", default="outputs/stream_10cam_detailed.csv")
    parser.add_argument("--summary-json", default="outputs/stream_10cam_summary.json")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None):
    return build_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    return run(parse_args(argv))
