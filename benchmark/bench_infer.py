"""
Standalone inference benchmark — za rocprofv3 profiling.
Nema multiprocessing, nema spawn — samo model load + inference loop.

Koristiti:
  rocprofv3 --hip-trace --kernel-trace -- python benchmark/bench_infer.py --iters 200
"""
import argparse
import time
import numpy as np
import migraphx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="models/pose_model1_fp16_ref1.mxr")
    ap.add_argument("--iters",  type=int, default=100, help="broj inference iteracija")
    ap.add_argument("--warmup", type=int, default=10,  help="warmup iteracija (ne mjeri se)")
    ap.add_argument("--roctx",  action="store_true",   help="ROCTx markeri na timeline-u")
    args = ap.parse_args()

    # --- ROCTx (opcionalno) ---
    roctx = None
    if args.roctx:
        try:
            import ctypes
            # Probaj novi ROCm 7.x SDK (v3 API) prije starog v1 shima
            try:
                _lib = ctypes.CDLL("librocprofiler-sdk-roctx.so")
            except OSError:
                _lib = ctypes.CDLL("libroctx64.so")
            _lib.roctxRangePushA.restype  = ctypes.c_int
            _lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
            _lib.roctxRangePop.restype    = ctypes.c_int
            _lib.roctxRangePop.argtypes   = []
            roctx = _lib
            print("[bench_infer] ROCTx učitan OK")
        except Exception as e:
            print(f"[bench_infer] ROCTx nije dostupan: {e}")

    def push(name):
        if roctx: roctx.roctxRangePushA(name.encode())
    def pop():
        if roctx: roctx.roctxRangePop()

    # --- učitaj model ---
    print(f"[bench_infer] Učitavam model: {args.model} ...", flush=True)
    t0 = time.perf_counter()
    model = migraphx.load(args.model)
    print(f"[bench_infer] Model učitan za {time.perf_counter()-t0:.2f}s", flush=True)

    # ulazni shape: [1, 3, 544, 968] fp16
    dummy = np.random.rand(1, 3, 544, 968).astype(np.float16)

    # --- warmup ---
    print(f"[bench_infer] Warmup ({args.warmup} iter) ...", flush=True)
    for _ in range(args.warmup):
        model.run({"input": migraphx.argument(dummy)})

    # --- benchmark loop ---
    print(f"[bench_infer] Benchmark ({args.iters} iter) ...", flush=True)
    times_cast = []
    times_run  = []
    times_dec  = []

    t_total = time.perf_counter()
    for i in range(args.iters):
        # 1. dtype cast
        push("dtype_cast")
        tc0 = time.perf_counter()
        inp = dummy.astype(np.float16)          # simulira float32→float16
        tc1 = time.perf_counter()
        pop()

        # 2. GPU inference
        push("migraphx_run")
        tr0 = time.perf_counter()
        out = model.run({"input": migraphx.argument(inp)})
        tr1 = time.perf_counter()
        pop()

        # 3. decode outputs (numpy view)
        push("decode_outputs")
        td0 = time.perf_counter()
        results = [np.array(o) for o in out]
        td1 = time.perf_counter()
        pop()

        times_cast.append((tc1 - tc0) * 1000)
        times_run .append((tr1 - tr0) * 1000)
        times_dec .append((td1 - td0) * 1000)

    elapsed = time.perf_counter() - t_total

    # --- ispis ---
    def stats(arr):
        a = np.array(arr)
        return f"avg={a.mean():.2f}ms  p50={np.percentile(a,50):.2f}ms  p95={np.percentile(a,95):.2f}ms  min={a.min():.2f}ms  max={a.max():.2f}ms"

    print("\n=== BENCHMARK REZULTATI ===")
    print(f"  Itera:          {args.iters}")
    print(f"  Ukupno:         {elapsed*1000:.1f} ms")
    print(f"  Throughput:     {args.iters/elapsed:.1f} infer/s")
    print(f"  dtype_cast:     {stats(times_cast)}")
    print(f"  migraphx_run:   {stats(times_run)}")
    print(f"  decode_outputs: {stats(times_dec)}")
    total = np.array(times_cast) + np.array(times_run) + np.array(times_dec)
    print(f"  per-iter total: {stats(total)}")

if __name__ == "__main__":
    main()
