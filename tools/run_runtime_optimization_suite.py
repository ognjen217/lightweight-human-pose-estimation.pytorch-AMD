#!/usr/bin/env python3
"""Runtime optimization benchmark pack for the merged MIGraphX live-feed path.

This tool is measurement infrastructure. It reuses simulate_camera_stream.py,
caches per-run summary.json/detailed.csv under outputs/plot_cache/, then emits
baseline_manifest.json, baseline_summary.csv, a decision sheet, and five plots.
"""
from __future__ import annotations

import argparse, csv, json, math, os, statistics, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("outputs/plot_cache/runtime_optimization_suite")
OUT = Path("outputs/runtime_optimization_suite")
VARIANT = "mx_merged_pose_fused_pruned"
MODEL = "models/merged_pose_fused_pruned_batchaware/pose_fused_pruned_batchaware_b{b}_1080x1920_k20_m20_thr0p1_r6_separable.mxr"

FIELDS = [
    "scenario_id","deployment_class","description","status","batch_size","num_cameras","camera_fps",
    "target_output_fps_per_camera","migraphx_batch_timeout_ms","infer_workers","post_workers",
    "steady_fps","steady_fps_per_camera","avg_e2e_ms","p95_e2e_ms","avg_preprocess_ms",
    "avg_queue_pre_to_infer_ms","avg_inference_ms_per_frame","avg_inference_ms_per_batch",
    "avg_decode_ms","avg_queue_infer_to_post_ms","avg_post_ms","avg_real_batch_size","batch_fill_ratio",
    "camera_replaced_before_infer","inference_replaced_before_post","skipped_due_backpressure",
    "camera_replaced_before_infer_ratio","inference_replaced_before_post_ratio","skipped_due_backpressure_ratio",
    "fairness_min_to_max_camera_fps","gpu_avg_pct","vram_avg_mb","total_processed_frames",
    "warmup_discarded_frames","summary_json","detailed_csv","run_log",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def num(x: Any, d: float = 0.0) -> float:
    try:
        y = float(x)
        return d if math.isnan(y) or math.isinf(y) else y
    except Exception:
        return d

def mean(xs):
    xs = [num(x) for x in xs if x not in (None, "")]
    return statistics.fmean(xs) if xs else 0.0

def p95(xs):
    xs = sorted(num(x) for x in xs if x not in (None, ""))
    if not xs: return 0.0
    if len(xs) == 1: return xs[0]
    p = (len(xs) - 1) * 0.95; lo = math.floor(p); hi = math.ceil(p)
    return xs[lo] if lo == hi else xs[lo] * (hi - p) + xs[hi] * (p - lo)

def mkdir(p: Path) -> None: p.mkdir(parents=True, exist_ok=True)
def load_json(p: Path) -> dict: return json.loads(p.read_text(encoding="utf-8"))
def save_json(p: Path, obj: Mapping[str, Any]) -> None:
    mkdir(p.parent); p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def load_csv(p: Path) -> list[dict[str, Any]]:
    if not p.exists(): return []
    with p.open("r", newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

def suite(profile: str) -> list[dict[str, Any]]:
    rows = [
        dict(id="latency_b1_c1_fps24_to0", cls="latency-oriented", b=1, cams=1, fps=24, to=0, target=0, desc="B1 single-camera latency reference."),
        dict(id="latency_b1_c4_fps24_to0", cls="latency-oriented", b=1, cams=4, fps=24, to=0, target=0, desc="B1 small multi-camera latency reference."),
        dict(id="latency_b4_c4_fps24_to1", cls="latency-oriented", b=4, cams=4, fps=24, to=1, target=0, desc="B4 latency comparison at small camera count."),
        dict(id="balanced_b4_c8_fps24_to1", cls="balanced-default", b=4, cams=8, fps=24, to=1, target=0, desc="B4 practical default at 8 cameras."),
        dict(id="balanced_b4_c10_fps24_to1", cls="balanced-default", b=4, cams=10, fps=24, to=1, target=0, desc="B4 practical default at 10 cameras."),
        dict(id="balanced_b4_c10_fps24_to4", cls="balanced-default", b=4, cams=10, fps=24, to=4, target=0, desc="B4 10-camera timeout sensitivity check."),
        dict(id="monitoring_b8_c10_fps24_to4", cls="many-camera-low-refresh", b=8, cams=10, fps=24, to=4, target=0, desc="B8 conditional candidate; not a default unless data proves it."),
        dict(id="monitoring_b8_c16_fps24_to4", cls="many-camera-low-refresh", b=8, cams=16, fps=24, to=4, target=0, desc="B8 many-camera relaxed-latency candidate."),
        dict(id="capacity_b4_c10_fps5_to1_target5", cls="capacity-lower-fps", b=4, cams=10, fps=5, to=1, target=5, desc="B4 capacity check at 5 FPS/camera."),
        dict(id="capacity_b4_c10_fps3_to1_target3", cls="capacity-lower-fps", b=4, cams=10, fps=3, to=1, target=3, desc="B4 capacity check at 3 FPS/camera."),
        dict(id="capacity_b8_c16_fps3_to4_target3", cls="capacity-lower-fps", b=8, cams=16, fps=3, to=4, target=3, desc="B8 16-camera low-refresh capacity check."),
    ]
    if profile == "smoke": return [rows[0], rows[4], rows[9]]
    return [r for r in rows if profile == "baseline" or r["cls"].startswith(profile) or profile in r["cls"]]

def command(s: Mapping[str, Any], a, run_dir: Path) -> list[str]:
    cmd = [sys.executable if a.python == "" else a.python, str(a.simulator), "--model", MODEL.format(b=s["b"]),
           "--variant", VARIANT, "--migraphx-batch-size", str(s["b"]), "--migraphx-batch-timeout-ms", str(s["to"]),
           "--num-cameras", str(s["cams"]), "--frames-per-camera", "0", "--duration-s", str(a.duration_s),
           "--camera-fps", str(s["fps"]), "--buffer-mode", "latest", "--backpressure-mode", "soft",
           "--infer-workers", "1", "--post-workers", str(a.post_workers), "--shared-input-slots", str(s["cams"]),
           "--worker-threads", str(a.worker_threads), "--warmup-s", str(a.warmup_s),
           "--summary-json", str(run_dir / "summary.json"), "--detailed-csv", str(run_dir / "detailed.csv"),
           "--print-every", str(a.print_every), "--mp-start-method", a.mp_start_method]
    if a.realtime: cmd.append("--realtime")
    if a.pin_cpus: cmd += ["--pin-cpus", "--pin-all-threads"]
    if num(s.get("target")) > 0: cmd += ["--target-output-fps-per-camera", str(s["target"])]
    if a.profile_system: cmd += ["--profile-system", "--profile-interval-s", str(a.profile_interval_s)]
    return cmd + (a.extra_simulator_arg or [])

def run_scenario(s: Mapping[str, Any], a) -> str:
    rd = a.cache_dir / s["id"]; mkdir(rd)
    cmd = command(s, a, rd)
    save_json(rd / "command.json", dict(generated_at=now(), scenario=s, cwd=str(a.repo_root), command=cmd))
    if (rd / "summary.json").exists() and (rd / "detailed.csv").exists() and not a.force_rerun: return "cached"
    if not a.run_missing: return "missing"
    if a.dry_run: return "dry_run"
    print(f"[run] {s['id']}", flush=True)
    env = os.environ.copy(); env.setdefault("PYTHONUNBUFFERED", "1")
    with (rd / "run.log").open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n"); f.flush()
        rc = subprocess.run(cmd, cwd=a.repo_root, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc and not a.keep_going: raise RuntimeError(f"{s['id']} failed: {rc}; see {rd / 'run.log'}")
    return "ran" if rc == 0 else f"failed:{rc}"

def stats(summary: Mapping[str, Any], stage: str | None = None) -> list[Mapping[str, Any]]:
    return [x for x in summary.get("stage_stats", []) if isinstance(x, Mapping) and (stage is None or x.get("stage") == stage)]
def sum_stat(sts, names) -> float:
    total = 0.0
    for st in sts:
        for n in names:
            if n in st: total += num(st[n]); break
    return total

def wmean(sts, key: str, weight: str = "processed") -> float:
    n = d = 0.0
    for st in sts:
        if key in st:
            w = max(1.0, num(st.get(weight), 1)); n += num(st[key]) * w; d += w
    return n / d if d else 0.0

def prof(obj: Any, words: tuple[str, ...], bad: tuple[str, ...] = ()) -> float:
    found = []
    def walk(x, k=""):
        if isinstance(x, Mapping):
            for kk, vv in x.items(): walk(vv, f"{k}.{kk}" if k else str(kk))
        elif isinstance(x, list):
            for i, vv in enumerate(x): walk(vv, f"{k}[{i}]")
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            low = k.lower()
            if all(w in low for w in words) and not any(b in low for b in bad): found.append((0 if "avg" in low or "mean" in low else 1, k, num(x)))
    walk(obj); found.sort(key=lambda x: (x[0], len(x[1]))); return found[0][2] if found else 0.0

def detail(rows: list[Mapping[str, Any]], summary: Mapping[str, Any], cams: int) -> dict[str, float]:
    if not rows:
        return dict(steady_fps=num(summary.get("aggregate_output_fps")), steady_fps_per_camera=num(summary.get("avg_output_fps_per_camera")), avg_e2e_ms=num(summary.get("avg_e2e_ms")), p95_e2e_ms=num(summary.get("p95_e2e_ms")))
    ts = [num(r.get("post_done_ts")) for r in rows if num(r.get("post_done_ts")) > 0]
    active = max(ts) - min(ts) if len(ts) > 1 and max(ts) > min(ts) else num(summary.get("wall_s"))
    steady = len(rows) / active if active else num(summary.get("aggregate_output_fps"))
    counts: dict[int, int] = {}
    for r in rows: counts[int(num(r.get("camera_id"), -1))] = counts.get(int(num(r.get("camera_id"), -1)), 0) + 1
    cfps = [c / active for cam, c in counts.items() if cam >= 0 and active]
    return dict(steady_fps=steady, steady_fps_per_camera=(sum(cfps)/len(cfps) if cfps else steady/max(1,cams)),
        avg_e2e_ms=mean(r.get("e2e_ms") for r in rows), p95_e2e_ms=p95(r.get("e2e_ms") for r in rows),
        avg_preprocess_ms=mean(r.get("preprocess_ms") for r in rows), avg_queue_pre_to_infer_ms=mean(r.get("queue_pre_to_infer_ms") for r in rows),
        avg_inference_ms_per_frame=mean(r.get("inference_ms") for r in rows), avg_decode_ms=mean(r.get("decode_ms") for r in rows),
        avg_queue_infer_to_post_ms=mean(r.get("queue_infer_to_post_ms") for r in rows), avg_post_ms=mean(r.get("post_ms") for r in rows),
        fairness_min_to_max_camera_fps=(min(cfps)/max(cfps) if cfps and max(cfps) else 0.0))

def extract(s: Mapping[str, Any], status: str, a) -> dict[str, Any]:
    rd = a.cache_dir / s["id"]; sj = rd / "summary.json"; dc = rd / "detailed.csv"
    row = dict(scenario_id=s["id"], deployment_class=s["cls"], description=s["desc"], status=status, batch_size=s["b"], num_cameras=s["cams"], camera_fps=s["fps"], target_output_fps_per_camera=s["target"], migraphx_batch_timeout_ms=s["to"], infer_workers=1, post_workers=a.post_workers, summary_json=str(sj), detailed_csv=str(dc), run_log=str(rd / "run.log"))
    if not sj.exists(): return row
    summary = load_json(sj); rows = load_csv(dc); row.update(detail(rows, summary, int(s["cams"])))
    ast, ist = stats(summary), stats(summary, "inference")
    real_b = wmean(ist, "avg_real_batch_size", "batch_runs"); inf_b = wmean(ist, "avg_inference_ms", "batch_runs")
    cam_rep = sum_stat(ast, ["camera_replaced_before_infer", "replaced_before_infer", "latest_replaced_before_infer"])
    inf_rep = sum_stat(ist, ["inference_replaced_before_post", "replaced_before_post", "latest_replaced_before_post"])
    skipped = sum_stat(ist, ["skipped_due_backpressure", "backpressure_skips"]); processed = max(1.0, num(summary.get("total_processed_frames")))
    row.update(total_processed_frames=int(num(summary.get("total_processed_frames"))), warmup_discarded_frames=int(num(summary.get("warmup_discarded_frames"))), avg_real_batch_size=real_b, avg_inference_ms_per_batch=inf_b, batch_fill_ratio=real_b/max(1, num(s["b"])), camera_replaced_before_infer=cam_rep, inference_replaced_before_post=inf_rep, skipped_due_backpressure=skipped, camera_replaced_before_infer_ratio=cam_rep/processed, inference_replaced_before_post_ratio=inf_rep/processed, skipped_due_backpressure_ratio=skipped/processed, gpu_avg_pct=prof(summary.get("system_profile", {}), ("gpu","avg"), ("mem","vram")) or prof(summary.get("system_profile", {}), ("busy","avg"), ("mem","vram")), vram_avg_mb=prof(summary.get("system_profile", {}), ("vram","avg")) or prof(summary.get("system_profile", {}), ("gpu","mem","avg")))
    for k in ["avg_preprocess_ms","avg_queue_pre_to_infer_ms","avg_decode_ms","avg_queue_infer_to_post_ms","avg_post_ms","avg_e2e_ms","p95_e2e_ms"]:
        row[k] = num(row.get(k)) or num(summary.get(k))
    row["avg_inference_ms_per_frame"] = num(row.get("avg_inference_ms_per_frame")) or (inf_b / max(1, real_b))
    return row

def write_summary_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    mkdir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore"); w.writeheader(); [w.writerow({k: r.get(k, "") for k in FIELDS}) for r in rows]

def pick(rows: list[Mapping[str, Any]], label: str, pool, key):
    vals = [r for r in rows if num(r.get("steady_fps")) > 0 and pool(r)]
    return (label, key(vals)) if vals else None

def decision(path: Path, rows: list[Mapping[str, Any]], manifest: Path, csvp: Path) -> None:
    wins = [x for x in [
        pick(rows, "latency-oriented best", lambda r: r["deployment_class"] == "latency-oriented", lambda v: min(v, key=lambda r: (num(r.get("p95_e2e_ms")), num(r.get("avg_e2e_ms"))))),
        pick(rows, "throughput/latency practical best", lambda r: r["deployment_class"] == "balanced-default" and int(r["batch_size"]) <= 4, lambda v: max(v, key=lambda r: (num(r.get("steady_fps")), -num(r.get("p95_e2e_ms"))))),
        pick(rows, "per-camera refresh best", lambda r: True, lambda v: max(v, key=lambda r: num(r.get("steady_fps_per_camera")))),
        pick(rows, "many-camera low-refresh best", lambda r: int(r["num_cameras"]) >= 10 and r["deployment_class"] in {"many-camera-low-refresh","capacity-lower-fps"}, lambda v: max(v, key=lambda r: (num(r.get("steady_fps_per_camera")), num(r.get("batch_fill_ratio"))))),
    ] if x]
    lines = [f"- **{lab}:** `{r['scenario_id']}` (B{r['batch_size']}, cameras={r['num_cameras']}, steady={num(r.get('steady_fps')):.2f} FPS, per-camera={num(r.get('steady_fps_per_camera')):.2f} FPS, p95={num(r.get('p95_e2e_ms')):.1f} ms)." for lab, r in wins] or ["- No completed runs found."]
    table = ["| scenario | class | B | cams | steady FPS | FPS/cam | p95 E2E ms | pre→infer ms | real batch | fill |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        table.append(f"| {r.get('scenario_id')} | {r.get('deployment_class')} | {r.get('batch_size')} | {r.get('num_cameras')} | {num(r.get('steady_fps')):.2f} | {num(r.get('steady_fps_per_camera')):.2f} | {num(r.get('p95_e2e_ms')):.1f} | {num(r.get('avg_queue_pre_to_infer_ms')):.1f} | {num(r.get('avg_real_batch_size')):.2f} | {num(r.get('batch_fill_ratio')):.2f} |")
    path.write_text(f"# Runtime Optimization Baseline Decision Sheet\n\nGenerated: `{now()}`\n\nThis is measurement infrastructure; it does not directly speed the simulator.\n\n## Embedded defaults\n\n- B1 is the latency-oriented reference.\n- B4 is the practical throughput/latency default.\n- B8 is a conditional many-camera / relaxed-latency candidate, not the default 10-camera winner.\n- CPU pinning and shared input are mandatory.\n\n## Winners\n\n" + "\n".join(lines) + "\n\n## Scenario summary\n\n" + "\n".join(table) + f"\n\n## Artifacts\n\n- Manifest: `{manifest}`\n- CSV summary: `{csvp}`\n- Plots: `{path.parent / 'plots'}`\n", encoding="utf-8")

def make_plots(rows: list[Mapping[str, Any]], od: Path) -> list[str]:
    mkdir(od)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        p = od / "PLOTS_NOT_GENERATED.txt"; p.write_text(str(e), encoding="utf-8"); return [str(p)]
    rows = [r for r in rows if num(r.get("steady_fps")) > 0]
    if not rows: return []
    labs = [r["scenario_id"] for r in rows]; xs = range(len(rows)); made = []
    def save(name): p = od/name; plt.tight_layout(); plt.savefig(p, dpi=160); plt.close(); made.append(str(p))
    plt.figure(figsize=(10,6));
    for r in rows: plt.scatter(num(r.get("p95_e2e_ms")), num(r.get("steady_fps"))); plt.annotate(f"B{r['batch_size']}/C{r['num_cameras']}", (num(r.get("p95_e2e_ms")), num(r.get("steady_fps"))), fontsize=8)
    plt.xlabel("p95 E2E latency (ms)"); plt.ylabel("steady FPS"); plt.title("Throughput vs latency Pareto"); plt.grid(True, alpha=.3); save("throughput_latency_pareto.png")
    plt.figure(figsize=(12,6)); plt.bar(labs, [num(r.get("avg_queue_pre_to_infer_ms")) for r in rows]); plt.xticks(rotation=65, ha="right", fontsize=8); plt.ylabel("avg queue pre→infer (ms)"); plt.title("Queue pressure before inference"); save("queue_pre_to_infer_pressure.png")
    a=[num(r.get("camera_replaced_before_infer_ratio")) for r in rows]; b=[num(r.get("inference_replaced_before_post_ratio")) for r in rows]; c=[num(r.get("skipped_due_backpressure_ratio")) for r in rows]
    plt.figure(figsize=(12,6)); plt.bar(xs,a,label="camera replaced before infer"); plt.bar(xs,b,bottom=a,label="inference replaced before post"); plt.bar(xs,c,bottom=[x+y for x,y in zip(a,b)],label="skipped backpressure"); plt.xticks(list(xs), labs, rotation=65, ha="right", fontsize=8); plt.legend(fontsize=8); plt.ylabel("ratio vs processed frames"); plt.title("Stale / replaced / skipped work"); save("stale_replaced_ratio.png")
    plt.figure(figsize=(12,6)); bottom=[0]*len(rows)
    for k,t in [("avg_preprocess_ms","preprocess"),("avg_queue_pre_to_infer_ms","pre→infer"),("avg_inference_ms_per_frame","infer/frame"),("avg_decode_ms","decode"),("avg_queue_infer_to_post_ms","infer→post"),("avg_post_ms","post")]:
        vals=[num(r.get(k)) for r in rows]; plt.bar(xs, vals, bottom=bottom, label=t); bottom=[x+y for x,y in zip(bottom, vals)]
    plt.xticks(list(xs), labs, rotation=65, ha="right", fontsize=8); plt.legend(fontsize=8, ncol=3); plt.ylabel("ms/frame"); plt.title("Stage latency breakdown"); save("stage_latency_breakdown.png")
    target=[num(r.get("target_output_fps_per_camera")) or num(r.get("camera_fps")) for r in rows]; actual=[num(r.get("steady_fps_per_camera")) for r in rows]; lim=max(target+actual+[1])*1.05
    plt.figure(figsize=(10,6)); plt.scatter(target, actual); plt.plot([0,lim],[0,lim], linestyle="--", linewidth=1); plt.xlabel("target refresh per camera"); plt.ylabel("measured FPS/camera"); plt.title("Capacity vs target refresh"); plt.grid(True, alpha=.3); save("capacity_vs_target_output_refresh.png")
    return made

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["baseline","smoke","latency","balanced","monitoring","capacity"], default="baseline")
    ap.add_argument("--repo-root", type=Path, default=ROOT); ap.add_argument("--simulator", type=Path, default=Path("simulate_camera_stream.py"))
    ap.add_argument("--cache-dir", type=Path, default=CACHE); ap.add_argument("--output-dir", type=Path, default=OUT); ap.add_argument("--python", default="")
    ap.add_argument("--duration-s", type=float, default=130.0); ap.add_argument("--warmup-s", type=float, default=20.0); ap.add_argument("--post-workers", type=int, default=3)
    ap.add_argument("--worker-threads", type=int, default=1); ap.add_argument("--print-every", type=int, default=100); ap.add_argument("--profile-interval-s", type=float, default=0.5)
    ap.add_argument("--mp-start-method", choices=["spawn","fork","forkserver"], default="spawn"); ap.add_argument("--extra-simulator-arg", action="append", default=[])
    ap.add_argument("--run-missing", dest="run_missing", action="store_true", default=True); ap.add_argument("--no-run-missing", dest="run_missing", action="store_false")
    ap.add_argument("--force-rerun", action="store_true"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--realtime", dest="realtime", action="store_true", default=True); ap.add_argument("--no-realtime", dest="realtime", action="store_false")
    ap.add_argument("--pin-cpus", dest="pin_cpus", action="store_true", default=True); ap.add_argument("--no-pin-cpus", dest="pin_cpus", action="store_false")
    ap.add_argument("--profile-system", dest="profile_system", action="store_true", default=True); ap.add_argument("--no-profile-system", dest="profile_system", action="store_false")
    a = ap.parse_args(); a.repo_root = a.repo_root.resolve()
    if not a.cache_dir.is_absolute(): a.cache_dir = a.repo_root / a.cache_dir
    if not a.output_dir.is_absolute(): a.output_dir = a.repo_root / a.output_dir
    mkdir(a.cache_dir); mkdir(a.output_dir)
    rows = []
    selected = suite(a.profile)
    for s in selected:
        try: status = run_scenario(s, a)
        except Exception as e:
            if not a.keep_going: raise
            status = f"error:{e}"
        rows.append(extract(s, status, a))
    csvp = a.output_dir / "baseline_summary.csv"; man = a.output_dir / "baseline_manifest.json"; dec = a.output_dir / "optimization_decision_sheet.md"
    plots = make_plots(rows, a.output_dir / "plots"); write_summary_csv(csvp, rows)
    save_json(man, dict(suite_name="runtime_optimization_suite", generated_at=now(), variant=VARIANT, profile=a.profile, cache_dir=str(a.cache_dir), output_dir=str(a.output_dir), embedded_conclusions=dict(B1="latency-oriented reference", B4="practical throughput/latency default", B8="conditional many-camera / relaxed-latency candidate", mandatory_path_assumptions=["pin_cpus", "shared_input_slots >= num_cameras"]), scenarios=selected, results=rows, plots=plots))
    decision(dec, rows, man, csvp)
    print(f"manifest: {man}\nsummary CSV: {csvp}\ndecision sheet: {dec}\nplots: {a.output_dir / 'plots'}\ncache: {a.cache_dir}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
