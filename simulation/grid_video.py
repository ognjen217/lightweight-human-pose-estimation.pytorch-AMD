"""Optional security-monitor grid video rendering."""

from __future__ import annotations

import queue as py_queue
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .utils import ensure_parent, safe_float


def draw_poses_on_frame(frame: np.ndarray, pose_entries: np.ndarray, all_keypoints: np.ndarray) -> None:
    """Draw skeletons returned by postprocess_from_maps on a BGR frame in-place."""
    if pose_entries is None or all_keypoints is None or len(all_keypoints) == 0:
        return

    try:
        import cv2
        from modules.keypoints import BODY_PARTS_KPT_IDS
    except Exception:
        return

    for pose in pose_entries:
        for part_id in range(len(BODY_PARTS_KPT_IDS)):
            kpt_a_id = pose[BODY_PARTS_KPT_IDS[part_id][0]]
            kpt_b_id = pose[BODY_PARTS_KPT_IDS[part_id][1]]
            if kpt_a_id != -1 and kpt_b_id != -1:
                kpt_a = all_keypoints[int(kpt_a_id)]
                kpt_b = all_keypoints[int(kpt_b_id)]
                cv2.line(
                    frame,
                    (int(kpt_a[0]), int(kpt_a[1])),
                    (int(kpt_b[0]), int(kpt_b[1])),
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )

        for kpt_id in pose[:-2]:
            if kpt_id != -1:
                kpt = all_keypoints[int(kpt_id)]
                cv2.circle(
                    frame,
                    (int(kpt[0]), int(kpt[1])),
                    3,
                    (0, 255, 0),
                    -1,
                    lineType=cv2.LINE_AA,
                )


def make_monitor_grid_frame(
    *,
    latest_frames: Dict[int, Dict[str, Any]],
    num_cameras: int,
    grid_rows: int,
    grid_cols: int,
    cell_w: int,
    cell_h: int,
    camera_sources_: Sequence[str],
) -> np.ndarray:
    """Create one BGR 4x4-like monitor frame from latest per-camera outputs."""
    import cv2

    grid = np.zeros((grid_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for cam_id in range(min(num_cameras, grid_rows * grid_cols)):
        r = cam_id // grid_cols
        c = cam_id % grid_cols
        y0 = r * cell_h
        x0 = c * cell_w
        tile = grid[y0 : y0 + cell_h, x0 : x0 + cell_w]
        packet = latest_frames.get(cam_id)

        if packet is not None and packet.get("frame_bgr") is not None:
            frame = packet["frame_bgr"]
            try:
                resized = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                tile[:] = resized
            except Exception:
                tile[:] = 0
        else:
            tile[:] = 18
            cv2.putText(tile, "NO SIGNAL", (16, cell_h // 2), font, 0.8, (120, 120, 120), 2, cv2.LINE_AA)

        # Dark label strip for readability.
        cv2.rectangle(tile, (0, 0), (cell_w, 42), (0, 0, 0), -1)
        source_name = Path(camera_sources_[cam_id]).name if cam_id < len(camera_sources_) else ""
        if packet is None:
            label1 = f"CAM {cam_id:02d}"
            label2 = source_name
        else:
            label1 = (
                f"CAM {cam_id:02d}  f={int(packet.get('frame_id', 0))}  "
                f"poses={int(packet.get('num_poses', 0))}"
            )
            label2 = f"e2e={safe_float(packet.get('e2e_ms', 0.0)):.0f}ms  {source_name}"

        cv2.putText(tile, label1[:48], (8, 16), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, label2[:58], (8, 34), font, 0.40, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (cell_w - 1, cell_h - 1), (70, 70, 70), 1)

    return grid


def grid_video_writer_worker(
    *,
    grid_q,
    output_path: str,
    num_cameras: int,
    grid_rows: int,
    grid_cols: int,
    cell_w: int,
    cell_h: int,
    fps: float,
    codec: str,
    camera_sources_: Sequence[str],
    stop_event,
    stats_q,
    error_q,
) -> None:
    """Write a single security-monitor-style concatenated grid video.

    The writer receives already-drawn per-camera frames from postprocess workers,
    keeps the newest frame per camera, and writes a fixed-rate grid video.

    Important for MP4: the process must exit cleanly and call
    VideoWriter.release(); otherwise ffprobe reports "moov atom not found".
    """
    writer = None
    released = False
    try:
        import cv2

        ensure_parent(output_path)
        fps = float(fps if fps > 0 else 10.0)
        period_s = 1.0 / fps
        grid_w = int(grid_cols * cell_w)
        grid_h = int(grid_rows * cell_h)
        fourcc_text = (codec or "mp4v")[:4]
        if len(fourcc_text) < 4:
            fourcc_text = "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*fourcc_text)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open grid video writer: {output_path}")

        latest_frames: Dict[int, Dict[str, Any]] = {}
        packets_received = 0
        frames_written = 0
        first_packet_ts: Optional[float] = None
        t0 = time.perf_counter()
        next_write_ts = t0

        print(
            f"[GRID] Writing monitor video: {output_path} "
            f"({grid_cols}x{grid_rows}, {grid_w}x{grid_h}, {fps:.2f} FPS)",
            flush=True,
        )

        # Write one initial frame immediately. This makes the output container
        # valid even if no postprocess packet ever arrives, and it also makes
        # debugging easier because ffprobe/ffmpeg can still open the file.
        initial_frame = make_monitor_grid_frame(
            latest_frames=latest_frames,
            num_cameras=num_cameras,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            cell_w=cell_w,
            cell_h=cell_h,
            camera_sources_=camera_sources_,
        )
        writer.write(initial_frame)
        frames_written += 1

        while True:
            drained = 0
            while drained < 256:
                try:
                    packet = grid_q.get_nowait()
                except py_queue.Empty:
                    break
                cam_id = int(packet.get("camera_id", -1))
                if 0 <= cam_id < num_cameras:
                    latest_frames[cam_id] = packet
                    packets_received += 1
                    if first_packet_ts is None:
                        first_packet_ts = time.perf_counter()
                drained += 1

            now = time.perf_counter()
            if latest_frames and now >= next_write_ts:
                frame = make_monitor_grid_frame(
                    latest_frames=latest_frames,
                    num_cameras=num_cameras,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    cell_w=cell_w,
                    cell_h=cell_h,
                    camera_sources_=camera_sources_,
                )
                writer.write(frame)
                frames_written += 1
                next_write_ts += period_s
                if next_write_ts < now - period_s:
                    next_write_ts = now + period_s

            if stop_event.is_set():
                # Drain remaining packets once, write a final frame, and exit.
                if grid_q.empty():
                    if latest_frames:
                        frame = make_monitor_grid_frame(
                            latest_frames=latest_frames,
                            num_cameras=num_cameras,
                            grid_rows=grid_rows,
                            grid_cols=grid_cols,
                            cell_w=cell_w,
                            cell_h=cell_h,
                            camera_sources_=camera_sources_,
                        )
                        writer.write(frame)
                        frames_written += 1
                    break

            time.sleep(0.002)

        writer.release()
        released = True
        wall_s = time.perf_counter() - t0
        stats_q.put(
            {
                "stage": "grid_video_writer",
                "output_path": output_path,
                "packets_received": packets_received,
                "frames_written": frames_written,
                "fps": fps,
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
                "cell_w": cell_w,
                "cell_h": cell_h,
                "wall_s": wall_s,
            }
        )
        print(
            f"[GRID] Done. packets={packets_received} frames_written={frames_written} output={output_path}",
            flush=True,
        )

    except Exception:
        error_q.put({"stage": "grid_video_writer", "traceback": traceback.format_exc()})
    finally:
        # Always try to finalize the container. Without this, MP4 output may
        # exist but be unreadable because the moov atom was never written.
        try:
            if writer is not None and not released:
                writer.release()
        except Exception:
            pass
