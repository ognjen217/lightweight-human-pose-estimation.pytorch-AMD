"""Queue and newest-frame buffer helpers."""

from __future__ import annotations

import queue as py_queue
import time
from typing import Any, Dict, List, Sequence, Tuple


def latest_put_with_dropped(q, item):
    try:
        q.put_nowait(item)
        return None
    except py_queue.Full:
        pass
    dropped = None
    try:
        dropped = q.get_nowait()
    except py_queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except py_queue.Full:
        try:
            newer = q.get_nowait()
            dropped = newer if dropped is None else dropped
        except py_queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except py_queue.Full:
            return item
    return dropped

def collect_queue_batch(
    *,
    first_item: Dict[str, Any],
    in_q,
    batch_size: int,
    batch_timeout_ms: float,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Collect a small batch from a FIFO queue. Returns (items, saw_stop_token)."""
    batch_size = max(1, int(batch_size))
    timeout_s = max(0.0, float(batch_timeout_ms)) / 1000.0
    batch_items = [first_item]
    saw_stop = False

    if batch_size <= 1:
        return batch_items, saw_stop

    deadline = time.perf_counter() + timeout_s
    while len(batch_items) < batch_size:
        try:
            if timeout_s > 0.0:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                item = in_q.get(timeout=remaining)
            else:
                item = in_q.get_nowait()
        except py_queue.Empty:
            break

        if item is None:
            saw_stop = True
            break
        batch_items.append(item)

    return batch_items, saw_stop

def latest_put(q, item) -> int:
    """Put newest item into a maxsize=1 queue, replacing the previous item if needed.

    Returns 1 when an older item had to be discarded.
    """
    dropped = 0
    try:
        q.put_nowait(item)
        return dropped
    except py_queue.Full:
        pass

    try:
        q.get_nowait()
        dropped = 1
    except py_queue.Empty:
        pass

    try:
        q.put_nowait(item)
    except py_queue.Full:
        # Rare race if another producer filled it; keep newest semantics by dropping one more.
        try:
            q.get_nowait()
            dropped = 1
        except py_queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except py_queue.Full:
            dropped = 1
    return dropped


def all_done(done_flags) -> bool:
    try:
        return all(bool(v) for v in done_flags[:])
    except Exception:
        return False


def all_queues_empty(queues: Sequence[Any]) -> bool:
    for q in queues:
        try:
            if not q.empty():
                return False
        except Exception:
            return False
    return True
