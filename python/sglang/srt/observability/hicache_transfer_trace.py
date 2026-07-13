"""Low-overhead JSONL tracing for logical HiCache storage transfers."""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HiCacheTransferTracer:
    def __init__(self, path: Optional[str] = None):
        configured_path = path or os.getenv("SGLANG_HICACHE_TRANSFER_TRACE_PATH")
        self.path = (
            configured_path.format(
                pid=os.getpid(),
                rank=os.getenv("RANK", "0"),
                local_rank=os.getenv("LOCAL_RANK", "0"),
            )
            if configured_path
            else None
        )
        self.enabled = bool(self.path)
        self._lock = threading.Lock()
        self._file = None
        if self.enabled:
            try:
                trace_path = Path(self.path)
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                self._file = trace_path.open("a", buffering=1)
                atexit.register(self.close)
            except OSError:
                logger.warning(
                    "Failed to initialize HiCache transfer tracing at %s",
                    self.path,
                    exc_info=True,
                )
                self.enabled = False

    def emit(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = {
            "schema_version": 1,
            "timestamp_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "rank": int(os.getenv("RANK", "0")),
            "local_rank": int(os.getenv("LOCAL_RANK", "0")),
            **event,
        }
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.write(json.dumps(payload, separators=(",", ":")) + "\n")
            except (OSError, TypeError, ValueError):
                logger.warning(
                    "Failed to write HiCache transfer trace event",
                    exc_info=True,
                )

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


def transfer_event(
    *,
    operation: str,
    operation_id: int,
    request_id: Optional[str],
    status: str,
    enqueue_ns: int,
    io_start_ns: Optional[int],
    io_end_ns: int,
    requested_tokens: int,
    completed_tokens: int,
    page_size: int,
    bytes_per_token: int,
    storage_backend: Optional[str],
    error: Optional[str] = None,
) -> dict[str, Any]:
    requested_bytes = requested_tokens * bytes_per_token
    completed_bytes = completed_tokens * bytes_per_token
    queue_ms = (io_start_ns - enqueue_ns) / 1e6 if io_start_ns is not None else None
    io_ms = (io_end_ns - io_start_ns) / 1e6 if io_start_ns is not None else None
    total_ms = (io_end_ns - enqueue_ns) / 1e6
    throughput_mib_s = (
        completed_bytes / 2**20 / (io_ms / 1000)
        if io_ms is not None and io_ms > 0
        else None
    )
    return {
        "event": "hicache_storage_transfer",
        "operation": operation,
        "operation_id": operation_id,
        "request_id": request_id,
        "status": status,
        "enqueue_ns": enqueue_ns,
        "io_start_ns": io_start_ns,
        "io_end_ns": io_end_ns,
        "queue_ms": queue_ms,
        "io_ms": io_ms,
        "total_ms": total_ms,
        "requested_tokens": requested_tokens,
        "completed_tokens": completed_tokens,
        "requested_pages": requested_tokens // page_size,
        "completed_pages": completed_tokens // page_size,
        "page_size": page_size,
        "bytes_per_token": bytes_per_token,
        "requested_bytes": requested_bytes,
        "completed_bytes": completed_bytes,
        "throughput_mib_s": throughput_mib_s,
        "storage_backend": storage_backend,
        "error": error,
    }
