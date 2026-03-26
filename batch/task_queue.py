"""
Synchronous batch task queue for Gradio WebUI.

Uses a single worker thread to process tasks sequentially,
since GPU inference must be serialized.
"""
import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import scipy.io.wavfile as wavfile

from batch.file_parser import ParsedScript

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerConfig:
    """Immutable speaker configuration for a batch run."""
    spk1_prompt_audio: Optional[str] = None
    spk1_prompt_text: str = ""
    spk1_dialect_prompt_text: str = ""
    spk2_prompt_audio: Optional[str] = None
    spk2_prompt_text: str = ""
    spk2_dialect_prompt_text: str = ""
    seed: int = 1988


# Task status constants
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_INVALID = "invalid"


@dataclass
class BatchTask:
    """Mutable task record tracked by the queue."""
    task_id: str
    filename: str
    dialogue_text: str
    speaker_config: SpeakerConfig
    status: str = STATUS_PENDING
    progress: int = 0
    output_path: Optional[Path] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def elapsed_seconds(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()


class BatchTaskQueue:
    """
    Thread-safe batch task queue with a single background worker.

    The synthesize_fn callback is called with (dialogue_text, SpeakerConfig)
    and must return (sample_rate: int, audio_array: np.ndarray).
    """

    def __init__(
        self,
        output_dir: Path,
        synthesize_fn: Optional[Callable] = None,
        model_lock: Optional[threading.Lock] = None,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._synthesize_fn = synthesize_fn
        self._model_lock = model_lock or threading.Lock()

        # Task storage: task_id -> BatchTask
        self._tasks: dict[str, BatchTask] = {}
        self._task_order: list[str] = []  # insertion order
        self._lock = threading.Lock()

        # Work queue for the worker thread
        self._queue: queue.Queue = queue.Queue(maxsize=500)

        # Worker thread (daemon so it dies with the main process)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_worker(self):
        """Start the background worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="batch-worker"
        )
        self._worker_thread.start()
        logger.info("Batch worker thread started")

    def _worker_loop(self):
        """Main loop of the background worker thread."""
        while not self._stop_event.is_set():
            try:
                task_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_task(task_id)
            except Exception as e:
                logger.error(f"Unexpected error processing {task_id}: {e}", exc_info=True)
            finally:
                self._queue.task_done()

    def _process_task(self, task_id: str):
        """Process a single task from the queue."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            # Skip cancelled tasks
            if task.status == STATUS_CANCELLED:
                logger.info(f"Skipping cancelled task: {task.filename}")
                return
            task.status = STATUS_PROCESSING
            task.started_at = datetime.now()
            task.progress = 10

        try:
            if self._synthesize_fn is None:
                raise RuntimeError("synthesize_fn not set")

            # Acquire the global model lock (shared with single-synthesis Tab)
            with self._model_lock:
                # Re-check cancellation after acquiring lock
                with self._lock:
                    if task.status == STATUS_CANCELLED:
                        return

                with self._lock:
                    task.progress = 30

                sample_rate, audio_array = self._synthesize_fn(
                    task.dialogue_text, task.speaker_config
                )

            with self._lock:
                task.progress = 80

            # Save wav file (include task_id to guarantee uniqueness)
            stem = Path(task.filename).stem
            output_path = self._output_dir / f"{stem}_{task.task_id[:8]}.wav"

            wavfile.write(str(output_path), sample_rate, audio_array)

            with self._lock:
                task.output_path = output_path
                task.status = STATUS_COMPLETED
                task.progress = 100
                task.completed_at = datetime.now()

            logger.info(
                f"Task completed: {task.filename} -> {output_path.name} "
                f"({task.elapsed_seconds:.1f}s)"
            )

        except Exception as e:
            with self._lock:
                task.status = STATUS_FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
            logger.error(f"Task failed: {task.filename}: {e}", exc_info=True)

    # ---- Public API ----

    def set_synthesize_fn(self, fn: Callable):
        self._synthesize_fn = fn

    def add_tasks(
        self, scripts: list[ParsedScript], config: SpeakerConfig
    ) -> list[str]:
        """
        Add parsed scripts to the queue.

        Invalid scripts are recorded with STATUS_INVALID (not queued).
        Returns list of all task_ids (including invalid ones).
        """
        task_ids = []
        ids_to_enqueue = []

        # Phase 1: register tasks under lock (no blocking I/O)
        with self._lock:
            for script in scripts:
                task_id = uuid.uuid4().hex[:12]

                if not script.is_valid:
                    # Record but don't queue
                    task = BatchTask(
                        task_id=task_id,
                        filename=script.filename,
                        dialogue_text="",
                        speaker_config=config,
                        status=STATUS_INVALID,
                        error=script.error_msg,
                    )
                    self._tasks[task_id] = task
                    self._task_order.append(task_id)
                    task_ids.append(task_id)
                    continue

                task = BatchTask(
                    task_id=task_id,
                    filename=script.filename,
                    dialogue_text=script.dialogue_text,
                    speaker_config=config,
                )
                self._tasks[task_id] = task
                self._task_order.append(task_id)
                ids_to_enqueue.append(task_id)
                task_ids.append(task_id)

        # Phase 2: enqueue outside lock to avoid deadlock when queue is full
        # (worker needs self._lock in _process_task to drain the queue)
        for tid in ids_to_enqueue:
            self._queue.put(tid)

        logger.info(f"Added {len(task_ids)} tasks to batch queue")
        return task_ids

    def cancel_all_pending(self) -> int:
        """Cancel all pending tasks. Returns number cancelled."""
        count = 0
        with self._lock:
            for task in self._tasks.values():
                if task.status == STATUS_PENDING:
                    task.status = STATUS_CANCELLED
                    count += 1
        logger.info(f"Cancelled {count} pending tasks")
        return count

    def delete_completed(self) -> int:
        """
        Delete all completed task records and their wav files.
        Returns number of tasks deleted.
        """
        count = 0
        with self._lock:
            to_delete = [
                tid for tid, t in self._tasks.items()
                if t.status in (STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED, STATUS_INVALID)
            ]
            delete_set = set(to_delete)
            for tid in to_delete:
                task = self._tasks[tid]
                # Remove wav file from disk
                if task.output_path and task.output_path.exists():
                    try:
                        task.output_path.unlink()
                    except OSError as e:
                        logger.warning(f"Failed to delete {task.output_path}: {e}")
                del self._tasks[tid]
                count += 1
            self._task_order = [t for t in self._task_order if t not in delete_set]
        logger.info(f"Deleted {count} completed tasks")
        return count

    def get_status_table(self) -> list[list[str]]:
        """
        Return task status as a 2D list for gr.Dataframe.
        Columns: [filename, status, progress, elapsed, error]
        """
        rows = []
        with self._lock:
            for tid in self._task_order:
                task = self._tasks[tid]
                elapsed = ""
                if task.elapsed_seconds is not None:
                    elapsed = f"{task.elapsed_seconds:.1f}s"
                progress = f"{task.progress}%" if task.status not in (STATUS_INVALID, STATUS_CANCELLED) else ""
                rows.append([
                    task.filename,
                    task.status,
                    progress,
                    elapsed,
                    task.error or "",
                ])
        return rows

    def get_summary(self) -> str:
        """Return a one-line summary string like '已完成 2/5, 处理中 1'."""
        with self._lock:
            total = len(self._tasks)
            if total == 0:
                return "无任务"
            counts = {}
            for task in self._tasks.values():
                counts[task.status] = counts.get(task.status, 0) + 1

        parts = []
        completed = counts.get(STATUS_COMPLETED, 0)
        parts.append(f"已完成 {completed}/{total}")
        if counts.get(STATUS_PROCESSING, 0):
            parts.append(f"处理中 {counts[STATUS_PROCESSING]}")
        if counts.get(STATUS_PENDING, 0):
            parts.append(f"排队 {counts[STATUS_PENDING]}")
        if counts.get(STATUS_FAILED, 0):
            parts.append(f"失败 {counts[STATUS_FAILED]}")
        if counts.get(STATUS_CANCELLED, 0):
            parts.append(f"已取消 {counts[STATUS_CANCELLED]}")
        if counts.get(STATUS_INVALID, 0):
            parts.append(f"格式错误 {counts[STATUS_INVALID]}")
        return ", ".join(parts)

    def get_completed_paths(self) -> list[Path]:
        """Return output paths of all completed tasks."""
        with self._lock:
            return [
                t.output_path
                for t in self._tasks.values()
                if t.status == STATUS_COMPLETED and t.output_path and t.output_path.exists()
            ]

    def is_busy(self) -> bool:
        """Return True if any task is pending or processing."""
        with self._lock:
            return any(
                t.status in (STATUS_PENDING, STATUS_PROCESSING)
                for t in self._tasks.values()
            )

    def has_tasks(self) -> bool:
        with self._lock:
            return len(self._tasks) > 0

    def shutdown(self):
        """Stop the worker thread."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)
            logger.info("Batch worker thread stopped")
