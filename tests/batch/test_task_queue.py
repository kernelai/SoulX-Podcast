"""
Tests for batch.task_queue module.

All model calls are mocked — no GPU required.
"""
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from batch.file_parser import ParsedScript
from batch.task_queue import (
    BatchTaskQueue,
    SpeakerConfig,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    STATUS_INVALID,
)


def _make_script(filename: str, valid: bool = True, error: str = None) -> ParsedScript:
    """Helper to create ParsedScript test fixtures."""
    if valid:
        return ParsedScript(
            filename=filename,
            dialogue_text=f"[S1] Hello from {filename}\n[S2] Hi there",
            is_valid=True,
            error_msg=None,
        )
    return ParsedScript(
        filename=filename,
        dialogue_text="",
        is_valid=False,
        error_msg=error or "test error",
    )


def _make_config() -> SpeakerConfig:
    return SpeakerConfig(seed=42)


def _mock_synthesize_ok(dialogue_text: str, config: SpeakerConfig):
    """Mock synthesize function that returns valid audio data."""
    # Return 1 second of silence at 24kHz
    audio = np.zeros(24000, dtype=np.float32)
    return (24000, audio)


def _mock_synthesize_slow(dialogue_text: str, config: SpeakerConfig):
    """Mock synthesize that takes a bit of time."""
    time.sleep(0.2)
    return _mock_synthesize_ok(dialogue_text, config)


def _mock_synthesize_fail(dialogue_text: str, config: SpeakerConfig):
    """Mock synthesize that raises an error."""
    raise RuntimeError("GPU out of memory")


class TestAddTasks:
    """Tests for adding tasks to the queue."""

    def test_add_valid_tasks(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        scripts = [_make_script("a.txt"), _make_script("b.txt"), _make_script("c.txt")]
        ids = q.add_tasks(scripts, _make_config())

        assert len(ids) == 3
        for tid in ids:
            assert q._tasks[tid].status == STATUS_PENDING

    def test_add_invalid_tasks_marked_as_invalid(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        scripts = [_make_script("bad.txt", valid=False, error="no tags")]
        ids = q.add_tasks(scripts, _make_config())

        assert len(ids) == 1
        task = q._tasks[ids[0]]
        assert task.status == STATUS_INVALID
        assert task.error == "no tags"

    def test_add_mixed_valid_and_invalid(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        scripts = [
            _make_script("good.txt"),
            _make_script("bad.txt", valid=False),
            _make_script("good2.txt"),
        ]
        ids = q.add_tasks(scripts, _make_config())
        assert len(ids) == 3

        statuses = [q._tasks[tid].status for tid in ids]
        assert statuses == [STATUS_PENDING, STATUS_INVALID, STATUS_PENDING]

    def test_queue_size_matches_valid_count(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        scripts = [
            _make_script("a.txt"),
            _make_script("bad.txt", valid=False),
            _make_script("b.txt"),
        ]
        q.add_tasks(scripts, _make_config())
        assert q._queue.qsize() == 2  # only valid tasks queued


class TestCancelPending:
    """Tests for cancel_all_pending."""

    def test_cancel_all_pending(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        q.add_tasks([_make_script("a.txt"), _make_script("b.txt")], _make_config())
        count = q.cancel_all_pending()

        assert count == 2
        for task in q._tasks.values():
            assert task.status == STATUS_CANCELLED

    def test_cancel_skips_invalid(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        q.add_tasks([_make_script("bad.txt", valid=False)], _make_config())
        count = q.cancel_all_pending()
        assert count == 0
        assert list(q._tasks.values())[0].status == STATUS_INVALID

    def test_cancel_returns_zero_when_no_pending(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        count = q.cancel_all_pending()
        assert count == 0


class TestWorkerProcessing:
    """Tests for the background worker processing tasks."""

    def test_worker_processes_task_to_completion(self, tmp_output_dir):
        q = BatchTaskQueue(
            output_dir=tmp_output_dir,
            synthesize_fn=_mock_synthesize_ok,
        )
        q.add_tasks([_make_script("test.txt")], _make_config())
        q.start_worker()

        # Wait for processing to complete
        _wait_until_done(q, timeout=5)

        task = list(q._tasks.values())[0]
        assert task.status == STATUS_COMPLETED
        assert task.output_path is not None
        assert task.output_path.exists()
        assert task.output_path.suffix == ".wav"
        q.shutdown()

    def test_worker_processes_multiple_sequentially(self, tmp_output_dir):
        q = BatchTaskQueue(
            output_dir=tmp_output_dir,
            synthesize_fn=_mock_synthesize_ok,
        )
        scripts = [_make_script(f"file{i}.txt") for i in range(3)]
        q.add_tasks(scripts, _make_config())
        q.start_worker()

        _wait_until_done(q, timeout=10)

        completed = [t for t in q._tasks.values() if t.status == STATUS_COMPLETED]
        assert len(completed) == 3
        q.shutdown()

    def test_worker_skips_cancelled_tasks(self, tmp_output_dir):
        call_count = {"n": 0}

        def counting_synth(text, config):
            call_count["n"] += 1
            return _mock_synthesize_ok(text, config)

        q = BatchTaskQueue(
            output_dir=tmp_output_dir,
            synthesize_fn=counting_synth,
        )
        q.add_tasks([_make_script("a.txt"), _make_script("b.txt")], _make_config())
        q.cancel_all_pending()
        q.start_worker()

        # Give worker time to process (it should skip both)
        time.sleep(1)

        assert call_count["n"] == 0
        q.shutdown()

    def test_task_failure_does_not_block_queue(self, tmp_output_dir):
        call_num = {"n": 0}

        def fail_then_succeed(text, config):
            call_num["n"] += 1
            if call_num["n"] == 1:
                raise RuntimeError("boom")
            return _mock_synthesize_ok(text, config)

        q = BatchTaskQueue(
            output_dir=tmp_output_dir,
            synthesize_fn=fail_then_succeed,
        )
        q.add_tasks(
            [_make_script("fail.txt"), _make_script("ok.txt")],
            _make_config(),
        )
        q.start_worker()

        _wait_until_done(q, timeout=5)

        tasks = list(q._tasks.values())
        statuses = {t.filename: t.status for t in tasks}
        assert statuses["fail.txt"] == STATUS_FAILED
        assert statuses["ok.txt"] == STATUS_COMPLETED
        assert "boom" in tasks[0].error
        q.shutdown()

    def test_output_filename_matches_input(self, tmp_output_dir):
        q = BatchTaskQueue(
            output_dir=tmp_output_dir,
            synthesize_fn=_mock_synthesize_ok,
        )
        q.add_tasks([_make_script("my_podcast.txt")], _make_config())
        q.start_worker()

        _wait_until_done(q, timeout=5)

        task = list(q._tasks.values())[0]
        assert task.output_path.stem.startswith("my_podcast_")
        q.shutdown()


class TestDeleteCompleted:
    """Tests for deleting completed tasks and files."""

    def test_delete_removes_files_and_records(self, tmp_output_dir):
        q = BatchTaskQueue(
            output_dir=tmp_output_dir,
            synthesize_fn=_mock_synthesize_ok,
        )
        q.add_tasks([_make_script("a.txt"), _make_script("b.txt")], _make_config())
        q.start_worker()
        _wait_until_done(q, timeout=5)

        # Verify files exist before delete
        paths = q.get_completed_paths()
        assert len(paths) == 2
        assert all(p.exists() for p in paths)

        # Delete
        count = q.delete_completed()
        assert count == 2
        assert len(q._tasks) == 0
        # Files should be removed from disk
        assert all(not p.exists() for p in paths)
        q.shutdown()

    def test_delete_keeps_pending_tasks(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        q.add_tasks(
            [_make_script("a.txt"), _make_script("b.txt")],
            _make_config(),
        )
        # Don't start worker — tasks stay pending
        count = q.delete_completed()
        assert count == 0
        assert len(q._tasks) == 2

    def test_delete_also_removes_failed_and_cancelled(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        scripts = [
            _make_script("a.txt"),
            _make_script("bad.txt", valid=False),
        ]
        q.add_tasks(scripts, _make_config())
        q.cancel_all_pending()

        count = q.delete_completed()
        # Cancelled + Invalid = 2
        assert count == 2
        assert len(q._tasks) == 0


class TestStatusTable:
    """Tests for get_status_table output format."""

    def test_status_table_format(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        q.add_tasks(
            [_make_script("a.txt"), _make_script("bad.txt", valid=False)],
            _make_config(),
        )
        table = q.get_status_table()

        assert len(table) == 2
        # Each row: [filename, status, progress, elapsed, error]
        assert len(table[0]) == 5
        assert table[0][0] == "a.txt"
        assert table[0][1] == STATUS_PENDING
        assert table[1][0] == "bad.txt"
        assert table[1][1] == STATUS_INVALID

    def test_empty_queue_table(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        table = q.get_status_table()
        assert table == []


class TestSummary:
    """Tests for get_summary."""

    def test_summary_no_tasks(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        assert q.get_summary() == "无任务"

    def test_summary_with_tasks(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        q.add_tasks([_make_script("a.txt"), _make_script("b.txt")], _make_config())
        summary = q.get_summary()
        assert "已完成 0/2" in summary
        assert "排队 2" in summary


class TestIsBusy:
    """Tests for is_busy."""

    def test_not_busy_when_empty(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        assert q.is_busy() is False

    def test_busy_when_pending(self, tmp_output_dir):
        q = BatchTaskQueue(output_dir=tmp_output_dir)
        q.add_tasks([_make_script("a.txt")], _make_config())
        assert q.is_busy() is True

    def test_not_busy_after_completion(self, tmp_output_dir):
        q = BatchTaskQueue(
            output_dir=tmp_output_dir,
            synthesize_fn=_mock_synthesize_ok,
        )
        q.add_tasks([_make_script("a.txt")], _make_config())
        q.start_worker()
        _wait_until_done(q, timeout=5)
        assert q.is_busy() is False
        q.shutdown()


class TestGetCompletedPaths:
    """Tests for get_completed_paths."""

    def test_returns_only_completed(self, tmp_output_dir):
        q = BatchTaskQueue(
            output_dir=tmp_output_dir,
            synthesize_fn=_mock_synthesize_ok,
        )
        q.add_tasks([_make_script("a.txt")], _make_config())
        # Before processing
        assert q.get_completed_paths() == []

        q.start_worker()
        _wait_until_done(q, timeout=5)
        paths = q.get_completed_paths()
        assert len(paths) == 1
        assert paths[0].exists()
        q.shutdown()


class TestModelLock:
    """Tests for model lock integration."""

    def test_worker_acquires_model_lock(self, tmp_output_dir):
        lock = threading.Lock()
        lock_acquired = {"value": False}

        def check_lock_synth(text, config):
            # If we're here, the lock should be held by _process_task
            lock_acquired["value"] = not lock.acquire(blocking=False)
            if not lock_acquired["value"]:
                lock.release()
            return _mock_synthesize_ok(text, config)

        q = BatchTaskQueue(
            output_dir=tmp_output_dir,
            synthesize_fn=check_lock_synth,
            model_lock=lock,
        )
        q.add_tasks([_make_script("a.txt")], _make_config())
        q.start_worker()
        _wait_until_done(q, timeout=5)

        assert lock_acquired["value"] is True
        q.shutdown()


def _wait_until_done(q: BatchTaskQueue, timeout: float = 5):
    """Helper: block until queue has no pending/processing tasks."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not q.is_busy():
            return
        time.sleep(0.1)
    pytest.fail(f"Queue did not finish within {timeout}s")
