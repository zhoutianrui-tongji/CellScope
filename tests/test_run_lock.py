import json
import os

import pytest
from click.testing import CliRunner

from cellscope import cli
from cellscope.cli import main
from cellscope.run_lock import (
    LOCK_FILENAME,
    OutputDirectoryLockError,
    OutputDirectoryRunLock,
)


def _load_lock_record(out_dir):
    with open(os.path.join(out_dir, LOCK_FILENAME), "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_cli_run_makes_output_tree_group_readable(tmp_path):
    @cli._with_output_directory_lock
    def write_private_output(*, out_dir):
        private_dir = os.path.join(out_dir, "intermediate_results")
        os.mkdir(private_dir, 0o700)
        private_file = os.path.join(private_dir, "atomic-output.tmp")
        with open(private_file, "w", encoding="utf-8") as handle:
            handle.write("result")
        os.chmod(private_file, 0o600)

    write_private_output(out_dir=str(tmp_path))

    private_dir = tmp_path / "intermediate_results"
    private_file = private_dir / "atomic-output.tmp"
    assert private_dir.stat().st_mode & 0o050 == 0o050
    assert private_file.stat().st_mode & 0o040
    assert not private_file.stat().st_mode & 0o020


def test_live_owner_blocks_second_run_and_reports_owner(tmp_path):
    first = OutputDirectoryRunLock(str(tmp_path)).acquire()
    try:
        with pytest.raises(OutputDirectoryLockError) as exc_info:
            OutputDirectoryRunLock(str(tmp_path)).acquire()
        message = str(exc_info.value)
        assert "another live CellScoper process" in message
        assert f"pid: {os.getpid()}" in message
        assert str(tmp_path) in message
    finally:
        first.release()


def test_stale_record_is_recovered_without_deleting_outputs(tmp_path):
    sentinel = tmp_path / "final_data.parquet"
    sentinel.write_bytes(b"user output")
    lock_path = tmp_path / LOCK_FILENAME
    lock_path.write_text(
        json.dumps(
            {
                "status": "running",
                "pid": 12345,
                "host": "old-host",
                "started_at": "2000-01-01T00:00:00Z",
                "command": "old command",
            }
        ),
        encoding="utf-8",
    )

    with OutputDirectoryRunLock(str(tmp_path)) as lock:
        record = _load_lock_record(tmp_path)
        assert record["status"] == "running"
        assert record["pid"] == os.getpid()
        assert record["recovered_stale_owner"]["pid"] == 12345
        assert lock.acquired

    assert sentinel.read_bytes() == b"user output"
    released = _load_lock_record(tmp_path)
    assert released["status"] == "completed"
    assert "released_at" in released


def test_lock_is_released_and_records_error_on_exception(tmp_path):
    with pytest.raises(RuntimeError, match="pipeline failed"):
        with OutputDirectoryRunLock(str(tmp_path)):
            raise RuntimeError("pipeline failed")

    failed = _load_lock_record(tmp_path)
    assert failed["status"] == "failed"
    assert failed["error_type"] == "RuntimeError"

    replacement = OutputDirectoryRunLock(str(tmp_path)).acquire()
    try:
        assert replacement.acquired
    finally:
        replacement.release()


def test_cli_refuses_to_write_into_an_actively_locked_directory(tmp_path):
    spatial = tmp_path / "spatial.csv"
    cell = tmp_path / "cell.csv"
    nucleus = tmp_path / "nucleus.csv"
    for path in (spatial, cell, nucleus):
        path.write_text("x,y\n0,0\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    owner = OutputDirectoryRunLock(str(out_dir)).acquire()
    try:
        result = CliRunner().invoke(
            main,
            [
                "--spatial",
                str(spatial),
                "--cell-boundaries",
                str(cell),
                "--nucleus-boundaries",
                str(nucleus),
                "--out-dir",
                str(out_dir),
                "--no-show-sample",
            ],
        )
        assert result.exit_code == 1
        assert "another live CellScoper process" in result.output
        assert f"pid: {os.getpid()}" in result.output
    finally:
        owner.release()


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="requires O_NOFOLLOW")
def test_symbolic_link_lock_path_cannot_overwrite_user_output(tmp_path):
    sentinel = tmp_path / "important.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    (tmp_path / LOCK_FILENAME).symlink_to(sentinel)

    with pytest.raises(OutputDirectoryLockError, match="symbolic-link"):
        OutputDirectoryRunLock(str(tmp_path)).acquire()

    assert sentinel.read_text(encoding="utf-8") == "keep me"
