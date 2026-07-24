"""Non-destructive, per-output-directory lock for CellScoper CLI runs."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import shlex
import socket
import stat
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


LOCK_FILENAME = ".cellscoper.run.lock"


class OutputDirectoryLockError(RuntimeError):
    """Raised when another process is actively writing to an output directory."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_metadata(fd: int) -> Dict[str, Any]:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        payload = os.read(fd, 1024 * 1024)
        parsed = json.loads(payload.decode("utf-8")) if payload else {}
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _write_metadata(fd: int, metadata: Dict[str, Any]) -> None:
    payload = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("Could not write CellScoper run-lock metadata")
        view = view[written:]
    os.fsync(fd)


def _owner_summary(metadata: Dict[str, Any]) -> str:
    if not metadata:
        return "owner metadata is not yet available"
    fields = (
        ("pid", metadata.get("pid")),
        ("host", metadata.get("host")),
        ("started", metadata.get("started_at")),
        ("command", metadata.get("command")),
    )
    return "\n".join(f"  {name}: {value}" for name, value in fields if value is not None)


class OutputDirectoryRunLock:
    """Hold an atomic OS lock while a process writes one output directory.

    The small JSON lock file is deliberately retained after a run as an audit
    record. The advisory lock, rather than the file's existence or a PID probe,
    defines ownership. Therefore a crash automatically releases ownership and a
    later run can safely replace stale metadata without deleting user outputs.
    """

    def __init__(self, out_dir: str) -> None:
        self.out_dir = os.path.realpath(os.path.abspath(out_dir))
        self.path = os.path.join(self.out_dir, LOCK_FILENAME)
        self._fd: Optional[int] = None
        self.metadata: Dict[str, Any] = {}

    @property
    def acquired(self) -> bool:
        return self._fd is not None

    def acquire(self) -> "OutputDirectoryRunLock":
        if self._fd is not None:
            return self

        os.makedirs(self.out_dir, exist_ok=True)
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise OutputDirectoryLockError(
                    f"Refusing symbolic-link run-lock path: {self.path}"
                ) from None
            raise
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OutputDirectoryLockError(
                    f"Run-lock path is not a regular file: {self.path}"
                )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                owner = _read_metadata(fd)
                raise OutputDirectoryLockError(
                    "Output directory is already in use by another live "
                    "CellScoper process.\n"
                    f"  out_dir: {self.out_dir}\n"
                    f"  lock: {self.path}\n"
                    f"{_owner_summary(owner)}\n"
                    "Wait for that run to finish or choose a different --out-dir."
                ) from None

            previous = _read_metadata(fd)
            metadata: Dict[str, Any] = {
                "schema_version": 1,
                "status": "running",
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
                "host": socket.gethostname(),
                "started_at": _utc_now(),
                "cwd": os.getcwd(),
                "argv": list(sys.argv),
                "command": shlex.join(sys.argv),
                "out_dir": self.out_dir,
            }
            if previous.get("status") == "running":
                metadata["recovered_stale_owner"] = {
                    key: previous.get(key)
                    for key in ("pid", "parent_pid", "host", "started_at", "command")
                    if previous.get(key) is not None
                }
            _write_metadata(fd, metadata)
            self._fd = fd
            self.metadata = metadata
            return self
        except BaseException:
            os.close(fd)
            raise

    def release(
        self,
        *,
        status: str = "completed",
        error_type: Optional[str] = None,
    ) -> None:
        fd = self._fd
        if fd is None:
            return
        try:
            metadata = dict(self.metadata)
            metadata["status"] = status
            metadata["released_at"] = _utc_now()
            if error_type:
                metadata["error_type"] = error_type
            try:
                try:
                    _write_metadata(fd, metadata)
                    self.metadata = metadata
                except OSError:
                    # Never keep a directory locked, or mask a pipeline error,
                    # solely because final audit metadata could not be written.
                    pass
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            self._fd = None

    def __enter__(self) -> "OutputDirectoryRunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release(
            status="completed" if exc_type is None else "failed",
            error_type=exc_type.__name__ if exc_type is not None else None,
        )
        return False
