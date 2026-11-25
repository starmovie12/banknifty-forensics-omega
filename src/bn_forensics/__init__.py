"""banknifty-forensics-omega core package (FILE-023).

Exposes version metadata, the deterministic exit-code contract (§10.3), the exception hierarchy,
and the stdlib-only primitives every module needs (DEF-07):

* atomic writes via temp file + ``os.replace`` (NFR-04) with a guarded writer that refuses
  protected roots such as ``raw/`` (NFR-06);
* path resolution that rejects traversal and symlinks on the target and every parent (NFR-05);
* app-private locks: ``fcntl.flock`` first, ``O_CREAT|O_EXCL`` PID-file fallback (§8.3);
* SHA-256 helpers, canonical JSON, IST clock (fixed in selftest for byte-identical output);
* environment scrubbing of broker tokens (NFR-05, NG-3).

This module imports only the standard library. It never imports ``core_stats`` or any data
loader, so the F1.2 and F5 isolation of ``skeptic_checks`` and ``twin_impl`` holds.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fnmatch
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

__version__ = "1.0.0"
PRD_VERSION = "3.0.0"
PROMPT_VERSION = "V5.0+Omega1.0+V3"
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------------------------
# Exit-code contract (§10.3, FR-19)
# ---------------------------------------------------------------------------------------------
EXIT_OK = 0
EXIT_PARTIAL = 2
EXIT_INPUT = 3
EXIT_INVALID = 4
EXIT_GAPS = 5
EXIT_CODES: dict[int, str] = {
    EXIT_OK: "OK",
    EXIT_PARTIAL: "PARTIAL",
    EXIT_INPUT: "INPUT_OR_SCHEMA_ERROR",
    EXIT_INVALID: "MODE_INVALID",
    EXIT_GAPS: "COMPLETE_WITH_GAPS",
}

# IST is the contract offset of every store timestamp (Q6). It is a data contract constant,
# not a session time: session open/close and bar counts come from data and era_calendar.yaml.
IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30), name="+05:30")
IST_OFFSET_MINUTES = 330

STAGES: tuple[str, ...] = tuple(f"Ω-S{i}" for i in range(10))
STAGE_STATES: dict[str, str] = {
    "Ω-S0": "INTAKE_QUARANTINE",
    "Ω-S1": "SPLIT_SEAL",
    "Ω-S2": "NULL_LAB_AUDIT",
    "Ω-S3": "PLAYGROUND_HUNT",
    "Ω-S4": "DOCKET_FREEZE",
    "Ω-S5": "COURT_SESSION",
    "Ω-S6": "SKEPTIC_VETO_GRADE",
    "Ω-S7": "LEDGER_INIT_TICK",
    "Ω-S8": "FUTILITY_AND_REPORT",
    "Ω-S9": "FUTILITY_AND_REPORT",
}
HUNT_MODES: tuple[str, ...] = tuple(f"H{i}" for i in range(1, 14))
OMEGA_GRADES: tuple[str, ...] = ("Ω-A", "Ω-B", "Ω-C")
RUN_ID_RE = re.compile(r"^(latest|[A-Za-z0-9_-]{1,64})$")
NA_DATA_MISSING = "N/A — DATA MISSING"


# ---------------------------------------------------------------------------------------------
# Exceptions (each carries its deterministic exit code)
# ---------------------------------------------------------------------------------------------
class BnfError(Exception):
    """Base error of the lab; ``exit_code`` maps it onto the §10.3 contract."""

    exit_code = EXIT_INPUT

    def __init__(self, message: str, *, tripwire_id: str | None = None) -> None:
        super().__init__(message)
        self.tripwire_id = tripwire_id


class InputError(BnfError):
    """Unreadable input, invalid argument or malformed file (exit 3)."""

    exit_code = EXIT_INPUT


class SchemaError(InputError):
    """JSON Schema validation failure of an artifact (exit 3, FR-18)."""


class PathViolation(InputError):
    """Path traversal, symlink, or write into a protected root (exit 3, NFR-05/06)."""


class LockContention(InputError):
    """Another bnf process holds a required lock (exit 3, §8.3)."""


class CorruptStateError(InputError):
    """Corrupt ledger or run state file (exit 3)."""


class InvalidPipeline(BnfError):
    """Critical tripwire breach: MODE: INVALID (exit 4)."""

    exit_code = EXIT_INVALID


class SealViolation(InvalidPipeline):
    """Playground code touched a sealed Court file (F31, exit 4)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, tripwire_id="F31")


class DocketTamper(InvalidPipeline):
    """Court docket changed after its SHA-256 freeze (F32, exit 4)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, tripwire_id="F32")


class RowConservationError(InvalidPipeline):
    """rows_in != rows_out + rows_dropped_logged (F6 / Q10, exit 4)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, tripwire_id="F6")


class PartialCheckpoint(BnfError):
    """Budget reached or unresolved verification demotion: resumable PARTIAL (exit 2)."""

    exit_code = EXIT_PARTIAL

    def __init__(self, message: str, *, stage: str = "", reason: str = "") -> None:
        super().__init__(message)
        self.stage = stage
        self.reason = reason or message


# ---------------------------------------------------------------------------------------------
# Clock (IST). Selftest fixes the clock so every artifact is byte-identical across runs.
# ---------------------------------------------------------------------------------------------
_FIXED_NOW: _dt.datetime | None = None


def set_fixed_clock(moment: _dt.datetime | None) -> None:
    """Pin ``now_ist()`` to ``moment`` (tz-aware) or release it with ``None``."""
    global _FIXED_NOW
    if moment is not None and moment.tzinfo is None:
        raise InputError("fixed clock must be timezone-aware")
    _FIXED_NOW = None if moment is None else moment.astimezone(IST)


def now_ist() -> _dt.datetime:
    """Current IST time, or the pinned selftest time."""
    if _FIXED_NOW is not None:
        return _FIXED_NOW
    return _dt.datetime.now(tz=IST).replace(microsecond=0)


def iso_ist(moment: _dt.datetime | None = None) -> str:
    """ISO-8601 IST timestamp with ``+05:30`` suffix and second resolution."""
    value = (moment or now_ist()).astimezone(IST).replace(microsecond=0)
    return value.isoformat()


# ---------------------------------------------------------------------------------------------
# Hashing and canonical JSON
# ---------------------------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hex SHA-256 of UTF-8 text."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, compact separators, UTF-8)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=_json_default, allow_nan=False)


def pretty_json(obj: Any) -> str:
    """Deterministic indented JSON with a trailing newline."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False,
                      default=_json_default, allow_nan=False) + "\n"


def sha256_json(obj: Any) -> str:
    """SHA-256 of the canonical JSON form."""
    return sha256_text(canonical_json(obj))


def sha256_tree(paths: Iterable[str | os.PathLike[str]]) -> str:
    """Order-independent combined SHA-256 over files (name + content)."""
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------
# Path safety (NFR-05) and the guarded writer (NFR-06)
# ---------------------------------------------------------------------------------------------
_PROTECTED_ROOTS: list[Path] = []


def protect_root(root: str | os.PathLike[str]) -> Path:
    """Register a read-only root (``raw/``, a source ``updl`` store). Writes under it fail."""
    resolved = Path(root).expanduser().resolve()
    if resolved not in _PROTECTED_ROOTS:
        _PROTECTED_ROOTS.append(resolved)
    return resolved


def unprotect_all() -> None:
    """Clear the protected-root registry (used between independent CLI invocations)."""
    _PROTECTED_ROOTS.clear()


def protected_roots() -> tuple[Path, ...]:
    """Currently registered protected roots."""
    return tuple(_PROTECTED_ROOTS)


def _check_no_symlinks(path: Path, root: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise PathViolation(f"symlink rejected: {current}")
        if current == root or current.parent == current:
            break
        current = current.parent


def safe_path(path: str | os.PathLike[str], allowed_root: str | os.PathLike[str]) -> Path:
    """Resolve ``path`` strictly inside ``allowed_root``.

    Rejects ``..`` traversal, absolute escapes, and symlinks on the target or any intermediate
    parent up to the root (Q1, NFR-05). Returns the absolute, non-resolved-through-links path.
    """
    root = Path(allowed_root).expanduser().absolute()
    if root.is_symlink():
        raise PathViolation(f"allowed root is a symlink: {root}")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.normpath(candidate.absolute()))
    root_norm = Path(os.path.normpath(root))
    if not candidate.is_relative_to(root_norm):
        raise PathViolation(f"path escapes allowed root: {path} (root {root})")
    _check_no_symlinks(candidate, root_norm)
    if candidate.exists() and not candidate.resolve().is_relative_to(root_norm.resolve()):
        raise PathViolation(f"resolved path escapes allowed root: {candidate}")
    return candidate


def _assert_writable(path: Path) -> None:
    target = Path(os.path.normpath(path.absolute()))
    resolved_parent = target.parent.resolve() if target.parent.exists() else target.parent
    for root in _PROTECTED_ROOTS:
        if target.is_relative_to(root) or resolved_parent.is_relative_to(root):
            raise PathViolation(f"guarded writer refused protected root {root}: {target}")
    if target.is_symlink():
        raise PathViolation(f"refusing to write through a symlink: {target}")


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """Write bytes atomically: temp file in the same directory, fsync, ``os.replace``."""
    target = Path(path)
    _assert_writable(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
    return target


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    """Atomic UTF-8 text write with LF line endings."""
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> Path:
    """Atomic deterministic JSON write."""
    return atomic_write_text(path, pretty_json(obj))


def append_line(path: str | os.PathLike[str], line: str) -> None:
    """Append one complete line (JSONL logs). Partial trailing lines are trimmed on resume."""
    target = Path(path)
    _assert_writable(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")
        handle.flush()


def truncate_incomplete_jsonl(path: str | os.PathLike[str]) -> int:
    """Drop a trailing line that is not valid JSON (NFR-04 resume rule). Returns lines kept."""
    target = Path(path)
    if not target.exists():
        return 0
    raw = target.read_text(encoding="utf-8")
    lines = raw.split("\n")
    kept: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            break
        kept.append(line)
    text = "".join(f"{line}\n" for line in kept)
    if text != raw:
        atomic_write_text(target, text)
    return len(kept)


def tree_manifest(root: str | os.PathLike[str]) -> dict[str, str]:
    """``relative path -> sha256`` for every regular file under ``root`` (pre/post checks)."""
    base = Path(root)
    if not base.exists():
        return {}
    manifest: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if path.is_file() and not path.is_symlink():
            manifest[path.relative_to(base).as_posix()] = sha256_file(path)
    return manifest


# ---------------------------------------------------------------------------------------------
# YAML / JSON loading (safe loader only; data is never executed, R26 / F28)
# ---------------------------------------------------------------------------------------------
def load_yaml(path: str | os.PathLike[str]) -> Any:
    """Load YAML with the safe loader; raises InputError on unreadable/malformed input."""
    import yaml  # local import keeps this module stdlib-only at import time

    try:
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise InputError(f"file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise InputError(f"malformed YAML {path}: {exc}") from exc


def dump_yaml(obj: Any) -> str:
    """Deterministic YAML text (sorted keys, block style, unicode kept)."""
    import yaml

    return yaml.safe_dump(obj, sort_keys=True, allow_unicode=True, default_flow_style=False,
                          width=120)


def atomic_write_yaml(path: str | os.PathLike[str], obj: Any) -> Path:
    """Atomic deterministic YAML write."""
    return atomic_write_text(path, dump_yaml(obj))


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load JSON; raises CorruptStateError on malformed content."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise InputError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CorruptStateError(f"corrupt JSON {path}: {exc}") from exc


# ---------------------------------------------------------------------------------------------
# Repository / packaged asset resolution
# ---------------------------------------------------------------------------------------------
_ASSET_KINDS = ("config", "schemas", "prompts", "fixtures")


def asset_dir(kind: str) -> Path:
    """Directory of ``config``, ``schemas``, ``prompts`` or ``fixtures``.

    Editable/dev installs read the repository root; wheel installs read ``_assets`` inside the
    package (force-included by ``pyproject.toml``).
    """
    if kind not in _ASSET_KINDS:
        raise InputError(f"unknown asset kind: {kind}")
    repo_root = Path(__file__).resolve().parents[2]
    repo_candidate = repo_root / ("tests/fixtures" if kind == "fixtures" else kind)
    if repo_candidate.is_dir():
        return repo_candidate
    packaged = Path(__file__).resolve().parent / "_assets" / kind
    if packaged.is_dir():
        return packaged
    raise InputError(f"asset directory not found for {kind!r}")


def resolve_config_path(path: str | os.PathLike[str] | None) -> Path:
    """Use ``path`` if it exists, else the packaged/repository default config."""
    if path is not None:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate
        default_rel = Path("config/default_config.yaml")
        if candidate.as_posix() != default_rel.as_posix():
            raise InputError(f"config file not found: {path}")
    return asset_dir("config") / "default_config.yaml"


def load_config(path: str | os.PathLike[str] | None = None,
                overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load ``default_config.yaml`` (FILE-012) and apply dotted-key overrides."""
    cfg = load_yaml(resolve_config_path(path))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != "1.0":
        raise InputError("config must be a mapping with schema_version '1.0'")
    for dotted, value in (overrides or {}).items():
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return cfg


# ---------------------------------------------------------------------------------------------
# App-private locks (§8.3, NFR-04): flock first, O_CREAT|O_EXCL PID file fallback
# ---------------------------------------------------------------------------------------------
def lock_dir() -> Path:
    """``BNF_LOCK_DIR`` or ``~/.bnf/locks`` (outside Android shared storage)."""
    return Path(os.environ.get("BNF_LOCK_DIR", "~/.bnf/locks")).expanduser()


def _process_start_marker(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat")
    try:
        fields = stat.read_text(encoding="ascii").rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, IndexError):
        return "unknown"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class AppLock:
    """Exclusive non-blocking lock on a logical resource path.

    The lock file lives at ``BNF_LOCK_DIR/<sha256(abs_path)[:16]>.lock``. ``fcntl.flock`` is tried
    first; when the filesystem does not support it (``OSError`` other than contention) an
    ``O_CREAT|O_EXCL`` file holding PID and process start time is used. A fallback lock whose PID
    no longer exists is stale and is taken over. Contention raises :class:`LockContention`.
    """

    def __init__(self, resource: str | os.PathLike[str], *, force_fallback: bool = False) -> None:
        self.resource = str(Path(resource).expanduser().absolute())
        digest = sha256_text(self.resource)[:16]
        self.path = lock_dir() / f"{digest}.lock"
        self.mode = "none"
        self._fd: int | None = None
        self._force_fallback = force_fallback

    def _try_flock(self) -> bool:
        import fcntl

        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise LockContention(f"lock held by another bnf process: {self.resource}") from exc
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()} {self.resource}\n".encode())
        self._fd = fd
        self.mode = "flock"
        return True

    def _try_excl(self) -> None:
        payload = f"{os.getpid()} {_process_start_marker(os.getpid())} {self.resource}\n"
        for _attempt in range(2):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                try:
                    owner = self.path.read_text(encoding="utf-8").split()
                    pid, start = int(owner[0]), owner[1]
                except (OSError, ValueError, IndexError):
                    pid, start = -1, "unknown"
                alive = _pid_alive(pid) and (start in ("unknown", _process_start_marker(pid)))
                if alive:
                    raise LockContention(
                        f"lock held by pid {pid}: {self.resource}") from None
                with contextlib.suppress(FileNotFoundError):
                    self.path.unlink()
                continue
            os.write(fd, payload.encode())
            os.close(fd)
            self.mode = "o_excl"
            return
        raise LockContention(f"could not take stale lock: {self.resource}")

    def acquire(self) -> AppLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._force_fallback or not self._try_flock():
            self._try_excl()
        return self

    def release(self) -> None:
        if self.mode == "flock" and self._fd is not None:
            import fcntl

            with contextlib.suppress(OSError):
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
        elif self.mode == "o_excl":
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
        self.mode = "none"

    def __enter__(self) -> AppLock:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


@contextlib.contextmanager
def acquire_locks(resources: Iterable[str | os.PathLike[str]], *,
                  force_fallback: bool = False) -> Iterator[list[AppLock]]:
    """Take exclusive locks in fixed lexical order (deadlock-free, §8.3)."""
    ordered = sorted({str(Path(r).expanduser().absolute()) for r in resources})
    with contextlib.ExitStack() as stack:
        locks = [stack.enter_context(AppLock(r, force_fallback=force_fallback)) for r in ordered]
        yield locks


# ---------------------------------------------------------------------------------------------
# Secret hygiene (NFR-05, NG-3, X31)
# ---------------------------------------------------------------------------------------------
_SECRET_NAMES = ("UPSTOX_RO_TOKEN",)
_SECRET_PATTERNS = ("*TOKEN*", "*SECRET*")
TOKEN_LIKE_RE = re.compile(
    r"(eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,})"
    r"|((?i:token|secret)[^\n]{0,24}?[:=\s\"']+[A-Fa-f0-9]{32,})"
    r"|((?i:token|secret)[^\n]{0,24}?[:=\s\"']+[A-Za-z0-9+/]{32,}={0,2})"
)


def scrub_secret_env(environ: dict[str, str] | None = None) -> list[str]:
    """Remove broker tokens and ``*TOKEN*``/``*SECRET*`` variables; returns removed names."""
    env = os.environ if environ is None else environ
    removed: list[str] = []
    for name in sorted(env):
        upper = name.upper()
        if upper in _SECRET_NAMES or any(fnmatch.fnmatchcase(upper, p) for p in _SECRET_PATTERNS):
            removed.append(name)
    for name in removed:
        env.pop(name, None)
    return removed


def find_token_like(text: str) -> list[str]:
    """Token-like strings (JWT ``eyJ…``; 32+ hex/base64 next to token/secret) in ``text``."""
    return [m.group(0)[:48] for m in TOKEN_LIKE_RE.finditer(text)]


# ---------------------------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------------------------
def parse_hhmm(value: str) -> int:
    """``'HH:MM'`` -> minutes after midnight (for times read from config, never literals)."""
    match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", str(value).strip())
    if not match:
        raise InputError(f"invalid HH:MM time: {value!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


def resolve_seed(seed: int | None = None) -> int:
    """Seed precedence: explicit argument, ``BNF_SEED``, then 42 (NFR-03)."""
    if seed is not None:
        return int(seed)
    env_seed = os.environ.get("BNF_SEED", "").strip()
    if env_seed:
        try:
            return int(env_seed)
        except ValueError as exc:
            raise InputError(f"BNF_SEED must be an integer: {env_seed!r}") from exc
    return DEFAULT_SEED


def validate_run_id(run_id: str) -> str:
    """``latest`` or ``^[A-Za-z0-9_-]{1,64}$``; anything else is exit 3 (NFR-05)."""
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise InputError(f"invalid --run-id {run_id!r}: must be 'latest' or [A-Za-z0-9_-]{{1,64}}")
    return run_id


class EventLog:
    """Structured JSONL log (§13): ``timestamp_ist, level, run_id, stage, role, event, payload``."""

    LEVELS = ("DEBUG", "INFO", "WARN", "ERROR", "CRITICAL")

    def __init__(self, path: str | os.PathLike[str] | None, *, run_id: str = "",
                 data_version: str = "", echo_json: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        self.run_id = run_id
        self.data_version = data_version
        self.echo_json = echo_json
        self.records: list[dict[str, Any]] = []

    def emit(self, event: str, payload: Mapping[str, Any] | None = None, *, level: str = "INFO",
             stage: str = "", role: str = "") -> dict[str, Any]:
        if level not in self.LEVELS:
            raise InputError(f"invalid log level {level}")
        record = {
            "timestamp_ist": iso_ist(),
            "level": level,
            "run_id": self.run_id,
            "stage": stage,
            "role": role,
            "data_version": self.data_version,
            "event": event,
            "payload": dict(payload or {}),
        }
        self.records.append(record)
        line = canonical_json(record)
        if self.path is not None:
            append_line(self.path, line)
        if self.echo_json:
            sys.stdout.write(line + "\n")
        return record


def monotonic_minutes(start: float) -> float:
    """Elapsed wall-clock minutes since ``start`` (``time.monotonic`` origin)."""
    return (time.monotonic() - start) / 60.0


__all__ = [
    "DEFAULT_SEED",
    "EXIT_CODES",
    "EXIT_GAPS",
    "EXIT_INPUT",
    "EXIT_INVALID",
    "EXIT_OK",
    "EXIT_PARTIAL",
    "HUNT_MODES",
    "IST",
    "IST_OFFSET_MINUTES",
    "NA_DATA_MISSING",
    "OMEGA_GRADES",
    "PRD_VERSION",
    "PROMPT_VERSION",
    "RUN_ID_RE",
    "STAGES",
    "STAGE_STATES",
    "AppLock",
    "BnfError",
    "CorruptStateError",
    "DocketTamper",
    "EventLog",
    "InputError",
    "InvalidPipeline",
    "LockContention",
    "PartialCheckpoint",
    "PathViolation",
    "RowConservationError",
    "SchemaError",
    "SealViolation",
    "__version__",
    "acquire_locks",
    "append_line",
    "asset_dir",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "atomic_write_yaml",
    "canonical_json",
    "dump_yaml",
    "find_token_like",
    "iso_ist",
    "load_config",
    "load_json",
    "load_yaml",
    "lock_dir",
    "monotonic_minutes",
    "now_ist",
    "parse_hhmm",
    "pretty_json",
    "protect_root",
    "protected_roots",
    "resolve_config_path",
    "resolve_seed",
    "safe_path",
    "scrub_secret_env",
    "set_fixed_clock",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "sha256_tree",
    "tree_manifest",
    "truncate_incomplete_jsonl",
    "unprotect_all",
    "validate_run_id",
]
