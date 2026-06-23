from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signalforge import __version__

DEFAULT_HASH_SIZE_LIMIT_BYTES = 25 * 1024 * 1024


def write_run_manifest(
    output_dir: str | Path,
    *,
    run_type: str,
    parameters: Mapping[str, Any] | None = None,
    inputs: Mapping[str, str | Path | None] | None = None,
    outputs: Mapping[str, str | Path] | Iterable[str | Path] | None = None,
    as_of_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    code_cwd: str | Path | None = None,
    created_at_utc: str | None = None,
    hash_size_limit_bytes: int = DEFAULT_HASH_SIZE_LIMIT_BYTES,
) -> Path:
    """Build and write a stable machine-readable manifest for a run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        run_type=run_type,
        parameters=parameters,
        inputs=inputs,
        outputs=outputs,
        as_of_date=as_of_date,
        start_date=start_date,
        end_date=end_date,
        code_cwd=code_cwd,
        created_at_utc=created_at_utc,
        hash_size_limit_bytes=hash_size_limit_bytes,
    )
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest_path


def build_run_manifest(
    *,
    run_type: str,
    parameters: Mapping[str, Any] | None = None,
    inputs: Mapping[str, str | Path | None] | None = None,
    outputs: Mapping[str, str | Path] | Iterable[str | Path] | None = None,
    as_of_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    code_cwd: str | Path | None = None,
    created_at_utc: str | None = None,
    hash_size_limit_bytes: int = DEFAULT_HASH_SIZE_LIMIT_BYTES,
) -> dict[str, Any]:
    """Create a reusable run manifest payload."""
    normalized_inputs = _normalize_mapping(inputs or {})
    artifact_names = output_artifact_names(outputs)
    if "manifest.json" not in artifact_names:
        artifact_names.append("manifest.json")

    manifest: dict[str, Any] = {
        "run_type": run_type,
        "created_at_utc": created_at_utc or utc_now_iso(),
    }
    if as_of_date is not None:
        manifest["as_of_date"] = str(as_of_date)
    if start_date is not None:
        manifest["start_date"] = str(start_date)
    if end_date is not None:
        manifest["end_date"] = str(end_date)

    manifest.update(
        {
            "parameters": _json_safe(parameters or {}),
            "inputs": normalized_inputs,
            "input_file_metadata": {
                name: file_metadata(
                    path,
                    hash_size_limit_bytes=hash_size_limit_bytes,
                )
                for name, path in normalized_inputs.items()
            },
            "outputs": artifact_names,
            "code_metadata": git_code_metadata(cwd=code_cwd),
            "environment": environment_metadata(),
        }
    )
    return manifest


def file_metadata(
    path: str | Path | None,
    *,
    hash_size_limit_bytes: int = DEFAULT_HASH_SIZE_LIMIT_BYTES,
) -> dict[str, Any]:
    """Return safe metadata for an input file, hashing only small files."""
    metadata: dict[str, Any] = {
        "path": None if path is None else str(path),
        "exists": False,
        "size_bytes": None,
        "modified_at_utc": None,
        "sha256": None,
        "sha256_skipped": True,
        "sha256_skip_reason": "missing",
    }
    if path is None:
        return metadata

    file_path = Path(path)
    if not file_path.exists():
        return metadata

    stat = file_path.stat()
    metadata.update(
        {
            "exists": True,
            "size_bytes": int(stat.st_size),
            "modified_at_utc": _format_utc(
                datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            ),
        }
    )
    if not file_path.is_file():
        metadata["sha256_skip_reason"] = "not_a_file"
    elif stat.st_size > hash_size_limit_bytes:
        metadata["sha256_skip_reason"] = "size_above_threshold"
    else:
        metadata["sha256"] = _sha256_file(file_path)
        metadata["sha256_skipped"] = False
        metadata["sha256_skip_reason"] = None
    return metadata


def output_artifact_names(
    outputs: Mapping[str, str | Path] | Iterable[str | Path] | None,
) -> list[str]:
    """Normalize output artifact paths to stable file names."""
    if outputs is None:
        return []
    raw_outputs: Iterable[str | Path]
    if isinstance(outputs, Mapping):
        raw_outputs = outputs.values()
    else:
        raw_outputs = outputs

    names = []
    seen = set()
    for artifact in raw_outputs:
        artifact_path = Path(artifact)
        name = (
            artifact_path.as_posix()
            if not artifact_path.is_absolute() and artifact_path.parent != Path(".")
            else artifact_path.name
        )
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def git_code_metadata(cwd: str | Path | None = None) -> dict[str, str | bool | None]:
    """Return git metadata, falling back to null values outside git repositories."""
    try:
        commit = _git_stdout(["rev-parse", "HEAD"], cwd=cwd)
        branch = _git_stdout(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
        status = _git_stdout(["status", "--short"], cwd=cwd)
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_branch": None, "git_dirty": None}
    return {
        "git_commit": commit or None,
        "git_branch": branch or None,
        "git_dirty": bool(status),
    }


def environment_metadata() -> dict[str, str]:
    """Return runtime metadata that does not depend on the user's input files."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_name": "signalforge",
        "package_version": __version__,
    }


def utc_now_iso() -> str:
    return _format_utc(datetime.now(UTC))


def _normalize_mapping(values: Mapping[str, str | Path | None]) -> dict[str, str | None]:
    return {key: None if value is None else str(value) for key, value in values.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, set):
        return sorted(_json_safe(inner) for inner in value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_stdout(args: list[str], *, cwd: str | Path | None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
