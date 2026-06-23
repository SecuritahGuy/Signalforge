from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ScriptStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass
class ScriptInfo:
    name: str
    description: str
    path: str
    args: list[dict] = field(default_factory=list)


@dataclass
class RunState:
    run_id: str
    script_name: str
    status: ScriptStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    output: list[str] = field(default_factory=list)
