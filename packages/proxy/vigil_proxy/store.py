"""Persistence layer. One interface, two backends.

Slice 1 ships the SQLite backend (local mode) — the app is fully functional on SQLite alone.
The Redis backend (pub/sub + RedisVL vector index) arrives in a later slice behind `REDIS_URL`;
`make_store` is the single seam where it plugs in.

Migrations are idempotent: `CREATE TABLE IF NOT EXISTS` plus an `ADD COLUMN` helper that is a
no-op when the column already exists, so re-running setup is always safe.
"""

from __future__ import annotations

import abc
import json
from typing import Any

import aiosqlite

from .logging_config import get_logger, log_event
from .models import Step
from .settings import Settings

logger = get_logger("store")

_STEP_COLUMNS = [
    ("session_id", "TEXT NOT NULL"),
    ("step_index", "INTEGER NOT NULL"),
    ("model_requested", "TEXT"),
    ("model_used", "TEXT"),
    ("tool_name", "TEXT"),
    ("tool_args", "TEXT"),
    ("assistant_text", "TEXT"),
    ("prompt_tokens", "INTEGER"),
    ("completion_tokens", "INTEGER"),
    ("tokens_before_compression", "INTEGER"),
    ("tokens_after_compression", "INTEGER"),
    ("timestamp", "TEXT"),
    ("caused_state_mutation", "INTEGER DEFAULT 0"),
    ("breaker_override", "INTEGER DEFAULT 0"),
    ("breaker_state", "TEXT"),
]


class Store(abc.ABC):
    """Backend-agnostic persistence interface."""

    @abc.abstractmethod
    async def init(self) -> None: ...

    @abc.abstractmethod
    async def next_step_index(self, session_id: str) -> int: ...

    @abc.abstractmethod
    async def add_step(self, step: Step) -> int: ...

    @abc.abstractmethod
    async def append_step(self, step: Step) -> int: ...

    @abc.abstractmethod
    async def get_steps(self, session_id: str) -> list[Step]: ...

    @abc.abstractmethod
    async def list_sessions(self) -> list[str]: ...

    @abc.abstractmethod
    async def close(self) -> None: ...


class SQLiteStore(Store):
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL;")
        col_defs = ", ".join(f"{name} {decl}" for name, decl in _STEP_COLUMNS)
        await self._db.execute(
            f"CREATE TABLE IF NOT EXISTS steps (id INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs});"
        )
        # Idempotent forward migrations for any column added after a table already existed.
        await self._migrate_columns()
        # UNIQUE so a concurrent same-session race can never persist a duplicate index;
        # it also serves the ordered (session_id, step_index) lookups the watchdog uses.
        await self._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_steps_session "
            "ON steps(session_id, step_index);"
        )
        await self._db.commit()
        log_event(logger, 20, "store.init", backend="sqlite", path=self._db_path)

    async def _migrate_columns(self) -> None:
        assert self._db is not None
        cur = await self._db.execute("PRAGMA table_info(steps);")
        existing = {row["name"] for row in await cur.fetchall()}
        for name, decl in _STEP_COLUMNS:
            if name not in existing:
                await self._db.execute(f"ALTER TABLE steps ADD COLUMN {name} {decl};")
                log_event(logger, 20, "store.migrate", added_column=name)

    async def append_step(self, step: Step) -> int:
        """Atomically assign the next per-session step_index and insert; return the index.

        Assignment and insert are a single statement, so two concurrent same-session captures
        cannot read the same MAX and collide (the shared connection serializes statements; the
        UNIQUE index is a backstop). Prefer this over next_step_index + add_step on the hot path.
        """
        assert self._db is not None
        cur = await self._db.execute(
            """
            INSERT INTO steps (
                session_id, step_index, model_requested, model_used, tool_name, tool_args,
                assistant_text, prompt_tokens, completion_tokens, tokens_before_compression,
                tokens_after_compression, timestamp, caused_state_mutation, breaker_override,
                breaker_state
            ) VALUES (
                ?,
                (SELECT COALESCE(MAX(step_index), -1) + 1 FROM steps WHERE session_id = ?),
                ?,?,?,?,?,?,?,?,?,?,?,?,?
            );
            """,
            (
                step.session_id,
                step.session_id,
                step.model_requested,
                step.model_used,
                step.tool_name,
                json.dumps(step.tool_args) if step.tool_args is not None else None,
                step.assistant_text,
                step.prompt_tokens,
                step.completion_tokens,
                step.tokens_before_compression,
                step.tokens_after_compression,
                step.timestamp,
                int(step.caused_state_mutation),
                int(step.breaker_override),
                step.breaker_state,
            ),
        )
        await self._db.commit()
        row = await (
            await self._db.execute("SELECT step_index FROM steps WHERE id = ?;", (cur.lastrowid,))
        ).fetchone()
        return int(row["step_index"]) if row else 0

    async def next_step_index(self, session_id: str) -> int:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT COALESCE(MAX(step_index), -1) + 1 AS n FROM steps WHERE session_id = ?;",
            (session_id,),
        )
        row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def add_step(self, step: Step) -> int:
        assert self._db is not None
        cur = await self._db.execute(
            """
            INSERT INTO steps (
                session_id, step_index, model_requested, model_used, tool_name, tool_args,
                assistant_text, prompt_tokens, completion_tokens, tokens_before_compression,
                tokens_after_compression, timestamp, caused_state_mutation, breaker_override,
                breaker_state
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
            """,
            (
                step.session_id,
                step.step_index,
                step.model_requested,
                step.model_used,
                step.tool_name,
                json.dumps(step.tool_args) if step.tool_args is not None else None,
                step.assistant_text,
                step.prompt_tokens,
                step.completion_tokens,
                step.tokens_before_compression,
                step.tokens_after_compression,
                step.timestamp,
                int(step.caused_state_mutation),
                int(step.breaker_override),
                step.breaker_state,
            ),
        )
        await self._db.commit()
        return int(cur.lastrowid or 0)

    async def get_steps(self, session_id: str) -> list[Step]:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT * FROM steps WHERE session_id = ? ORDER BY step_index ASC;",
            (session_id,),
        )
        return [_row_to_step(row) for row in await cur.fetchall()]

    async def list_sessions(self) -> list[str]:
        assert self._db is not None
        cur = await self._db.execute("SELECT DISTINCT session_id FROM steps ORDER BY session_id;")
        return [row["session_id"] for row in await cur.fetchall()]

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


def _row_to_step(row: Any) -> Step:
    args = row["tool_args"]
    return Step(
        session_id=row["session_id"],
        step_index=row["step_index"],
        model_requested=row["model_requested"] or "",
        model_used=row["model_used"] or "",
        tool_name=row["tool_name"],
        tool_args=json.loads(args) if args else None,
        assistant_text=row["assistant_text"] or "",
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        tokens_before_compression=row["tokens_before_compression"],
        tokens_after_compression=row["tokens_after_compression"],
        timestamp=row["timestamp"] or "",
        caused_state_mutation=bool(row["caused_state_mutation"]),
        breaker_override=bool(row["breaker_override"]),
        breaker_state=row["breaker_state"],
    )


async def make_store(settings: Settings) -> Store:
    """Single seam for backend selection. Redis backend plugs in here in a later slice."""
    store: Store = SQLiteStore(settings.db_path)
    await store.init()
    return store
