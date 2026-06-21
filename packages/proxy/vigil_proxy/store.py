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
from datetime import UTC
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
    ("sim_score", "REAL"),
    ("tool_entropy", "REAL"),
    ("state_penalty", "REAL"),
    ("final_score", "REAL"),
    ("watchdog_breach", "INTEGER DEFAULT 0"),
    ("watchdog_streak", "INTEGER DEFAULT 0"),
    ("watchdog_tripped", "INTEGER DEFAULT 0"),
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
    async def recent_steps(self, limit: int) -> list[Step]: ...

    @abc.abstractmethod
    async def list_sessions(self) -> list[str]: ...

    @abc.abstractmethod
    async def update_breaker_fields(
        self, session_id: str, step_index: int, breaker_state: str | None, breaker_override: bool
    ) -> None: ...

    @abc.abstractmethod
    async def get_breaker(self, session_id: str) -> dict | None: ...

    @abc.abstractmethod
    async def set_breaker(self, session_id: str, data: dict) -> None: ...

    @abc.abstractmethod
    async def cache_exchange(
        self,
        session_id: str,
        step_index: int,
        request_hash: str,
        request: dict,
        response: dict,
        model: str | None,
    ) -> None: ...

    @abc.abstractmethod
    async def get_exchanges(self, session_id: str) -> list[dict]: ...

    @abc.abstractmethod
    async def get_cached_response(self, request_hash: str) -> dict | None: ...

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
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS breaker_states ("
            "session_id TEXT PRIMARY KEY, data TEXT, updated_at TEXT);"
        )
        # Forensic cache (spec 4.7): the (request -> response) exchange per step, content-addressed
        # by request_hash for cached-trace replay. request_json is the request BODY only (never a
        # provider key — keys live in headers, which we never store).
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS exchanges ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
            "step_index INTEGER NOT NULL, request_hash TEXT NOT NULL, request_json TEXT NOT NULL, "
            "response_json TEXT NOT NULL, model TEXT, created_at TEXT);"
        )
        await self._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_exchanges_session "
            "ON exchanges(session_id, step_index);"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_exchanges_hash ON exchanges(request_hash);"
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
        values = _step_values(step)
        # Build the INSERT from _STEP_COLUMNS so adding a column never desyncs the placeholders;
        # step_index is computed in-statement rather than passed.
        cols = [name for name, _ in _STEP_COLUMNS]
        exprs: list[str] = []
        params: list[object] = []
        for name in cols:
            if name == "step_index":
                exprs.append(
                    "(SELECT COALESCE(MAX(step_index), -1) + 1 FROM steps WHERE session_id = ?)"
                )
                params.append(step.session_id)
            else:
                exprs.append("?")
                params.append(values[name])
        sql = f"INSERT INTO steps ({', '.join(cols)}) VALUES ({', '.join(exprs)});"
        cur = await self._db.execute(sql, params)
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
        values = _step_values(step)
        cols = [name for name, _ in _STEP_COLUMNS]
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO steps ({', '.join(cols)}) VALUES ({placeholders});"
        cur = await self._db.execute(sql, [values[name] for name in cols])
        await self._db.commit()
        return int(cur.lastrowid or 0)

    async def get_steps(self, session_id: str) -> list[Step]:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT * FROM steps WHERE session_id = ? ORDER BY step_index ASC;",
            (session_id,),
        )
        return [_row_to_step(row) for row in await cur.fetchall()]

    async def recent_steps(self, limit: int) -> list[Step]:
        """Most recent `limit` steps across all sessions, in chronological order.

        A single bounded query so dashboard snapshots never load the full history into memory.
        """
        assert self._db is not None
        cur = await self._db.execute("SELECT * FROM steps ORDER BY id DESC LIMIT ?;", (limit,))
        rows = list(await cur.fetchall())
        return [_row_to_step(row) for row in reversed(rows)]

    async def list_sessions(self) -> list[str]:
        assert self._db is not None
        cur = await self._db.execute("SELECT DISTINCT session_id FROM steps ORDER BY session_id;")
        return [row["session_id"] for row in await cur.fetchall()]

    async def update_breaker_fields(
        self, session_id: str, step_index: int, breaker_state: str | None, breaker_override: bool
    ) -> None:
        """Write the breaker state onto an already-persisted step row (the breaker verdict is
        computed after the step's index is assigned)."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE steps SET breaker_state = ?, breaker_override = ? "
            "WHERE session_id = ? AND step_index = ?;",
            (breaker_state, int(breaker_override), session_id, step_index),
        )
        await self._db.commit()

    async def get_breaker(self, session_id: str) -> dict | None:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT data FROM breaker_states WHERE session_id = ?;", (session_id,)
        )
        row = await cur.fetchone()
        if row is None or row["data"] is None:
            return None
        parsed = json.loads(row["data"])
        return parsed if isinstance(parsed, dict) else None

    async def set_breaker(self, session_id: str, data: dict) -> None:
        assert self._db is not None
        from datetime import datetime

        await self._db.execute(
            "INSERT INTO breaker_states (session_id, data, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET data = excluded.data, "
            "updated_at = excluded.updated_at;",
            (session_id, json.dumps(data), datetime.now(UTC).isoformat()),
        )
        await self._db.commit()

    async def cache_exchange(
        self,
        session_id: str,
        step_index: int,
        request_hash: str,
        request: dict,
        response: dict,
        model: str | None,
    ) -> None:
        """Idempotent: re-capturing the same (session, step) refreshes the cached exchange."""
        assert self._db is not None
        from datetime import datetime

        await self._db.execute(
            "INSERT INTO exchanges "
            "(session_id, step_index, request_hash, request_json, response_json, model, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id, step_index) DO UPDATE SET "
            "request_hash = excluded.request_hash, request_json = excluded.request_json, "
            "response_json = excluded.response_json, model = excluded.model;",
            (
                session_id,
                step_index,
                request_hash,
                json.dumps(request),
                json.dumps(response),
                model,
                datetime.now(UTC).isoformat(),
            ),
        )
        await self._db.commit()

    async def get_exchanges(self, session_id: str) -> list[dict]:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT step_index, request_hash, request_json, response_json, model "
            "FROM exchanges WHERE session_id = ? ORDER BY step_index ASC;",
            (session_id,),
        )
        return [
            {
                "step_index": row["step_index"],
                "request_hash": row["request_hash"],
                "request": json.loads(row["request_json"]),
                "response": json.loads(row["response_json"]),
                "model": row["model"],
            }
            for row in await cur.fetchall()
        ]

    async def get_cached_response(self, request_hash: str) -> dict | None:
        """Content-addressed lookup: any exchange with this request hash (identical content =>
        identical response, so replay serves it without ever calling upstream)."""
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT response_json FROM exchanges WHERE request_hash = ? LIMIT 1;", (request_hash,)
        )
        row = await cur.fetchone()
        if row is None or row["response_json"] is None:
            return None
        parsed = json.loads(row["response_json"])
        return parsed if isinstance(parsed, dict) else None

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


def _step_values(step: Step) -> dict[str, object]:
    """Column -> value for every persisted column. Keyed by name so callers build SQL from
    _STEP_COLUMNS and a new column can never drift the placeholder order."""
    return {
        "session_id": step.session_id,
        "step_index": step.step_index,
        "model_requested": step.model_requested,
        "model_used": step.model_used,
        "tool_name": step.tool_name,
        "tool_args": json.dumps(step.tool_args) if step.tool_args is not None else None,
        "assistant_text": step.assistant_text,
        "prompt_tokens": step.prompt_tokens,
        "completion_tokens": step.completion_tokens,
        "tokens_before_compression": step.tokens_before_compression,
        "tokens_after_compression": step.tokens_after_compression,
        "timestamp": step.timestamp,
        "caused_state_mutation": int(step.caused_state_mutation),
        "sim_score": step.sim_score,
        "tool_entropy": step.tool_entropy,
        "state_penalty": step.state_penalty,
        "final_score": step.final_score,
        "watchdog_breach": int(step.watchdog_breach),
        "watchdog_streak": step.watchdog_streak,
        "watchdog_tripped": int(step.watchdog_tripped),
        "breaker_override": int(step.breaker_override),
        "breaker_state": step.breaker_state,
    }


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
        sim_score=row["sim_score"],
        tool_entropy=row["tool_entropy"],
        state_penalty=row["state_penalty"],
        final_score=row["final_score"],
        watchdog_breach=bool(row["watchdog_breach"]),
        watchdog_streak=row["watchdog_streak"] or 0,
        watchdog_tripped=bool(row["watchdog_tripped"]),
        breaker_override=bool(row["breaker_override"]),
        breaker_state=row["breaker_state"],
    )


async def make_store(settings: Settings) -> Store:
    """Single seam for backend selection. Redis backend plugs in here in a later slice."""
    store: Store = SQLiteStore(settings.db_path)
    await store.init()
    return store
