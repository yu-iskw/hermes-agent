"""Scope-aware adapters for the MCP runtime's process-global registries.

``tools.mcp_tool`` was designed around one long-lived connection per logical
server name. In a shared gateway that is not a sufficient security boundary:
per-user OAuth needs the authenticated transport *and* its connection-health
state to follow the requesting human.

This module provides dict/set-compatible adapters so existing ``mcp_tool``
call sites continue to use logical names while storage is keyed internally by
``server@@<opaque OAuthScope>``. There is deliberately no "find any connection
with the same server name" fallback.

The adapters are installed only for explicit ``per_user`` mode. Historical
``shared`` OAuth therefore keeps the original runtime objects and semantics
byte-for-byte outside this module.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from typing import Any, Iterable, Iterator

from tools.mcp_oauth_identity import (
    McpOAuthScope,
    MissingMcpOAuthIdentityError,
    connection_registry_key,
    get_oauth_identity_mode,
    try_resolve_oauth_scope,
)

_TASK_SCOPES: "weakref.WeakKeyDictionary[asyncio.Task, dict[str, McpOAuthScope]]" = (
    weakref.WeakKeyDictionary()
)
_TASK_SCOPE_LOCK = threading.Lock()
_INSTALL_LOCK = threading.Lock()
_MISSING_SCOPE_SUFFIX = "@@<missing-authenticated-identity>"


def _task_scope(server_name: str) -> McpOAuthScope | None:
    # Request context wins. This lets a shared gateway worker safely handle
    # Alice and then Bob without inheriting Alice's transport identity.
    ambient = try_resolve_oauth_scope()
    if ambient is not None:
        return ambient

    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is None:
        return None
    with _TASK_SCOPE_LOCK:
        scopes = _TASK_SCOPES.get(task)
        return scopes.get(server_name) if scopes else None


def bind_runtime_scope(server_name: str, scope: McpOAuthScope) -> None:
    """Pin a resolved OAuth scope to the current long-lived server task."""
    if not scope.is_per_user:
        return
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is None:
        # Sync callers already carry their gateway ContextVars. Never create a
        # process-global fallback just to make a lookup convenient.
        return
    with _TASK_SCOPE_LOCK:
        current = dict(_TASK_SCOPES.get(task) or {})
        current[server_name] = scope
        _TASK_SCOPES[task] = current


def _is_internal_scoped_key(key: str) -> bool:
    return "@@u-v1-" in key or key.endswith(_MISSING_SCOPE_SUFFIX)


def _logical_state_key(server_name: str) -> str:
    """Return exact internal state key for the current OAuth principal.

    Availability/error state may be recorded before a human is bound (for
    example a headless startup probe). Such state is put in a dedicated
    missing-identity bucket. It is never credential state and, critically,
    never blocks or poisons a later authenticated user's bucket.
    """
    scope = _task_scope(server_name)
    if scope is None:
        return f"{server_name}{_MISSING_SCOPE_SUFFIX}"
    return connection_registry_key(server_name, scope)


class _ScopedNameMixin:
    """Common name translation for per-user OAuth runtime state."""

    oauth_servers: set[str]

    def _mark_oauth_server(self, server_name: str) -> None:
        self.oauth_servers.add(server_name)

    def _read_key(self, key: Any) -> Any:
        if not isinstance(key, str) or key not in self.oauth_servers:
            return key
        if _is_internal_scoped_key(key):
            return key
        return _logical_state_key(key)

    def _write_key(self, key: Any) -> Any:
        return self._read_key(key)

    def _visible_logical_keys(self, raw_keys: Iterable[Any]) -> list[Any]:
        """Project internal keys into the current request's logical view.

        ``mcp_tool`` takes snapshots with ``dict(_servers)`` and
        ``set(_server_connecting)``. Returning logical keys here preserves
        those public/status semantics without exposing another user's entries.
        """
        raw = list(raw_keys)
        visible: list[Any] = []
        raw_set = set(raw)

        for key in raw:
            if not isinstance(key, str):
                visible.append(key)
            elif key not in self.oauth_servers and not _is_internal_scoped_key(key):
                visible.append(key)

        for server_name in self.oauth_servers:
            if _logical_state_key(server_name) in raw_set:
                visible.append(server_name)
        return visible


class ScopedMCPServerRegistry(_ScopedNameMixin, dict):
    """Exact per-user registry for long-lived OAuth ``MCPServerTask`` objects."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        dict.__init__(self, initial or {})
        self.oauth_servers: set[str] = set()

    def mark_oauth_server(self, server_name: str) -> None:
        self._mark_oauth_server(server_name)
        # Anything at the historical raw key predates the per-user boundary;
        # ownership is unknowable, so it cannot be assigned to the first user.
        dict.pop(self, server_name, None)

    def _connection_key_for_write(self, key: Any) -> Any:
        if not isinstance(key, str) or key not in self.oauth_servers:
            return key
        if _is_internal_scoped_key(key):
            return key
        scope = _task_scope(key)
        if scope is None:
            raise MissingMcpOAuthIdentityError(
                f"refusing to register OAuth MCP server {key!r} without an "
                "authenticated per-user scope"
            )
        return connection_registry_key(key, scope)

    def __getitem__(self, key: Any) -> Any:
        return dict.__getitem__(self, self._read_key(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        dict.__setitem__(self, self._connection_key_for_write(key), value)

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, self._read_key(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return dict.get(self, self._read_key(key), default)

    def pop(self, key: Any, default: Any = ...):
        resolved = self._read_key(key)
        if default is ...:
            return dict.pop(self, resolved)
        return dict.pop(self, resolved, default)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        return dict.setdefault(self, self._connection_key_for_write(key), default)

    def keys(self):
        # ``dict(self)`` consults keys()+__getitem__ for dict subclasses.
        return self._visible_logical_keys(dict.keys(self))


class ScopedNameDict(_ScopedNameMixin, dict):
    """Per-user view for connect errors/backoff/circuit-breaker dictionaries."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        dict.__init__(self, initial or {})
        self.oauth_servers: set[str] = set()

    def mark_oauth_server(self, server_name: str) -> None:
        self._mark_oauth_server(server_name)
        # Pre-bound state belongs to no authenticated principal.
        dict.pop(self, server_name, None)

    def __getitem__(self, key: Any) -> Any:
        return dict.__getitem__(self, self._read_key(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        dict.__setitem__(self, self._write_key(key), value)

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, self._read_key(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return dict.get(self, self._read_key(key), default)

    def pop(self, key: Any, default: Any = ...):
        resolved = self._read_key(key)
        if default is ...:
            return dict.pop(self, resolved)
        return dict.pop(self, resolved, default)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        return dict.setdefault(self, self._write_key(key), default)

    def keys(self):
        return self._visible_logical_keys(dict.keys(self))


class ScopedNameSet(_ScopedNameMixin, set):
    """Per-user view for the MCP ``_server_connecting`` deduplication set."""

    def __init__(self, initial: Iterable[str] | None = None) -> None:
        set.__init__(self, initial or ())
        self.oauth_servers: set[str] = set()

    def mark_oauth_server(self, server_name: str) -> None:
        self._mark_oauth_server(server_name)
        set.discard(self, server_name)

    def add(self, element: Any) -> None:
        set.add(self, self._write_key(element))

    def discard(self, element: Any) -> None:
        set.discard(self, self._read_key(element))

    def remove(self, element: Any) -> None:
        set.remove(self, self._read_key(element))

    def __contains__(self, element: object) -> bool:
        return set.__contains__(self, self._read_key(element))

    def update(self, *others: Iterable[Any]) -> None:
        for other in others:
            for element in other:
                self.add(element)

    def difference_update(self, *others: Iterable[Any]) -> None:
        for other in others:
            for element in other:
                self.discard(element)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._visible_logical_keys(list(set.__iter__(self))))


class PersistentPerUserLazyConfigs(dict):
    """Keep OAuth server config available for every user's first connection."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        super().__init__(initial or {})
        self.oauth_servers: set[str] = set()

    def mark_oauth_server(self, server_name: str) -> None:
        self.oauth_servers.add(server_name)

    def pop(self, key: Any, default: Any = ...):
        if isinstance(key, str) and key in self.oauth_servers and key in self:
            return self[key]
        if default is ...:
            return dict.pop(self, key)
        return dict.pop(self, key, default)


def _wrap_scoped_dict(mcp_tool, attr_name: str, server_name: str) -> None:
    current = getattr(mcp_tool, attr_name)
    if not isinstance(current, ScopedNameDict):
        current = ScopedNameDict(dict(current))
        setattr(mcp_tool, attr_name, current)
    current.mark_oauth_server(server_name)


def prepare_oauth_server_runtime(server_name: str) -> None:
    """Install/mark per-user runtime views before principal resolution.

    ``shared`` mode returns immediately so existing single-user CLI/profile
    deployments keep the original plain dict/set objects and lifecycle.
    Invalid configuration propagates from ``get_oauth_identity_mode`` rather
    than silently falling back to shared semantics.
    """
    if get_oauth_identity_mode() != "per_user":
        return

    from tools import mcp_tool

    with _INSTALL_LOCK:
        servers = mcp_tool._servers
        if not isinstance(servers, ScopedMCPServerRegistry):
            servers = ScopedMCPServerRegistry(dict(servers))
            mcp_tool._servers = servers
        servers.mark_oauth_server(server_name)

        connecting = mcp_tool._server_connecting
        if not isinstance(connecting, ScopedNameSet):
            connecting = ScopedNameSet(set(connecting))
            mcp_tool._server_connecting = connecting
        connecting.mark_oauth_server(server_name)

        for attr_name in (
            "_server_connect_errors",
            "_server_error_counts",
            "_server_breaker_opened_at",
            "_server_connect_retry_after",
            "_server_connect_failures",
        ):
            _wrap_scoped_dict(mcp_tool, attr_name, server_name)

        lazy = mcp_tool._lazy_server_configs
        if not isinstance(lazy, PersistentPerUserLazyConfigs):
            lazy = PersistentPerUserLazyConfigs(dict(lazy))
            mcp_tool._lazy_server_configs = lazy
        lazy.mark_oauth_server(server_name)

        if server_name not in lazy:
            try:
                config = (mcp_tool._load_mcp_config() or {}).get(server_name)
            except Exception:
                config = None
            if isinstance(config, dict):
                dict.__setitem__(lazy, server_name, config)
