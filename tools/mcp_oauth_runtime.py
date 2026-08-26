"""Scope-aware adapter for the MCP runtime's long-lived server registry.

``tools.mcp_tool`` historically indexes ``_servers`` by logical server name.
Changing only token files/provider caches would therefore leave a critical
cross-user path: User B could obtain User A's already-authenticated
``MCPServerTask``.  This module upgrades that registry in-place to exact
``(server, OAuthScope)`` lookup while preserving the existing dictionary API
used throughout the large MCP runtime.

The adapter is installed only when an OAuth server is prepared. Non-OAuth MCP
servers retain their historical shared connection key.  Per-user lookups have
NO "find any server with the same name" fallback: missing scope/connection
returns missing and the existing lazy-connect path can create the caller's own
connection.

This compatibility layer keeps #78174 narrowly scoped instead of spreading
identity plumbing through every MCP handler.  Scope selection remains outside
the model/tool argument surface and is driven by trusted ContextVars.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Any

from tools.mcp_oauth_identity import (
    McpOAuthScope,
    MissingMcpOAuthIdentityError,
    connection_registry_key,
    try_resolve_oauth_scope,
)

# A long-lived MCPServerTask keeps the context captured when it was created.
# Binding the already-resolved scope here also makes registry operations robust
# if a host's async scheduling implementation does not propagate gateway
# ContextVars exactly as CPython's run_coroutine_threadsafe currently does.
_TASK_SCOPES: ContextVar[dict[str, McpOAuthScope]] = ContextVar(
    "mcp_oauth_runtime_task_scopes",
    default={},
)

_INSTALL_LOCK = threading.Lock()


def _task_scope(server_name: str) -> McpOAuthScope | None:
    explicit = _TASK_SCOPES.get().get(server_name)
    if explicit is not None:
        return explicit
    return try_resolve_oauth_scope()


def bind_runtime_scope(server_name: str, scope: McpOAuthScope) -> None:
    """Bind a resolved scope to the current server task's ContextVar state."""
    current = dict(_TASK_SCOPES.get())
    current[server_name] = scope
    _TASK_SCOPES.set(current)


class ScopedMCPServerRegistry(dict):
    """Dict-compatible registry with exact per-user keys for OAuth servers."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        super().__init__(initial or {})
        self.oauth_servers: set[str] = set()

    def mark_oauth_server(self, server_name: str) -> None:
        self.oauth_servers.add(server_name)

    def _key_for_read(self, key: Any) -> Any:
        if not isinstance(key, str) or key not in self.oauth_servers:
            return key
        # Encoded/internal keys pass through. They are intentionally opaque.
        if "@@u-v1-" in key:
            return key
        scope = _task_scope(key)
        if scope is None:
            # Missing authenticated identity must never select a shared/other
            # connection. A guaranteed-missing key gives ordinary dict callers
            # fail-closed semantics without changing every call site.
            return f"{key}@@<missing-authenticated-identity>"
        return connection_registry_key(key, scope)

    def _key_for_write(self, key: Any) -> Any:
        if not isinstance(key, str) or key not in self.oauth_servers:
            return key
        if "@@u-v1-" in key:
            return key
        scope = _task_scope(key)
        if scope is None:
            raise MissingMcpOAuthIdentityError(
                f"refusing to register OAuth MCP server {key!r} without an "
                "authenticated per-user scope"
            )
        return connection_registry_key(key, scope)

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(self._key_for_read(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(self._key_for_write(key), value)

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._key_for_read(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return super().get(self._key_for_read(key), default)

    def pop(self, key: Any, default: Any = ...):
        resolved = self._key_for_read(key)
        if default is ...:
            return super().pop(resolved)
        return super().pop(resolved, default)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        return super().setdefault(self._key_for_write(key), default)


class PersistentPerUserLazyConfigs(dict):
    """Keep OAuth server config available for every user's first connection.

    ``mcp_tool._ensure_lazy_server_connected`` historically pops a lazy config
    after one successful connection because there was only one global server.
    Per-user OAuth has N independent long-lived connections, so that config is
    reusable metadata rather than one-shot state.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        super().__init__(initial or {})
        self.oauth_servers: set[str] = set()

    def mark_oauth_server(self, server_name: str) -> None:
        self.oauth_servers.add(server_name)

    def pop(self, key: Any, default: Any = ...):
        if isinstance(key, str) and key in self.oauth_servers and key in self:
            # Return without deleting so Alice's successful lazy connect does
            # not prevent Bob/Carol from creating their own scoped connection.
            return self[key]
        if default is ...:
            return super().pop(key)
        return super().pop(key, default)


def prepare_oauth_server_runtime(server_name: str) -> None:
    """Install/mark scope-aware MCP registries before resolving identity.

    This is intentionally called before ``resolve_oauth_scope``. On a headless
    gateway startup with no human bound, the provider build then fails closed
    but the runtime is already prepared for a later authenticated user's lazy
    connection.
    """
    from tools import mcp_tool

    with _INSTALL_LOCK:
        servers = mcp_tool._servers
        if not isinstance(servers, ScopedMCPServerRegistry):
            servers = ScopedMCPServerRegistry(dict(servers))
            mcp_tool._servers = servers
        servers.mark_oauth_server(server_name)

        lazy = mcp_tool._lazy_server_configs
        if not isinstance(lazy, PersistentPerUserLazyConfigs):
            lazy = PersistentPerUserLazyConfigs(dict(lazy))
            mcp_tool._lazy_server_configs = lazy
        lazy.mark_oauth_server(server_name)

        # Preserve the full connection config for later users.  This is safe
        # config metadata, not credential state; actual OAuth providers/tokens
        # are still created under the caller's immutable scope.
        if server_name not in lazy:
            try:
                config = (mcp_tool._load_mcp_config() or {}).get(server_name)
            except Exception:
                config = None
            if isinstance(config, dict):
                dict.__setitem__(lazy, server_name, config)
