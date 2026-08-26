#!/usr/bin/env python3
"""Central manager for MCP OAuth state.

The security key for every mutable OAuth object is now
``(Hermes home, MCP server, OAuth scope)``.  In ``shared`` mode the scope key
is the backwards-compatible constant ``shared``.  In ``per_user`` mode it is
an opaque digest of the authenticated gateway principal, so providers,
refreshes, locks, disk watches, and 401 deduplication can never silently cross
human-user boundaries (#78174).

The MCP SDK provider implementation lives in :mod:`tools.mcp_oauth_provider`;
this module remains the single process-wide manager and re-exports
``_HERMES_PROVIDER_CLS`` for existing tests/callers.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tools.mcp_oauth_identity import McpOAuthScope, resolve_oauth_scope
from tools.mcp_oauth_provider import _HERMES_PROVIDER_CLS
from tools.mcp_oauth_scoped_storage import ScopedHermesTokenStorage

logger = logging.getLogger(__name__)


@dataclass
class _ProviderEntry:
    """OAuth state isolated to one server and immutable credential scope."""

    server_url: str
    oauth_config: Optional[dict]
    oauth_scope: McpOAuthScope = field(default_factory=McpOAuthScope.shared)
    provider: Optional[Any] = None
    last_mtime_ns: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_401: dict[str, "asyncio.Future[bool]"] = field(default_factory=dict)


class MCPOAuthManager:
    """Single source of truth for scope-isolated MCP OAuth state.

    ``oauth_scope`` parameters are optional only for API compatibility.  When
    omitted they are resolved from the trusted runtime context.  Therefore an
    omitted scope in ``per_user`` mode fails closed when no authenticated
    principal is bound; it never falls back to shared credentials.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], _ProviderEntry] = {}
        self._entries_lock = threading.Lock()
        self._inflight_tasks: set[asyncio.Task] = set()

    @staticmethod
    def _resolve_scope(oauth_scope: McpOAuthScope | None) -> McpOAuthScope:
        return oauth_scope if oauth_scope is not None else resolve_oauth_scope()

    @classmethod
    def _key(
        cls,
        server_name: str,
        hermes_home: str | Path | None = None,
        oauth_scope: McpOAuthScope | None = None,
    ) -> tuple[str, str, str]:
        from hermes_constants import get_hermes_home

        home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
        scope = cls._resolve_scope(oauth_scope)
        return (
            str(home.expanduser().resolve(strict=False)),
            server_name,
            scope.key,
        )

    def get_or_build_provider(
        self,
        server_name: str,
        server_url: str,
        oauth_config: Optional[dict],
        *,
        oauth_scope: McpOAuthScope | None = None,
    ) -> Optional[Any]:
        """Return/build a provider for exactly one OAuth identity scope."""
        scope = self._resolve_scope(oauth_scope)
        key = self._key(server_name, oauth_scope=scope)
        with self._entries_lock:
            entry = self._entries.get(key)
            if entry is not None and entry.server_url != server_url:
                logger.info(
                    "MCP OAuth '%s': URL changed; discarding provider for scope %s",
                    server_name,
                    scope.key,
                )
                entry = None

            if entry is None:
                entry = _ProviderEntry(
                    server_url=server_url,
                    oauth_config=oauth_config,
                    oauth_scope=scope,
                )
                self._entries[key] = entry

            if entry.provider is None:
                entry.provider = self._build_provider(server_name, entry)
                if entry.provider is not None:
                    entry.provider._hermes_home = key[0]

            return entry.provider

    def _build_provider(
        self,
        server_name: str,
        entry: _ProviderEntry,
    ) -> Optional[Any]:
        if _HERMES_PROVIDER_CLS is None:
            logger.warning("MCP OAuth '%s': SDK auth module unavailable", server_name)
            return None

        from tools.mcp_oauth import (
            OAuthNonInteractiveError,
            _OAUTH_AVAILABLE,
            _build_client_metadata,
            _configure_callback_port,
            _is_interactive,
            _make_callback_waiter,
            _make_redirect_handler,
            _maybe_preregister_client,
            apply_oauth_provider_defaults,
            cimd_provider_kwargs,
            token_request_user_agent,
        )

        if not _OAUTH_AVAILABLE:
            return None

        cfg = dict(entry.oauth_config or {})
        apply_oauth_provider_defaults(
            cfg,
            server_name=server_name,
            server_url=entry.server_url,
        )

        # The complete OAuth persistence surface follows the same scope:
        # tokens, DCR client info, AS metadata, and CIMD refusal marker.
        storage = ScopedHermesTokenStorage(server_name, entry.oauth_scope)

        from tools.mcp_dashboard_oauth import get_dashboard_oauth_flow

        if (
            get_dashboard_oauth_flow() is None
            and not _is_interactive()
            and not storage.has_cached_tokens()
        ):
            raise OAuthNonInteractiveError(
                "MCP OAuth for "
                f"'{server_name}': non-interactive environment and no cached "
                "tokens exist for the current OAuth identity. Complete "
                "authorization from an interactive surface first."
            )

        _configure_callback_port(cfg, storage)
        client_metadata = _build_client_metadata(cfg)
        _maybe_preregister_client(storage, cfg, client_metadata)

        resolved_port = cfg.get("_resolved_port", 0)
        redirect_handler = _make_redirect_handler(resolved_port)
        callback_handler = _make_callback_waiter(
            resolved_port,
            cfg.get("_cimd_url"),
            timeout=float(cfg.get("timeout", 300)),
        )

        return _HERMES_PROVIDER_CLS(
            server_name=server_name,
            oauth_scope=entry.oauth_scope,
            preregistered=bool(cfg.get("client_id")),
            server_url=entry.server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            token_user_agent=token_request_user_agent(cfg),
            **cimd_provider_kwargs(cfg),
        )

    def remove(
        self,
        server_name: str,
        *,
        hermes_home: str | Path | None = None,
        oauth_scope: McpOAuthScope | None = None,
    ) -> _ProviderEntry | None:
        """Evict and delete OAuth state for exactly one scope."""
        scope = self._resolve_scope(oauth_scope)
        with self._entries_lock:
            entry = self._entries.pop(
                self._key(server_name, hermes_home, scope),
                None,
            )

        ScopedHermesTokenStorage(
            server_name,
            scope,
            hermes_home=hermes_home,
        ).remove()
        logger.info(
            "MCP OAuth '%s': evicted provider and persisted state for scope %s",
            server_name,
            scope.key,
        )
        return entry

    def restore_entry(
        self,
        server_name: str,
        entry: _ProviderEntry | None,
        *,
        hermes_home: str | Path | None = None,
        oauth_scope: McpOAuthScope | None = None,
    ) -> None:
        """Restore a removed entry without overwriting a newer scoped entry."""
        if entry is None:
            return
        scope = oauth_scope if oauth_scope is not None else entry.oauth_scope
        with self._entries_lock:
            self._entries.setdefault(
                self._key(server_name, hermes_home, scope),
                entry,
            )

    def evict(
        self,
        server_name: str,
        *,
        hermes_home: str | Path | None = None,
        oauth_scope: McpOAuthScope | None = None,
    ) -> None:
        """Drop only the in-process provider for exactly one scope."""
        scope = self._resolve_scope(oauth_scope)
        with self._entries_lock:
            self._entries.pop(self._key(server_name, hermes_home, scope), None)

    async def invalidate_if_disk_changed(
        self,
        server_name: str,
        *,
        hermes_home: str | Path | None = None,
        oauth_scope: McpOAuthScope | None = None,
    ) -> bool:
        """Reload provider state only when THIS scope's token file changed."""
        scope = self._resolve_scope(oauth_scope)
        entry = self._entries.get(self._key(server_name, hermes_home, scope))
        if entry is None or entry.provider is None:
            return False

        async with entry.lock:
            tokens_path = ScopedHermesTokenStorage(
                server_name,
                scope,
                hermes_home=hermes_home,
            )._tokens_path()
            try:
                mtime_ns = tokens_path.stat().st_mtime_ns
            except (FileNotFoundError, OSError):
                return False

            if mtime_ns == entry.last_mtime_ns:
                return False

            old = entry.last_mtime_ns
            entry.last_mtime_ns = mtime_ns
            if hasattr(entry.provider, "_initialized"):
                entry.provider._initialized = False  # noqa: SLF001
            logger.info(
                "MCP OAuth '%s': scoped token file changed "
                "(mtime %d -> %d), forcing reload",
                server_name,
                old,
                mtime_ns,
            )
            return True

    async def handle_401(
        self,
        server_name: str,
        failed_access_token: Optional[str] = None,
        *,
        oauth_scope: McpOAuthScope | None = None,
    ) -> bool:
        """Recover a 401 without sharing dedup/refresh state across users."""
        scope = self._resolve_scope(oauth_scope)
        entry = self._entries.get(self._key(server_name, oauth_scope=scope))
        if entry is None or entry.provider is None:
            return False

        failed_key = failed_access_token or "<unknown>"
        loop = asyncio.get_running_loop()

        async with entry.lock:
            pending = entry.pending_401.get(failed_key)
            if pending is None:
                pending = loop.create_future()
                entry.pending_401[failed_key] = pending

                async def _do_handle() -> None:
                    try:
                        disk_changed = await self.invalidate_if_disk_changed(
                            server_name,
                            oauth_scope=scope,
                        )
                        if disk_changed:
                            if not pending.done():
                                pending.set_result(True)
                            return

                        provider = entry.provider
                        ctx = getattr(provider, "context", None)
                        can_refresh = False
                        if ctx is not None:
                            can_refresh_fn = getattr(ctx, "can_refresh_token", None)
                            if callable(can_refresh_fn):
                                try:
                                    can_refresh = bool(can_refresh_fn())
                                except Exception:
                                    can_refresh = False
                        if not pending.done():
                            pending.set_result(can_refresh)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning(
                            "MCP OAuth '%s': scoped 401 handler failed: %s",
                            server_name,
                            exc,
                        )
                        if not pending.done():
                            pending.set_result(False)
                    finally:
                        entry.pending_401.pop(failed_key, None)

                task = asyncio.create_task(_do_handle())
                self._inflight_tasks.add(task)
                task.add_done_callback(self._inflight_tasks.discard)

        try:
            return await pending
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "MCP OAuth '%s': awaiting scoped 401 handler failed: %s",
                server_name,
                exc,
            )
            return False


_MANAGER: Optional[MCPOAuthManager] = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> MCPOAuthManager:
    """Return the process-wide scoped MCP OAuth manager singleton."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = MCPOAuthManager()
        return _MANAGER


def reset_manager_for_tests() -> None:
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = None
