"""Scope-aware persistent storage for MCP OAuth state.

The existing :class:`tools.mcp_oauth.HermesTokenStorage` remains the storage
implementation (permissions, atomic writes, expiry handling, snapshots).  This
subclass changes only path resolution so every piece of OAuth state follows the
same immutable :class:`McpOAuthScope` boundary.

Shared mode intentionally preserves the historical layout exactly.  Per-user
mode stores all state under an opaque digest directory; legacy shared tokens are
never migrated or read as a fallback because their human owner is unknowable.
"""

from __future__ import annotations

from pathlib import Path

from tools.mcp_oauth import HermesTokenStorage, _get_token_dir
from tools.mcp_oauth_identity import McpOAuthScope


class ScopedHermesTokenStorage(HermesTokenStorage):
    """Hermes OAuth storage whose complete state is bound to one OAuth scope."""

    def __init__(
        self,
        server_name: str,
        scope: McpOAuthScope,
        *,
        hermes_home: str | Path | None = None,
    ) -> None:
        super().__init__(server_name, hermes_home=hermes_home)
        self._oauth_scope = scope

    @property
    def oauth_scope(self) -> McpOAuthScope:
        return self._oauth_scope

    def _scoped_token_dir(self) -> Path:
        root = _get_token_dir(self._hermes_home)
        if not self._oauth_scope.is_per_user:
            return root
        # Opaque digest only: never expose Slack/Discord/etc identifiers in
        # directory names, logs, backups, or path traversal diagnostics.
        return root / "by-user" / self._oauth_scope.key

    def _tokens_path(self) -> Path:
        return self._scoped_token_dir() / f"{self._server_name}.json"

    def _client_info_path(self) -> Path:
        return self._scoped_token_dir() / f"{self._server_name}.client.json"

    def _meta_path(self) -> Path:
        return self._scoped_token_dir() / f"{self._server_name}.meta.json"

    def _cimd_rejected_path(self) -> Path:
        return self._scoped_token_dir() / f"{self._server_name}.cimd-off"
