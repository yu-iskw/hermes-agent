"""Scope-aware persistent storage for MCP OAuth state.

The existing :class:`tools.mcp_oauth.HermesTokenStorage` remains the storage
implementation (permissions, atomic writes, expiry handling, snapshots). This
subclass changes path resolution so every piece of OAuth state follows the same
immutable :class:`McpOAuthScope` boundary.

Shared mode intentionally preserves the historical layout exactly. Per-user
mode stores all state under an opaque digest directory; legacy shared tokens are
never migrated or read as a fallback because their human owner is unknowable.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

from hermes_constants import secure_parent_dir
from tools.mcp_oauth import HermesTokenStorage, _get_token_dir
from tools.mcp_oauth_identity import McpOAuthScope

logger = logging.getLogger(__name__)


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
        return root / "by-user" / self._oauth_scope.key

    def _tokens_path(self) -> Path:
        return self._scoped_token_dir() / f"{self._server_name}.json"

    def _client_info_path(self) -> Path:
        return self._scoped_token_dir() / f"{self._server_name}.client.json"

    def _meta_path(self) -> Path:
        return self._scoped_token_dir() / f"{self._server_name}.meta.json"

    def _cimd_rejected_path(self) -> Path:
        return self._scoped_token_dir() / f"{self._server_name}.cimd-off"

    def restore(self, snapshot: dict[str, bytes], *, only_if_absent: bool = False) -> None:
        """Restore a reauth snapshot into THIS scope, never the shared root.

        ``HermesTokenStorage.restore`` historically reconstructs its destination
        directly from ``_get_token_dir`` because there was only one storage
        namespace. Calling that implementation from a scoped subclass would
        therefore copy Alice's rollback data into the legacy shared namespace.
        Keep the same concurrency and file-permission semantics while resolving
        the destination through ``_scoped_token_dir``.
        """
        if only_if_absent and any(
            path.exists()
            for path in (self._tokens_path(), self._client_info_path(), self._meta_path())
        ):
            logger.info(
                "Skipping OAuth rollback for %s because newer scoped state exists",
                self._server_name,
            )
            return

        self.remove()
        if not snapshot:
            return

        token_dir = self._scoped_token_dir()
        token_dir.mkdir(parents=True, exist_ok=True)
        secure_parent_dir(token_dir / ".scope-permissions")
        for fname, data in snapshot.items():
            # Snapshot names are produced only by ``snapshot()`` from the
            # storage's own known paths. Still collapse to basename so a
            # corrupted/injected snapshot cannot escape this principal dir.
            safe_name = Path(fname).name
            path = token_dir / safe_name
            try:
                fd = os.open(
                    str(path),
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                    stat.S_IRUSR | stat.S_IWUSR,
                )
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
            except OSError as exc:
                logger.warning("Failed to restore scoped OAuth state %s: %s", safe_name, exc)
