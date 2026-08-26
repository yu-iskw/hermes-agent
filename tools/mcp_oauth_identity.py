"""Trusted identity boundary for MCP OAuth authorization.

Issue #78174 requires OAuth credentials to follow the authenticated human who
initiated an MCP request.  This module deliberately keeps that identity out of
model-visible tool arguments: the only automatic source in ``per_user`` mode is
the gateway's task-local session binding.

Configuration::

    mcp:
      oauth:
        identity_mode: shared   # default, backwards compatible
        # identity_mode: per_user

``shared`` preserves the historic one-token-set-per-profile/server behaviour.
``per_user`` derives an immutable principal from the bound platform, platform
scope (Slack workspace / Discord guild / Matrix server), and user id.  Missing
identity fails closed.  Explicit but invalid configuration also fails closed;
a typo must never silently downgrade a multi-user deployment to shared creds.

Raw human identifiers are never used as filesystem/cache keys.  The stable
``storage_key`` is an opaque SHA-256 digest of a versioned canonical encoding.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Literal

IdentityMode = Literal["shared", "per_user"]

_SHARED = "shared"
_PER_USER = "per_user"
_VALID_MODES = frozenset({_SHARED, _PER_USER})
_SCOPE_VERSION = "mcp-oauth-principal-v1"


class McpOAuthIdentityError(RuntimeError):
    """Base class for MCP OAuth identity-boundary failures."""


class InvalidMcpOAuthIdentityModeError(McpOAuthIdentityError, ValueError):
    """Raised for an explicitly configured unknown identity mode."""


class MissingMcpOAuthIdentityError(McpOAuthIdentityError):
    """Raised when ``per_user`` mode has no authenticated bound principal."""


@dataclass(frozen=True, slots=True)
class McpOAuthPrincipal:
    """Authenticated human principal attached to one gateway request.

    ``scope_id`` is the platform-neutral account/workspace discriminator that
    Hermes already captures in ``gateway.session_context``.  It may be empty on
    platforms that do not expose one; platform + user_id still forms a stable
    principal there.
    """

    platform: str
    scope_id: str
    user_id: str

    def __post_init__(self) -> None:
        platform = self.platform.strip().lower()
        scope_id = self.scope_id.strip()
        user_id = self.user_id.strip()
        if not platform or not user_id:
            raise MissingMcpOAuthIdentityError(
                "MCP OAuth per-user identity requires a bound session platform "
                "and authenticated user id. Hermes will not reuse another "
                "user's OAuth credentials."
            )
        object.__setattr__(self, "platform", platform)
        object.__setattr__(self, "scope_id", scope_id)
        object.__setattr__(self, "user_id", user_id)

    @property
    def storage_key(self) -> str:
        """Opaque, collision-resistant key that does not expose raw identity."""
        canonical = json.dumps(
            [_SCOPE_VERSION, self.platform, self.scope_id, self.user_id],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return "u-v1-" + hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class McpOAuthScope:
    """Immutable credential/connection scope for an MCP OAuth operation."""

    mode: IdentityMode
    principal: McpOAuthPrincipal | None = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise InvalidMcpOAuthIdentityModeError(
                f"invalid MCP OAuth identity mode: {self.mode!r}"
            )
        if self.mode == _PER_USER and self.principal is None:
            raise MissingMcpOAuthIdentityError(
                "MCP OAuth per-user mode requires an authenticated principal"
            )
        if self.mode == _SHARED and self.principal is not None:
            raise McpOAuthIdentityError(
                "shared MCP OAuth scope must not carry a user principal"
            )

    @classmethod
    def shared(cls) -> "McpOAuthScope":
        return cls(_SHARED)

    @property
    def key(self) -> str:
        """Stable internal key used by provider/connection/cache registries."""
        if self.mode == _SHARED:
            return _SHARED
        assert self.principal is not None
        return self.principal.storage_key

    @property
    def is_per_user(self) -> bool:
        return self.mode == _PER_USER


# Explicit trusted override for administrative CLI flows and tests.  This is a
# typed scope, not a string selector accepted from the model/MCP argument map.
_EXPLICIT_SCOPE: ContextVar[McpOAuthScope | None] = ContextVar(
    "mcp_oauth_explicit_scope", default=None
)


@contextmanager
def explicit_oauth_scope(scope: McpOAuthScope) -> Iterator[None]:
    """Temporarily bind a trusted, already-validated OAuth scope."""
    token = _EXPLICIT_SCOPE.set(scope)
    try:
        yield
    finally:
        _EXPLICIT_SCOPE.reset(token)


def get_oauth_identity_mode() -> IdentityMode:
    """Return the configured mode; reject explicit unknown values.

    Absence means ``shared`` for backwards compatibility.  An explicit typo is
    a configuration error rather than a security downgrade.
    """
    from hermes_cli.config import load_config

    config = load_config() or {}
    mcp = config.get("mcp")
    if mcp is None:
        return _SHARED
    if not isinstance(mcp, dict):
        raise InvalidMcpOAuthIdentityModeError("config 'mcp' must be a mapping")
    oauth = mcp.get("oauth")
    if oauth is None:
        return _SHARED
    if not isinstance(oauth, dict):
        raise InvalidMcpOAuthIdentityModeError("config 'mcp.oauth' must be a mapping")
    raw = oauth.get("identity_mode")
    if raw is None:
        return _SHARED
    mode = str(raw).strip().lower()
    if mode not in _VALID_MODES:
        raise InvalidMcpOAuthIdentityModeError(
            "mcp.oauth.identity_mode must be 'shared' or 'per_user' "
            f"(got {raw!r})"
        )
    return mode  # type: ignore[return-value]


def _bound_session_value(var_name: str) -> str:
    """Read a ContextVar only when THIS task explicitly bound it.

    Security-sensitive identity resolution must not use ``get_session_env``'s
    process-global ``os.environ`` fallback: a long-lived shared gateway may
    retain legacy env mirrors from another request.  We intentionally inspect
    the gateway's task-local binding and reject ``_UNSET`` / empty values.
    """
    from gateway import session_context

    var = getattr(session_context, var_name)
    unset = getattr(session_context, "_UNSET")
    value = var.get()
    if value is unset:
        return ""
    return str(value or "").strip()


def current_bound_principal() -> McpOAuthPrincipal | None:
    """Return the authenticated principal bound to the current task, if any."""
    platform = _bound_session_value("_SESSION_PLATFORM")
    user_id = _bound_session_value("_SESSION_USER_ID")
    scope_id = _bound_session_value("_SESSION_SCOPE_ID")
    if not platform or not user_id:
        return None
    return McpOAuthPrincipal(
        platform=platform,
        scope_id=scope_id,
        user_id=user_id,
    )


def resolve_oauth_scope(*, require_identity: bool = True) -> McpOAuthScope:
    """Resolve the credential scope for the current trusted runtime context."""
    explicit = _EXPLICIT_SCOPE.get()
    if explicit is not None:
        return explicit

    mode = get_oauth_identity_mode()
    if mode == _SHARED:
        return McpOAuthScope.shared()

    principal = current_bound_principal()
    if principal is None:
        if require_identity:
            raise MissingMcpOAuthIdentityError(
                "mcp.oauth.identity_mode is 'per_user' but this request has no "
                "authenticated task-local user identity. Hermes will not fall "
                "back to shared or another user's MCP OAuth credentials."
            )
        # Callers that only need to determine whether a scope is currently
        # available can use ``try_resolve_oauth_scope`` below.  Returning a
        # fake/anonymous per-user scope here would create a shared backdoor, so
        # even the non-requiring form never manufactures one.
        raise MissingMcpOAuthIdentityError(
            "no authenticated MCP OAuth principal is bound to this task"
        )
    return McpOAuthScope(_PER_USER, principal)


def try_resolve_oauth_scope() -> McpOAuthScope | None:
    """Best-effort scope lookup without manufacturing an anonymous identity."""
    try:
        return resolve_oauth_scope()
    except MissingMcpOAuthIdentityError:
        return None


def connection_registry_key(server_name: str, scope: McpOAuthScope) -> str:
    """Opaque exact key for a long-lived authenticated MCP connection."""
    if not scope.is_per_user:
        return server_name
    return f"{server_name}@@{scope.key}"
