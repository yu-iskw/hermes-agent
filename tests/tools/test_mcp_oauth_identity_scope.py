"""Security-boundary tests for per-user MCP OAuth identity (#78174)."""

from __future__ import annotations

import re

import pytest


@pytest.fixture(autouse=True)
def _clear_session_context():
    from gateway.session_context import clear_session_vars

    clear_session_vars()
    yield
    clear_session_vars()


def _patch_config(monkeypatch, mode_marker=...):
    if mode_marker is ...:
        config = {}
    else:
        config = {"mcp": {"oauth": {"identity_mode": mode_marker}}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)


def _bind(*, platform="slack", scope_id="T_WORKSPACE", user_id="U_USER"):
    from gateway.session_context import set_session_vars

    set_session_vars(
        platform=platform,
        scope_id=scope_id,
        user_id=user_id,
    )


def test_absent_config_preserves_shared_mode(monkeypatch):
    from tools.mcp_oauth_identity import McpOAuthScope, resolve_oauth_scope

    _patch_config(monkeypatch)

    assert resolve_oauth_scope() == McpOAuthScope.shared()


def test_explicit_shared_mode_preserves_legacy_scope(monkeypatch):
    from tools.mcp_oauth_identity import McpOAuthScope, resolve_oauth_scope

    _patch_config(monkeypatch, "shared")
    _bind()

    assert resolve_oauth_scope() == McpOAuthScope.shared()


def test_invalid_explicit_mode_fails_closed(monkeypatch):
    from tools.mcp_oauth_identity import (
        InvalidMcpOAuthIdentityModeError,
        resolve_oauth_scope,
    )

    _patch_config(monkeypatch, "per-user")
    _bind()

    with pytest.raises(InvalidMcpOAuthIdentityModeError):
        resolve_oauth_scope()


def test_per_user_mode_requires_task_local_authenticated_identity(monkeypatch):
    from tools.mcp_oauth_identity import (
        MissingMcpOAuthIdentityError,
        resolve_oauth_scope,
    )

    _patch_config(monkeypatch, "per_user")

    with pytest.raises(MissingMcpOAuthIdentityError, match="will not fall back"):
        resolve_oauth_scope()


def test_process_environment_is_not_a_trusted_principal(monkeypatch):
    """Legacy process-global session mirrors cannot authorize per-user MCP."""
    from tools.mcp_oauth_identity import MissingMcpOAuthIdentityError, resolve_oauth_scope

    _patch_config(monkeypatch, "per_user")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "slack")
    monkeypatch.setenv("HERMES_SESSION_SCOPE_ID", "T_STALE")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "U_OTHER_USER")

    with pytest.raises(MissingMcpOAuthIdentityError):
        resolve_oauth_scope()


def test_per_user_scope_uses_platform_scope_and_user(monkeypatch):
    from tools.mcp_oauth_identity import resolve_oauth_scope

    _patch_config(monkeypatch, "per_user")
    _bind(platform="slack", scope_id="T_ONE", user_id="U_123")
    one = resolve_oauth_scope()

    # Same platform-local user identifier in another workspace is a distinct
    # authorization principal.
    _bind(platform="slack", scope_id="T_TWO", user_id="U_123")
    two = resolve_oauth_scope()

    assert one != two
    assert one.key != two.key


def test_scope_key_is_opaque_and_filesystem_safe(monkeypatch):
    from tools.mcp_oauth_identity import resolve_oauth_scope

    _patch_config(monkeypatch, "per_user")
    _bind(
        platform="slack",
        scope_id="T-sensitive-workspace",
        user_id="U-sensitive-person",
    )
    scope = resolve_oauth_scope()

    assert scope.key.startswith("u-v1-")
    assert re.fullmatch(r"u-v1-[0-9a-f]{64}", scope.key)
    assert "slack" not in scope.key
    assert "sensitive" not in scope.key


def test_connection_registry_keys_never_collide_between_users(monkeypatch):
    from tools.mcp_oauth_identity import connection_registry_key, resolve_oauth_scope

    _patch_config(monkeypatch, "per_user")
    _bind(user_id="U_ALICE")
    alice = resolve_oauth_scope()
    _bind(user_id="U_BOB")
    bob = resolve_oauth_scope()

    assert connection_registry_key("github", alice) != connection_registry_key(
        "github", bob
    )
    assert connection_registry_key("github", alice).startswith("github@@u-v1-")


def test_shared_connection_registry_key_is_backwards_compatible():
    from tools.mcp_oauth_identity import McpOAuthScope, connection_registry_key

    assert connection_registry_key("github", McpOAuthScope.shared()) == "github"


def test_explicit_scope_is_typed_not_model_argument(monkeypatch):
    from tools.mcp_oauth_identity import (
        McpOAuthPrincipal,
        McpOAuthScope,
        explicit_oauth_scope,
        resolve_oauth_scope,
    )

    _patch_config(monkeypatch, "per_user")
    admin_scope = McpOAuthScope(
        "per_user",
        McpOAuthPrincipal("slack", "T_ADMIN", "U_ADMIN"),
    )

    with explicit_oauth_scope(admin_scope):
        assert resolve_oauth_scope() == admin_scope


def test_scoped_storage_separates_all_oauth_state(monkeypatch, tmp_path):
    from tools.mcp_oauth_identity import McpOAuthPrincipal, McpOAuthScope
    from tools.mcp_oauth_scoped_storage import ScopedHermesTokenStorage

    alice_scope = McpOAuthScope(
        "per_user", McpOAuthPrincipal("slack", "T_ONE", "U_ALICE")
    )
    bob_scope = McpOAuthScope(
        "per_user", McpOAuthPrincipal("slack", "T_ONE", "U_BOB")
    )
    alice = ScopedHermesTokenStorage("github", alice_scope, hermes_home=tmp_path)
    bob = ScopedHermesTokenStorage("github", bob_scope, hermes_home=tmp_path)

    alice_paths = {
        alice._tokens_path(),
        alice._client_info_path(),
        alice._meta_path(),
        alice._cimd_rejected_path(),
    }
    bob_paths = {
        bob._tokens_path(),
        bob._client_info_path(),
        bob._meta_path(),
        bob._cimd_rejected_path(),
    }

    assert alice_paths.isdisjoint(bob_paths)
    assert all("by-user" in path.parts for path in alice_paths | bob_paths)
    assert all("U_ALICE" not in str(path) for path in alice_paths)
    assert all("U_BOB" not in str(path) for path in bob_paths)


def test_scoped_storage_shared_mode_keeps_historical_paths(tmp_path):
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_identity import McpOAuthScope
    from tools.mcp_oauth_scoped_storage import ScopedHermesTokenStorage

    legacy = HermesTokenStorage("github", hermes_home=tmp_path)
    scoped = ScopedHermesTokenStorage(
        "github", McpOAuthScope.shared(), hermes_home=tmp_path
    )

    assert scoped._tokens_path() == legacy._tokens_path()
    assert scoped._client_info_path() == legacy._client_info_path()
    assert scoped._meta_path() == legacy._meta_path()
    assert scoped._cimd_rejected_path() == legacy._cimd_rejected_path()


def test_per_user_storage_never_falls_back_to_legacy_shared_file(tmp_path):
    """Unknown ownership means shared tokens cannot be auto-migrated."""
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_identity import McpOAuthPrincipal, McpOAuthScope
    from tools.mcp_oauth_scoped_storage import ScopedHermesTokenStorage

    legacy = HermesTokenStorage("github", hermes_home=tmp_path)
    legacy._tokens_path().parent.mkdir(parents=True, exist_ok=True)
    legacy._tokens_path().write_text('{"access_token":"SHARED"}', encoding="utf-8")

    alice = ScopedHermesTokenStorage(
        "github",
        McpOAuthScope(
            "per_user", McpOAuthPrincipal("slack", "T_ONE", "U_ALICE")
        ),
        hermes_home=tmp_path,
    )

    assert not alice._tokens_path().exists()
    assert alice._tokens_path() != legacy._tokens_path()
