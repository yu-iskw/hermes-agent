"""Adversarial tests for scoped long-lived MCP server lookup (#78174)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_session_context():
    from gateway.session_context import clear_session_vars

    clear_session_vars()
    yield
    clear_session_vars()


def _patch_per_user(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"mcp": {"oauth": {"identity_mode": "per_user"}}},
    )


def _bind(user_id: str, *, scope_id: str = "T_ONE") -> None:
    from gateway.session_context import set_session_vars

    set_session_vars(
        platform="slack",
        scope_id=scope_id,
        user_id=user_id,
    )


def test_registry_never_falls_back_to_another_users_live_server(monkeypatch):
    from tools.mcp_oauth_identity import resolve_oauth_scope
    from tools.mcp_oauth_runtime import ScopedMCPServerRegistry, bind_runtime_scope

    _patch_per_user(monkeypatch)
    registry = ScopedMCPServerRegistry()
    registry.mark_oauth_server("github")

    alice_server = object()
    bob_server = object()

    _bind("U_ALICE")
    alice_scope = resolve_oauth_scope()
    bind_runtime_scope("github", alice_scope)
    registry["github"] = alice_server

    # A fresh task/request binding must not inherit the previous explicit
    # runtime map; bind Bob explicitly as a server task would after provider
    # construction.
    _bind("U_BOB")
    bob_scope = resolve_oauth_scope()
    bind_runtime_scope("github", bob_scope)

    assert registry.get("github") is None
    registry["github"] = bob_server
    assert registry.get("github") is bob_server

    _bind("U_ALICE")
    bind_runtime_scope("github", alice_scope)
    assert registry.get("github") is alice_server


def test_registry_missing_identity_is_not_shared_fallback(monkeypatch):
    from tools.mcp_oauth_runtime import ScopedMCPServerRegistry

    _patch_per_user(monkeypatch)
    registry = ScopedMCPServerRegistry()
    registry.mark_oauth_server("github")

    # Seed an encoded Alice entry without binding any request principal.
    dict.__setitem__(registry, "github@@u-v1-deadbeef", object())

    assert registry.get("github") is None
    assert "github" not in registry


def test_non_oauth_servers_keep_historical_shared_registry_semantics(monkeypatch):
    from tools.mcp_oauth_runtime import ScopedMCPServerRegistry

    _patch_per_user(monkeypatch)
    registry = ScopedMCPServerRegistry()
    filesystem = object()

    registry["filesystem"] = filesystem
    _bind("U_ALICE")
    assert registry.get("filesystem") is filesystem
    _bind("U_BOB")
    assert registry.get("filesystem") is filesystem


def test_lazy_oauth_config_survives_multiple_users():
    from tools.mcp_oauth_runtime import PersistentPerUserLazyConfigs

    config = {"url": "https://mcp.example/mcp", "auth": "oauth"}
    lazy = PersistentPerUserLazyConfigs({"github": config})
    lazy.mark_oauth_server("github")

    assert lazy.pop("github") == config
    assert "github" in lazy
    assert lazy.pop("github") == config
    assert "github" in lazy


def test_provider_manager_cache_isolated_by_principal(monkeypatch):
    """Provider entries for Alice and Bob cannot share locks/401 state."""
    from tools.mcp_oauth_identity import resolve_oauth_scope
    from tools.mcp_oauth_manager import MCPOAuthManager, _ProviderEntry

    _patch_per_user(monkeypatch)
    manager = MCPOAuthManager()

    _bind("U_ALICE")
    alice = resolve_oauth_scope()
    _bind("U_BOB")
    bob = resolve_oauth_scope()

    manager._entries[manager._key("github", oauth_scope=alice)] = _ProviderEntry(
        server_url="https://mcp.example/mcp",
        oauth_config=None,
        oauth_scope=alice,
        provider=object(),
    )
    manager._entries[manager._key("github", oauth_scope=bob)] = _ProviderEntry(
        server_url="https://mcp.example/mcp",
        oauth_config=None,
        oauth_scope=bob,
        provider=object(),
    )

    alice_entry = manager._entries[manager._key("github", oauth_scope=alice)]
    bob_entry = manager._entries[manager._key("github", oauth_scope=bob)]

    assert alice_entry is not bob_entry
    assert alice_entry.provider is not bob_entry.provider
    assert alice_entry.lock is not bob_entry.lock
    assert alice_entry.pending_401 is not bob_entry.pending_401
