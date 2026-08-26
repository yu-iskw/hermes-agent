"""Per-user MCP schema-cache isolation tests for shared gateways."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_session_context():
    from gateway.session_context import clear_session_vars

    clear_session_vars()
    yield
    clear_session_vars()


def _bind(user_id: str) -> None:
    from gateway.session_context import set_session_vars

    set_session_vars(platform="slack", scope_id="T_ONE", user_id=user_id)


def _config(mode: str = "per_user") -> dict:
    return {
        "mcp": {"oauth": {"identity_mode": mode}},
        "mcp_servers": {
            "github": {
                "url": "https://mcp.example/mcp",
                "auth": "oauth",
            },
            "filesystem": {"command": "server-filesystem"},
        },
    }


def test_per_user_oauth_schema_entries_are_not_shared(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import mcp_schema_cache

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _config())
    token = set_hermes_home_override(tmp_path)
    try:
        _bind("U_ALICE")
        mcp_schema_cache.write_cache_entry(
            "github",
            "fp",
            tools=[{"name": "alice-only"}],
            cache_scope="private",
        )

        _bind("U_BOB")
        assert mcp_schema_cache.get_cached_entry("github", "fp") is None
        mcp_schema_cache.write_cache_entry(
            "github",
            "fp",
            tools=[{"name": "bob-only"}],
            cache_scope="private",
        )

        _bind("U_ALICE")
        assert mcp_schema_cache.get_cached_entry("github", "fp")["tools"] == [
            {"name": "alice-only"}
        ]

        _bind("U_BOB")
        assert mcp_schema_cache.get_cached_entry("github", "fp")["tools"] == [
            {"name": "bob-only"}
        ]
    finally:
        reset_hermes_home_override(token)


def test_headless_per_user_startup_cannot_read_scoped_cache(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import mcp_schema_cache

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _config())
    token = set_hermes_home_override(tmp_path)
    try:
        _bind("U_ALICE")
        mcp_schema_cache.write_cache_entry(
            "github", "fp", tools=[{"name": "private-tool"}]
        )

        from gateway.session_context import clear_session_vars

        clear_session_vars()
        assert mcp_schema_cache.get_cached_entry("github", "fp") is None
    finally:
        reset_hermes_home_override(token)


def test_invalid_identity_mode_does_not_fall_back_to_shared_cache(monkeypatch):
    from tools.mcp_oauth_identity import InvalidMcpOAuthIdentityModeError
    from tools.mcp_schema_cache import get_cached_entry

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _config(mode="per-user"),
    )
    _bind("U_ALICE")

    with pytest.raises(InvalidMcpOAuthIdentityModeError):
        get_cached_entry("github", "fp")


def test_non_oauth_schema_cache_remains_shared(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import mcp_schema_cache

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _config())
    token = set_hermes_home_override(tmp_path)
    try:
        _bind("U_ALICE")
        mcp_schema_cache.write_cache_entry(
            "filesystem", "fp", tools=[{"name": "read_file"}]
        )
        _bind("U_BOB")
        assert mcp_schema_cache.get_cached_entry("filesystem", "fp")["tools"] == [
            {"name": "read_file"}
        ]
    finally:
        reset_hermes_home_override(token)
