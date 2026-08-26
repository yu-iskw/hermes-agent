"""Regression tests for per-user OAuth persistence rollback."""

from __future__ import annotations


def _scope(user_id: str):
    from tools.mcp_oauth_identity import McpOAuthPrincipal, McpOAuthScope

    return McpOAuthScope(
        "per_user",
        McpOAuthPrincipal("slack", "T_ONE", user_id),
    )


def test_restore_never_writes_snapshot_into_legacy_shared_namespace(tmp_path):
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_scoped_storage import ScopedHermesTokenStorage

    alice = ScopedHermesTokenStorage("github", _scope("U_ALICE"), hermes_home=tmp_path)
    legacy = HermesTokenStorage("github", hermes_home=tmp_path)

    alice.restore(
        {
            "github.json": b'{"access_token":"ALICE"}',
            "github.client.json": b'{"client_id":"alice-client"}',
            "github.meta.json": b'{"token_endpoint":"https://idp.example/token"}',
        }
    )

    assert alice._tokens_path().read_bytes() == b'{"access_token":"ALICE"}'
    assert alice._client_info_path().exists()
    assert alice._meta_path().exists()
    assert not legacy._tokens_path().exists()
    assert not legacy._client_info_path().exists()
    assert not legacy._meta_path().exists()


def test_restore_basename_guard_cannot_escape_principal_directory(tmp_path):
    from tools.mcp_oauth_scoped_storage import ScopedHermesTokenStorage

    alice = ScopedHermesTokenStorage("github", _scope("U_ALICE"), hermes_home=tmp_path)
    alice.restore({"../../escape.json": b"{}"})

    assert (alice._scoped_token_dir() / "escape.json").exists()
    assert not (tmp_path / "escape.json").exists()
