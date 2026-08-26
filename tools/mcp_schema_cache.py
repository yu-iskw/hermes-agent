"""Persistent MCP tool-schema cache for lazy server startup.

Stores tool manifests on disk so Hermes can register MCP tools without eagerly
spawning a server. Historically this file assumed the cache was "per-user local
disk". That assumption is false for a shared Hermes gateway: many authenticated
humans use the same HERMES_HOME. In ``mcp.oauth.identity_mode: per_user``, OAuth
server cache entries therefore use the same opaque requesting-user scope as the
credential/connection boundary (#78174).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "mcp_schema_cache.json"
_cache_lock = threading.Lock()


def _cache_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / _CACHE_FILENAME


def config_fingerprint(config: dict) -> str:
    """Stable hash of the connection-defining parts of an MCP server config."""
    tools_filter = config.get("tools") or {}
    payload = {
        "command": config.get("command"),
        "args": config.get("args") or [],
        "url": config.get("url"),
        "transport": config.get("transport"),
        "tools_include": sorted(tools_filter.get("include") or []),
        "tools_exclude": sorted(tools_filter.get("exclude") or []),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_all() -> Dict[str, Any]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("Could not read MCP schema cache %s: %s", path, exc)
        return {}


def _save_all(data: Dict[str, Any]) -> None:
    from utils import atomic_json_write

    atomic_json_write(_cache_path(), data, mode=0o600)


def _is_per_user_oauth_server(server_name: str) -> bool:
    """Return True only for OAuth servers under explicit per-user mode."""
    try:
        from tools.mcp_oauth_identity import get_oauth_identity_mode

        if get_oauth_identity_mode() != "per_user":
            return False
        from hermes_cli.config import load_config

        servers = (load_config() or {}).get("mcp_servers") or {}
        config = servers.get(server_name) if isinstance(servers, dict) else None
        return isinstance(config, dict) and str(config.get("auth") or "").strip().lower() == "oauth"
    except Exception:
        # Invalid identity configuration is not a reason to read a less-scoped
        # cache. The caller will surface the configuration error elsewhere.
        return False


def _scoped_cache_key(server_name: str) -> str | None:
    """Return the on-disk key, or None when per-user identity is unavailable."""
    if not _is_per_user_oauth_server(server_name):
        return server_name

    from tools.mcp_oauth_identity import try_resolve_oauth_scope

    scope = try_resolve_oauth_scope()
    if scope is None:
        # Headless startup has no authenticated human. Serving another user's
        # private schema would leak capability metadata and could register tools
        # that the current principal is not entitled to see.
        return None
    return f"{server_name}@@{scope.key}"


def get_cached_entry(server_name: str, fingerprint: str) -> Optional[dict]:
    """Return a valid entry for the exact current identity scope.

    MCP 2026-07-28 (SEP-2549) ``ttlMs`` freshness hints are preserved. In
    per-user OAuth mode we scope all entries, not only ones explicitly marked
    ``private``: doing so is conservative and avoids depending on an untrusted
    or older server to classify user-specific capability schemas correctly.
    """
    cache_key = _scoped_cache_key(server_name)
    if cache_key is None:
        return None
    with _cache_lock:
        entry = _load_all().get(cache_key)
    if not isinstance(entry, dict):
        return None
    if entry.get("fingerprint") != fingerprint:
        return None
    ttl_ms = entry.get("ttl_ms")
    written_at = entry.get("written_at")
    if isinstance(ttl_ms, (int, float)) and isinstance(written_at, (int, float)):
        if (time.time() - written_at) * 1000.0 >= float(ttl_ms):
            return None
    return entry


def has_cached_entry(server_name: str, fingerprint: str) -> bool:
    return get_cached_entry(server_name, fingerprint) is not None


def write_cache_entry(
    server_name: str,
    fingerprint: str,
    *,
    tools: List[dict],
    utility_tools: Optional[List[dict]] = None,
    ttl_ms: Optional[float] = None,
    cache_scope: Optional[str] = None,
) -> None:
    """Persist schemas under the exact current requesting-user scope."""
    cache_key = _scoped_cache_key(server_name)
    if cache_key is None:
        # Never write an anonymous/shared cache entry while per-user OAuth is
        # configured but no authenticated principal is bound.
        return

    entry = {
        "fingerprint": fingerprint,
        "tools": tools,
        "utility_tools": utility_tools or [],
    }
    if isinstance(ttl_ms, (int, float)):
        entry["ttl_ms"] = ttl_ms
        entry["written_at"] = time.time()
    if cache_scope:
        entry["cache_scope"] = cache_scope
    with _cache_lock:
        data = _load_all()
        if "written_at" not in entry and data.get(cache_key) == entry:
            return
        data[cache_key] = entry
        _save_all(data)


def clear_cache_entry(server_name: str) -> None:
    """Clear current scoped entry, or all scoped entries from admin context."""
    cache_key = _scoped_cache_key(server_name)
    with _cache_lock:
        data = _load_all()
        changed = False
        if cache_key is not None:
            if cache_key in data:
                del data[cache_key]
                changed = True
        elif _is_per_user_oauth_server(server_name):
            # No principal is bound (e.g. config/admin maintenance). Clearing is
            # intentionally destructive across this logical server's cache
            # entries but does not grant access to any cached content.
            prefix = f"{server_name}@@u-v1-"
            for key in list(data):
                if key.startswith(prefix):
                    del data[key]
                    changed = True
        if changed:
            _save_all(data)


def tools_from_cache_entry(entry: dict) -> List[dict]:
    """Return cached MCP tool dicts (name, description, inputSchema)."""
    tools = entry.get("tools")
    return list(tools) if isinstance(tools, list) else []


def utility_tools_from_cache_entry(entry: dict) -> List[dict]:
    util = entry.get("utility_tools")
    return list(util) if isinstance(util, list) else []
