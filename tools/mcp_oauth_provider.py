"""Hermes MCP OAuth provider implementation.

Extracted from :mod:`tools.mcp_oauth_manager` so the manager can key all
mutable OAuth state by an immutable requesting-user scope without entangling
provider protocol behaviour.  The provider keeps the existing MCP SDK
compatibility, metadata persistence, DCR/CIMD recovery, expiry handling, and
bidirectional auth-flow bridge.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _same_endpoint(a: str, b: str) -> bool:
    """Return True when URLs target the same endpoint, ignoring query/fragment."""
    from urllib.parse import urlsplit

    try:
        pa, pb = urlsplit(a), urlsplit(b)
    except ValueError:  # pragma: no cover
        return False
    return (
        pa.scheme == pb.scheme
        and pa.netloc.lower() == pb.netloc.lower()
        and pa.path.rstrip("/") == pb.path.rstrip("/")
    )


def _make_hermes_provider_class() -> Optional[type]:
    """Lazy-import the MCP SDK OAuth provider and return Hermes' subclass."""
    try:
        from mcp.client.auth.oauth2 import OAuthClientProvider
    except ImportError:  # pragma: no cover - SDK required in CI
        return None

    class HermesMCPOAuthProvider(OAuthClientProvider):
        """OAuthClientProvider with Hermes persistence/recovery hooks."""

        def __init__(
            self,
            *args: Any,
            server_name: str = "",
            oauth_scope=None,
            preregistered: bool = False,
            token_user_agent: "str | None" = None,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)
            self._hermes_server_name = server_name
            self._hermes_home = ""
            # Scope is captured when the provider is constructed.  Never
            # re-resolve ambient identity during refresh/background auth.
            self._hermes_oauth_scope = oauth_scope
            self._hermes_preregistered = preregistered
            self._hermes_token_user_agent = token_user_agent

        def _stamp_token_user_agent(self, request):
            ua = getattr(self, "_hermes_token_user_agent", None)
            if ua:
                request.headers["User-Agent"] = ua
            return request

        def _coerce_client_secret_post(self) -> None:
            info = getattr(self.context, "client_info", None)
            if not info or not getattr(info, "client_secret", None):
                return
            method = getattr(info, "token_endpoint_auth_method", None)
            if method not in (None, "none", ""):
                return
            from mcp.shared.auth import OAuthClientInformationFull

            data = info.model_dump(mode="json", exclude_none=True)
            data["token_endpoint_auth_method"] = "client_secret_post"
            self.context.client_info = OAuthClientInformationFull.model_validate(data)

        async def _exchange_token_authorization_code(self, *args: Any, **kwargs: Any):
            self._coerce_client_secret_post()
            request = await super()._exchange_token_authorization_code(*args, **kwargs)
            return self._stamp_token_user_agent(request)

        async def _refresh_token(self):
            self._coerce_client_secret_post()
            request = await super()._refresh_token()
            return self._stamp_token_user_agent(request)

        async def _handle_token_response(self, response):
            """Accept any 2xx token response without exposing token bodies."""
            if 200 <= response.status_code < 300:
                from httpx import HTTPError
                from mcp.client.auth.oauth2 import OAuthTokenError
                from mcp.client.auth.utils import handle_token_response_scopes

                try:
                    token_response = await handle_token_response_scopes(response)
                except (HTTPError, OAuthTokenError):
                    raise OAuthTokenError("Invalid token response") from None
                self.context.current_tokens = token_response
                self.context.update_token_expiry(token_response)
                await self.context.storage.set_tokens(token_response)
                return

            from mcp.client.auth.oauth2 import OAuthTokenError

            raise OAuthTokenError(f"Token exchange failed ({response.status_code})")

        async def _handle_refresh_response(self, response) -> bool:
            """Accept any 2xx refresh response without logging credential bodies."""
            if not (200 <= response.status_code < 300):
                logger.warning("Token refresh failed: %s", response.status_code)
                self.context.clear_tokens()
                return False

            from httpx import HTTPError
            from mcp.shared.auth import OAuthToken
            from pydantic import ValidationError

            try:
                content = await response.aread()
                token_response = OAuthToken.model_validate_json(content)
                self.context.current_tokens = token_response
                self.context.update_token_expiry(token_response)
                await self.context.storage.set_tokens(token_response)
                return True
            except (HTTPError, ValidationError):
                logger.warning("Invalid refresh response: %s", response.status_code)
                self.context.clear_tokens()
                return False

        async def _initialize(self) -> None:
            """Load persisted state, seed expiry, and restore/discover metadata."""
            await super()._initialize()
            tokens = self.context.current_tokens
            if tokens is not None and tokens.expires_in is not None:
                self.context.update_token_expiry(tokens)

            storage = self.context.storage
            from tools.mcp_oauth import HermesTokenStorage

            if isinstance(storage, HermesTokenStorage) and self.context.oauth_metadata is None:
                meta = storage.load_oauth_metadata()
                if meta is not None:
                    self.context.oauth_metadata = meta
                    logger.debug(
                        "MCP OAuth '%s': restored metadata from disk "
                        "(token_endpoint=%s)",
                        self._hermes_server_name,
                        meta.token_endpoint,
                    )

            if tokens is not None and self.context.oauth_metadata is None:
                try:
                    await self._prefetch_oauth_metadata()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "MCP OAuth '%s': pre-flight metadata discovery "
                        "failed (non-fatal): %s",
                        self._hermes_server_name,
                        exc,
                    )

        async def _prefetch_oauth_metadata(self) -> None:
            """Fetch PRM + authorization-server metadata before refresh."""
            from tools.mcp_tool import sdk_httpx

            httpx = sdk_httpx()
            if httpx is None:  # pragma: no cover
                return
            from mcp.client.auth.utils import (
                build_oauth_authorization_server_metadata_discovery_urls,
                build_protected_resource_metadata_discovery_urls,
                create_oauth_metadata_request,
                handle_auth_metadata_response,
                handle_protected_resource_response,
            )

            server_url = self.context.server_url
            async with httpx.AsyncClient(timeout=10.0) as client:
                for url in build_protected_resource_metadata_discovery_urls(None, server_url):
                    req = create_oauth_metadata_request(url)
                    try:
                        resp = await client.send(req)
                    except httpx.HTTPError as exc:
                        logger.debug(
                            "MCP OAuth '%s': PRM discovery to %s failed: %s",
                            self._hermes_server_name,
                            url,
                            exc,
                        )
                        continue
                    prm = await handle_protected_resource_response(resp)
                    if prm:
                        self.context.protected_resource_metadata = prm
                        if prm.authorization_servers:
                            self.context.auth_server_url = str(prm.authorization_servers[0])
                        break

                for url in build_oauth_authorization_server_metadata_discovery_urls(
                    self.context.auth_server_url, server_url
                ):
                    req = create_oauth_metadata_request(url)
                    try:
                        resp = await client.send(req)
                    except httpx.HTTPError as exc:
                        logger.debug(
                            "MCP OAuth '%s': ASM discovery to %s failed: %s",
                            self._hermes_server_name,
                            url,
                            exc,
                        )
                        continue
                    ok, asm = await handle_auth_metadata_response(resp)
                    if not ok:
                        break
                    if asm:
                        self.context.oauth_metadata = asm
                        storage = self.context.storage
                        from tools.mcp_oauth import HermesTokenStorage

                        if isinstance(storage, HermesTokenStorage):
                            storage.save_oauth_metadata(asm)
                        logger.debug(
                            "MCP OAuth '%s': pre-flight ASM discovered "
                            "token_endpoint=%s",
                            self._hermes_server_name,
                            asm.token_endpoint,
                        )
                        break

        def _persist_oauth_metadata_if_changed(self) -> None:
            meta = self.context.oauth_metadata
            if meta is None:
                return
            storage = self.context.storage
            from tools.mcp_oauth import HermesTokenStorage

            if not isinstance(storage, HermesTokenStorage):
                return
            existing = storage.load_oauth_metadata()
            if existing is None or str(existing.token_endpoint) != str(meta.token_endpoint):
                storage.save_oauth_metadata(meta)

        async def _maybe_flag_poisoned_client(self, response: Any) -> None:
            """Detect invalid_client and force safe re-registration when possible."""
            try:
                if self._hermes_preregistered:
                    return
                status = getattr(response, "status_code", None)
                if status not in (400, 401):
                    return
                meta = getattr(self.context, "oauth_metadata", None)
                token_endpoint = (
                    str(meta.token_endpoint)
                    if meta is not None and getattr(meta, "token_endpoint", None)
                    else None
                )
                req = getattr(response, "request", None)
                req_url = str(req.url) if req is not None else None
                if not token_endpoint or not req_url or not _same_endpoint(
                    req_url, token_endpoint
                ):
                    return
                body = await response.aread()
                if not re.search(rb"\binvalid_client\b", body.lower()):
                    return

                storage = self.context.storage
                from tools.mcp_oauth import HermesTokenStorage

                cimd_url = getattr(self.context, "client_metadata_url", None)
                rejected_id = getattr(self.context.client_info, "client_id", None)
                if cimd_url and rejected_id == cimd_url:
                    logger.warning(
                        "MCP OAuth '%s': authorization server rejected our "
                        "Client ID Metadata Document (%s) with invalid_client "
                        "— falling back to dynamic client registration.",
                        self._hermes_server_name,
                        cimd_url,
                    )
                    self.context.client_metadata_url = None
                    if isinstance(storage, HermesTokenStorage):
                        storage.mark_cimd_rejected()

                if isinstance(storage, HermesTokenStorage):
                    storage.poison_client_registration()
                self.context.client_info = None
                self._initialized = False
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "MCP OAuth '%s': invalid_client detection failed (non-fatal): %s",
                    self._hermes_server_name,
                    exc,
                )

        async def async_auth_flow(self, request):  # type: ignore[override]
            # Scope is immutable provider state.  Background refreshes must not
            # consult whichever gateway request happens to be running now.
            try:
                from tools.mcp_oauth_manager import get_manager

                await get_manager().invalidate_if_disk_changed(
                    self._hermes_server_name,
                    hermes_home=self._hermes_home,
                    oauth_scope=self._hermes_oauth_scope,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "MCP OAuth '%s': pre-flow disk-watch failed (non-fatal): %s",
                    self._hermes_server_name,
                    exc,
                )

            # Preserve the MCP SDK async-generator .asend(response) protocol.
            inner = super().async_auth_flow(request)
            try:
                outgoing = await inner.__anext__()
                while True:
                    incoming = yield outgoing
                    await self._maybe_flag_poisoned_client(incoming)
                    outgoing = await inner.asend(incoming)
            except StopAsyncIteration:
                self._persist_oauth_metadata_if_changed()
                return

    return HermesMCPOAuthProvider


_HERMES_PROVIDER_CLS: Optional[type] = _make_hermes_provider_class()
