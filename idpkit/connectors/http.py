"""Shared HTTP helpers for connector tools."""
from __future__ import annotations

from typing import Any, Optional

import httpx

from idpkit.connectors.base import ConnectorAuthError, ConnectorError

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


async def request(
    method: str,
    url: str,
    *,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    json_body: Optional[Any] = None,
    data: Optional[Any] = None,
    expect_json: bool = True,
) -> Any:
    """Issue an HTTP request and translate auth/HTTP errors into ConnectorError types.

    Returns parsed JSON when expect_json=True (or the raw response for non-JSON).
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.request(
                method, url, headers=headers, params=params, json=json_body, data=data,
            )
    except httpx.HTTPError as exc:
        raise ConnectorError(f"Network error calling {url}: {exc}") from exc

    if resp.status_code in (401, 403):
        raise ConnectorAuthError(
            f"Authentication failed ({resp.status_code}). Reconnect this integration."
        )
    if resp.status_code >= 400:
        body = resp.text[:500]
        raise ConnectorError(f"HTTP {resp.status_code} from {url}: {body}")
    if not expect_json:
        return resp
    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError as exc:
        raise ConnectorError(f"Non-JSON response from {url}: {exc}") from exc
