"""Base classes for SaaS connectors."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


class ConnectorError(Exception):
    """Base error class for connector failures (user-visible message)."""


class ConnectorAuthError(ConnectorError):
    """Authentication / authorisation failure — caller should mark connection disconnected."""


class ConnectorAuthType(str, enum.Enum):
    API_KEY = "api_key"          # single token / API key
    COMPOSITE = "composite"      # multiple required fields (e.g. S3: key+secret+bucket+region)
    OAUTH2 = "oauth2"            # OAuth 2.0 authorization-code flow


@dataclass
class ConnectorField:
    """A credential field collected from the user via the Connections UI."""
    key: str
    label: str
    type: str = "password"            # 'text' | 'password' | 'select'
    placeholder: str = ""
    required: bool = True
    help: str = ""
    options: Optional[list[dict[str, str]]] = None  # for type='select'


@dataclass
class ConnectorTool:
    """An LLM-callable tool exposed by a connector when the user is connected.

    `executor` receives `(args, credentials)` and returns a JSON-serialisable dict.
    Credentials are decrypted just-in-time and discarded after the call.
    """
    name: str                                  # canonical tool name e.g. "slack_send_message"
    description: str
    parameters: dict[str, Any]                 # JSON-Schema parameters
    executor: Callable[[dict, dict], Awaitable[dict]]

    def to_openai_function(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Connector:
    """A SaaS integration definition — declarative metadata + behaviour hooks."""
    id: str                                                # short id, e.g. "slack"
    display_name: str
    description: str
    icon: str = "fa-plug"                                  # FontAwesome icon class
    auth_type: ConnectorAuthType = ConnectorAuthType.API_KEY
    fields: list[ConnectorField] = field(default_factory=list)
    tools: list[ConnectorTool] = field(default_factory=list)
    docs_url: str = ""

    # OAuth2 flow hooks (only used when auth_type == OAUTH2)
    oauth_authorize_url_builder: Optional[Callable[[str, str], str]] = None
    oauth_exchange: Optional[Callable[[str, str], Awaitable[dict]]] = None
    oauth_refresh: Optional[Callable[[dict], Awaitable[dict]]] = None

    # Health check — returns (ok, account_label_or_error)
    health_check: Optional[Callable[[dict], Awaitable[tuple[bool, str]]]] = None

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def field_summary(self) -> list[dict]:
        return [
            {
                "key": f.key,
                "label": f.label,
                "type": f.type,
                "placeholder": f.placeholder,
                "required": f.required,
                "help": f.help,
                "options": f.options,
            }
            for f in self.fields
        ]

    def public_metadata(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "auth_type": self.auth_type.value,
            "fields": self.field_summary(),
            "tools": [
                {"name": t.name, "description": t.description}
                for t in self.tools
            ],
            "docs_url": self.docs_url,
        }
