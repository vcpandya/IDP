"""SaaS connector framework for IDP Kit.

Provides a registry of external integrations (Slack, Notion, GitHub, etc.),
secure credential storage (Fernet-encrypted, key derived from SECRET_KEY),
runtime tool injection into the agent, and skill compatibility checks.

Architectural rules (enforced throughout this package):
- Plaintext credentials never appear in logs or LLM context.
- Credentials are decrypted at execute time only — no in-memory caching.
- Refresh failure → connection marked disconnected, never retried silently.
- All connections are user-scoped.
"""
from idpkit.connectors.base import (
    Connector,
    ConnectorTool,
    ConnectorAuthType,
    ConnectorError,
    ConnectorAuthError,
)
from idpkit.connectors.crypto import encrypt_credentials, decrypt_credentials
from idpkit.connectors.registry import (
    get_connector,
    list_connectors,
    REGISTRY,
)

__all__ = [
    "Connector",
    "ConnectorTool",
    "ConnectorAuthType",
    "ConnectorError",
    "ConnectorAuthError",
    "encrypt_credentials",
    "decrypt_credentials",
    "get_connector",
    "list_connectors",
    "REGISTRY",
]
