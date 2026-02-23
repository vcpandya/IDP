"""IDP Kit custom exceptions."""


class IDPKitError(Exception):
    """Base exception for all IDP Kit errors."""


class LLMError(IDPKitError):
    """Error communicating with an LLM provider."""


class LLMMaxRetriesError(LLMError):
    """Maximum retries exceeded for an LLM API call."""


class ParsingError(IDPKitError):
    """Error parsing a document."""


class IndexingError(IDPKitError):
    """Error building a document index."""


class StorageError(IDPKitError):
    """Error with file storage operations."""


class ConfigurationError(IDPKitError):
    """Error with configuration loading or validation."""


class AuthenticationError(IDPKitError):
    """Error with authentication or authorization."""


class ToolError(IDPKitError):
    """Error executing a Smart Tool."""
