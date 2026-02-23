"""IDP Kit plugin system — discover and load plugins from entry points and directories."""

from .manager import PluginManager, plugin_manager

__all__ = ["PluginManager", "plugin_manager"]
