"""Plugin manager for IDP Kit — discovers, loads, and manages plugins."""

import importlib
import importlib.metadata
import logging
from pathlib import Path
from typing import Any

from idpkit.core.events import event_bus
from idpkit.tools.base import BaseTool

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "idpkit.plugins"


class PluginInfo:
    """Metadata about a loaded plugin."""

    __slots__ = ("name", "version", "description", "module", "tools", "hooks")

    def __init__(self, name: str, version: str = "0.0.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self.module = None
        self.tools: list[BaseTool] = []
        self.hooks: list[tuple[str, Any]] = []


class PluginManager:
    """Discovers and loads IDP Kit plugins.

    Plugins can register:
    - Smart Tools (subclasses of BaseTool)
    - Event hooks (functions bound to event_bus events)

    Discovery methods:
    1. Python entry points (group: idpkit.plugins)
    2. Plugin directory scanning
    """

    def __init__(self):
        self._plugins: dict[str, PluginInfo] = {}
        self._tool_registry: dict[str, BaseTool] = {}

    @property
    def plugins(self) -> dict[str, PluginInfo]:
        return dict(self._plugins)

    @property
    def tools(self) -> dict[str, BaseTool]:
        return dict(self._tool_registry)

    def load_entry_points(self) -> int:
        """Discover plugins via Python entry points (pip-installed packages)."""
        count = 0
        try:
            eps = importlib.metadata.entry_points()
            group = eps.get(ENTRY_POINT_GROUP, []) if isinstance(eps, dict) else eps.select(group=ENTRY_POINT_GROUP)
        except Exception:
            group = []

        for ep in group:
            try:
                module = ep.load()
                info = self._register_module(ep.name, module)
                if info:
                    count += 1
                    logger.info(f"Loaded plugin from entry point: {ep.name}")
            except Exception as e:
                logger.error(f"Failed to load plugin {ep.name}: {e}")
        return count

    def load_directory(self, directory: str | Path) -> int:
        """Discover plugins from a directory of Python files."""
        directory = Path(directory)
        if not directory.is_dir():
            return 0

        count = 0
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"idpkit_plugin_{path.stem}", str(path)
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    info = self._register_module(path.stem, module)
                    if info:
                        count += 1
                        logger.info(f"Loaded plugin from file: {path.name}")
            except Exception as e:
                logger.error(f"Failed to load plugin {path.name}: {e}")
        return count

    def register_tool(self, tool: BaseTool) -> None:
        """Register a single tool instance."""
        self._tool_registry[tool.name] = tool
        logger.info(f"Registered plugin tool: {tool.name}")

    def _register_module(self, name: str, module: Any) -> PluginInfo | None:
        """Register a plugin module, extracting tools and hooks."""
        if name in self._plugins:
            logger.warning(f"Plugin {name} already loaded, skipping")
            return None

        info = PluginInfo(
            name=name,
            version=getattr(module, "__version__", "0.0.0"),
            description=getattr(module, "__description__", ""),
        )
        info.module = module

        # Register tools — look for TOOLS list or BaseTool subclasses
        tools_list = getattr(module, "TOOLS", [])
        if not tools_list:
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseTool)
                    and attr is not BaseTool
                ):
                    try:
                        tools_list.append(attr())
                    except Exception as e:
                        logger.error(f"Failed to instantiate tool {attr_name}: {e}")

        for tool in tools_list:
            self._tool_registry[tool.name] = tool
            info.tools.append(tool)

        # Register event hooks — look for HOOKS dict {event_name: handler}
        hooks = getattr(module, "HOOKS", {})
        for event_name, handler in hooks.items():
            event_bus.on(event_name, handler)
            info.hooks.append((event_name, handler))

        # Call plugin init if present
        setup_fn = getattr(module, "setup", None)
        if callable(setup_fn):
            try:
                setup_fn(self)
            except Exception as e:
                logger.error(f"Plugin {name} setup() failed: {e}")

        self._plugins[name] = info
        return info

    def unload(self, name: str) -> bool:
        """Unload a plugin, removing its tools and hooks."""
        info = self._plugins.pop(name, None)
        if not info:
            return False

        for tool in info.tools:
            self._tool_registry.pop(tool.name, None)

        for event_name, handler in info.hooks:
            event_bus.off(event_name, handler)

        logger.info(f"Unloaded plugin: {name}")
        return True

    def list_plugins(self) -> list[dict]:
        """List all loaded plugins with their metadata."""
        return [
            {
                "name": info.name,
                "version": info.version,
                "description": info.description,
                "tools": [t.name for t in info.tools],
                "hooks": [h[0] for h in info.hooks],
            }
            for info in self._plugins.values()
        ]


# Global plugin manager instance
plugin_manager = PluginManager()
