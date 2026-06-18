"""
Plugin Manager for vram_core
==============================

Extensible plugin system for custom audio processing modules.

Features:
    - Plugin discovery and loading
    - Hook-based event system
    - Plugin lifecycle management
    - Plugin metadata and dependency checking

Usage:
    from vram_core.plugin_manager import PluginManager

    pm = PluginManager()
    pm.discover_plugins("plugins/")
    pm.load_plugin("my_plugin")

    # Register hooks
    @pm.hook("on_transcribe")
    def my_hook(audio, result):
        print(f"Transcribed: {result['text']}")
"""

import importlib
import inspect
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Plugin metadata."""
    name: str
    version: str = "0.0.1"
    author: str = "unknown"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    enabled: bool = True
    path: str = ""


class PluginBase(ABC):
    """
    Base class for vram_core plugins.

    Subclass this and implement the required methods:

        class MyPlugin(PluginBase):
            @property
            def info(self) -> PluginInfo:
                return PluginInfo(name="my_plugin", version="1.0.0")

            def on_load(self):
                print("Plugin loaded!")

            def on_unload(self):
                print("Plugin unloaded!")
    """

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Return plugin metadata."""
        pass

    def on_load(self):
        """Called when the plugin is loaded."""
        pass

    def on_unload(self):
        """Called when the plugin is unloaded."""
        pass

    def on_transcribe(self, audio, result) -> Optional[dict]:
        """Hook: called after transcription. Return modified result or None."""
        return None

    def on_diarize(self, audio, result) -> Optional[dict]:
        """Hook: called after diarization."""
        return None

    def on_emotion(self, audio, result) -> Optional[dict]:
        """Hook: called after emotion detection."""
        return None

    def on_noise_reduce(self, audio, cleaned_audio) -> Optional[Any]:
        """Hook: called after noise reduction."""
        return None


class PluginManager:
    """
    Plugin discovery, loading, and lifecycle management.

    Usage:
        pm = PluginManager()
        pm.discover_plugins("./plugins")
        pm.load_plugin("my_plugin")

        # Execute hooks
        result = pm.execute_hook("on_transcribe", audio=audio, result=result)
    """

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self._plugins: Dict[str, PluginBase] = {}
        self._plugin_infos: Dict[str, PluginInfo] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._plugin_dirs = plugin_dirs or []
        self._search_paths: List[str] = []
        self._lock = threading.Lock()

        # Add default plugin dirs
        default_dir = os.path.join(os.path.dirname(__file__), "..", "plugins")
        if os.path.isdir(default_dir):
            self._plugin_dirs.append(os.path.abspath(default_dir))

    def discover_plugins(self, path: str) -> List[str]:
        """
        Discover plugins in a directory.

        Looks for Python files containing a class that inherits from PluginBase.

        Args:
            path: Directory to scan.

        Returns:
            List of discovered plugin names.
        """
        discovered = []
        plugin_dir = Path(path)
        if not plugin_dir.is_dir():
            logger.warning("Plugin directory not found: %s", path)
            return discovered

        for py_file in plugin_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            plugin_name = py_file.stem
            try:
                spec = importlib.util.spec_from_file_location(
                    f"plugins.{plugin_name}", str(py_file)
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find PluginBase subclasses
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            inspect.isclass(attr)
                            and issubclass(attr, PluginBase)
                            and attr is not PluginBase
                        ):
                            self._search_paths.append((plugin_name, attr, str(py_file)))
                            discovered.append(plugin_name)
                            logger.info("Discovered plugin: %s at %s", plugin_name, py_file)
            except Exception as e:
                logger.warning("Error scanning %s: %s", py_file, e)

        return discovered

    def load_plugin(self, name: str) -> bool:
        """
        Load a discovered plugin by name.

        Args:
            name: Plugin name.

        Returns:
            True if loaded successfully.
        """
        for plugin_name, plugin_class, plugin_path in self._search_paths:
            if plugin_name != name:
                continue

            try:
                instance = plugin_class()
                info = instance.info

                # Check dependencies
                missing = self._check_dependencies(info.dependencies)
                if missing:
                    logger.warning(
                        "Plugin %s missing dependencies: %s", name, missing
                    )
                    continue

                self._plugins[name] = instance
                self._plugin_infos[name] = info
                self._plugin_infos[name].path = plugin_path

                # Register hooks
                for hook_name in info.hooks:
                    method = getattr(instance, hook_name, None)
                    if method and callable(method):
                        self._register_hook(hook_name, method)

                instance.on_load()
                logger.info("Plugin loaded: %s v%s", info.name, info.version)
                return True

            except Exception as e:
                logger.error("Failed to load plugin %s: %s", name, e)
                # Clean up partially loaded state
                self._plugins.pop(name, None)
                self._plugin_infos.pop(name, None)
                return False

        logger.warning("Plugin not found: %s", name)
        return False

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        if name not in self._plugins:
            logger.warning("Plugin not loaded: %s", name)
            return False

        plugin = self._plugins[name]
        try:
            plugin.on_unload()
        except Exception as e:
            logger.warning("Error during %s unload: %s", name, e)

        # Remove hooks
        info = self._plugin_infos.get(name)
        if info:
            for hook_name in info.hooks:
                self._remove_hook(hook_name, name)

        del self._plugins[name]
        if name in self._plugin_infos:
            del self._plugin_infos[name]

        logger.info("Plugin unloaded: %s", name)
        return True

    # 鈹€鈹€ Hook System 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _register_hook(self, hook_name: str, callback: Callable):
        """Register a hook callback."""
        with self._lock:
            if hook_name not in self._hooks:
                self._hooks[hook_name] = []
            self._hooks[hook_name].append(callback)

    def _remove_hook(self, hook_name: str, plugin_name: str):
        """Remove hooks from a specific plugin."""
        with self._lock:
            if hook_name in self._hooks:
                self._hooks[hook_name] = [
                    h for h in self._hooks[hook_name]
                    if not hasattr(h, '__self__') or getattr(h.__self__.info, 'name', '') != plugin_name
                ]

    def register_hook(self, hook_name: str, callback: Callable):
        """Manually register a hook callback (external use)."""
        self._register_hook(hook_name, callback)
        logger.info("Hook registered: %s", hook_name)

    def execute_hook(self, hook_name: str, **kwargs) -> List[Any]:
        """
        Execute all callbacks for a hook.

        Args:
            hook_name: Name of the hook to execute.
            **kwargs: Arguments passed to each callback.

        Returns:
            List of return values from callbacks.
        """
        results = []
        with self._lock:
            callbacks = list(self._hooks.get(hook_name, []))
        for callback in callbacks:
            try:
                result = callback(**kwargs)
                results.append(result)
            except Exception as e:
                logger.warning("Hook %s error: %s", hook_name, e)
        return results

    # 鈹€鈹€ Dependency Check 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def _check_dependencies(dependencies: List[str]) -> List[str]:
        """Check which dependencies are missing."""
        missing = []
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
        return missing

    # 鈹€鈹€ Info & Status 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def list_plugins(self) -> List[PluginInfo]:
        """List all loaded plugins."""
        return list(self._plugin_infos.values())

    def get_plugin(self, name: str) -> Optional[PluginBase]:
        """Get a loaded plugin instance by name."""
        return self._plugins.get(name)

    def is_loaded(self, name: str) -> bool:
        """Check if a plugin is loaded."""
        return name in self._plugins

    def list_hooks(self) -> Dict[str, int]:
        """List registered hooks and their callback counts."""
        return {name: len(cbs) for name, cbs in self._hooks.items()}

    def unload_all(self):
        """Unload all plugins."""
        for name in list(self._plugins.keys()):
            self.unload_plugin(name)

    def __del__(self):
        self.unload_all()