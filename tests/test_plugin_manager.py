"""
Tests for vram_core.plugin_manager module.

Covers:
    - PluginInfo data class
    - PluginBase abstract class
    - PluginManager (discover, load, unload, hooks, dependency check)
    - Plugin lifecycle (on_load, on_unload)
    - Hook registration and execution
    - Edge cases: missing deps, non-existent plugin, unload_all
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from vram_core.plugin_manager import (
    PluginInfo,
    PluginBase,
    PluginManager,
)


# 鈹€鈹€ Test Plugin Implementations 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class SimplePlugin(PluginBase):
    """A minimal test plugin."""

    def __init__(self):
        self.loaded = False
        self.unloaded = False

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="simple_plugin",
            version="1.0.0",
            author="test",
            description="A simple test plugin",
            hooks=["on_transcribe"],
        )

    def on_load(self):
        self.loaded = True

    def on_unload(self):
        self.unloaded = True

    def on_transcribe(self, audio, result):
        return {"modified": True, **(result or {})}


class NoHookPlugin(PluginBase):
    """Plugin without hooks."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(name="no_hook_plugin", version="0.1.0")

    def on_load(self):
        pass

    def on_unload(self):
        pass


class DependentPlugin(PluginBase):
    """Plugin with dependencies."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="dependent_plugin",
            dependencies=["nonexistent_module_xyz"],
            hooks=["on_emotion"],
        )


class BrokenPlugin(PluginBase):
    """Plugin that raises on load."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(name="broken_plugin")

    def on_load(self):
        raise RuntimeError("Intentional load failure")


class TestPluginInfo:
    """Test PluginInfo data class."""

    def test_default_values(self):
        info = PluginInfo(name="test")
        assert info.name == "test"
        assert info.version == "0.0.1"
        assert info.author == "unknown"
        assert info.enabled
        assert info.dependencies == []
        assert info.hooks == []

    def test_custom_values(self):
        info = PluginInfo(
            name="my_plugin",
            version="2.0.0",
            author="alice",
            description="Test plugin",
            dependencies=["numpy", "torch"],
            hooks=["on_transcribe", "on_emotion"],
            enabled=False,
        )
        assert info.version == "2.0.0"
        assert len(info.dependencies) == 2
        assert not info.enabled


class TestPluginBase:
    """Test PluginBase abstract class."""

    def test_cannot_instantiate_directly(self):
        """PluginBase cannot be instantiated (abstract)."""
        with pytest.raises(TypeError):
            PluginBase()

    def test_simple_plugin_info(self):
        p = SimplePlugin()
        assert p.info.name == "simple_plugin"
        assert p.info.version == "1.0.0"

    def test_simple_plugin_lifecycle(self):
        p = SimplePlugin()
        assert not p.loaded
        p.on_load()
        assert p.loaded
        p.on_unload()
        assert p.unloaded

    def test_on_transcribe_returns_dict(self):
        p = SimplePlugin()
        result = p.on_transcribe(b"audio", {"text": "hello"})
        assert result is not None
        assert result["modified"]

    def test_on_diarize_returns_none(self):
        p = SimplePlugin()
        result = p.on_diarize(b"audio", {})
        assert result is None

    def test_on_emotion_returns_none(self):
        p = SimplePlugin()
        result = p.on_emotion(b"audio", {})
        assert result is None

    def test_on_noise_reduce_returns_none(self):
        p = SimplePlugin()
        result = p.on_noise_reduce(b"audio", b"clean")
        assert result is None


class TestPluginManagerInit:
    """Test PluginManager initialization."""

    def test_init_default(self):
        pm = PluginManager()
        assert len(pm._plugins) == 0
        assert len(pm._hooks) == 0

    def test_init_with_plugin_dirs(self):
        pm = PluginManager(plugin_dirs=["/tmp/plugins"])
        assert "/tmp/plugins" in pm._plugin_dirs


class TestPluginManagerLoad:
    """Test plugin loading."""

    def test_load_direct_instance(self):
        """Test loading by registering directly."""
        pm = PluginManager()
        plugin = SimplePlugin()
        info = plugin.info

        # Manually add to search paths and load
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        result = pm.load_plugin("simple_plugin")
        assert result
        assert pm.is_loaded("simple_plugin")

    def test_load_nonexistent_plugin(self):
        """Loading non-existent plugin returns False."""
        pm = PluginManager()
        result = pm.load_plugin("nonexistent")
        assert not result

    def test_load_broken_plugin(self):
        """Loading plugin that raises on_load returns False."""
        pm = PluginManager()
        pm._search_paths.append(("broken_plugin", BrokenPlugin, ""))
        result = pm.load_plugin("broken_plugin")
        assert not result
        assert not pm.is_loaded("broken_plugin")

    def test_load_with_missing_deps(self):
        """Loading plugin with missing dependencies returns False."""
        pm = PluginManager()
        pm._search_paths.append(("dependent_plugin", DependentPlugin, ""))
        result = pm.load_plugin("dependent_plugin")
        assert not result

    def test_is_loaded_false_for_unloaded(self):
        pm = PluginManager()
        assert not pm.is_loaded("anything")


class TestPluginManagerUnload:
    """Test plugin unloading."""

    def test_unload_existing(self):
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm.load_plugin("simple_plugin")
        result = pm.unload_plugin("simple_plugin")
        assert result
        assert not pm.is_loaded("simple_plugin")

    def test_unload_nonexistent(self):
        pm = PluginManager()
        result = pm.unload_plugin("ghost")
        assert not result

    def test_unload_all(self):
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm._search_paths.append(("no_hook_plugin", NoHookPlugin, ""))
        pm.load_plugin("simple_plugin")
        pm.load_plugin("no_hook_plugin")
        assert len(pm._plugins) == 2
        pm.unload_all()
        assert len(pm._plugins) == 0


class TestPluginManagerHooks:
    """Test hook registration and execution."""

    def test_register_external_hook(self):
        pm = PluginManager()
        callback = MagicMock(return_value="result")
        pm.register_hook("on_test", callback)
        assert "on_test" in pm.list_hooks()
        assert pm.list_hooks()["on_test"] == 1

    def test_execute_hook_calls_callback(self):
        pm = PluginManager()
        callback = MagicMock(return_value="ok")
        pm.register_hook("on_test", callback)
        results = pm.execute_hook("on_test", audio=b"data", result={})
        assert len(results) == 1
        callback.assert_called_once()

    def test_execute_hook_with_no_callbacks(self):
        pm = PluginManager()
        results = pm.execute_hook("nonexistent_hook")
        assert results == []

    def test_execute_hook_multiple_callbacks(self):
        pm = PluginManager()
        cb1 = MagicMock(return_value="r1")
        cb2 = MagicMock(return_value="r2")
        pm.register_hook("on_test", cb1)
        pm.register_hook("on_test", cb2)
        results = pm.execute_hook("on_test")
        assert len(results) == 2

    def test_execute_hook_callback_error(self):
        """Hook execution continues even if one callback raises."""
        pm = PluginManager()
        bad_cb = MagicMock(side_effect=RuntimeError("fail"))
        good_cb = MagicMock(return_value="ok")
        pm.register_hook("on_test", bad_cb)
        pm.register_hook("on_test", good_cb)
        results = pm.execute_hook("on_test")
        # bad_cb fails silently, good_cb still runs
        assert len(results) == 1

    def test_plugin_hooks_registered_on_load(self):
        """Loading a plugin with hooks registers them."""
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm.load_plugin("simple_plugin")
        hooks = pm.list_hooks()
        assert "on_transcribe" in hooks

    def test_plugin_hooks_removed_on_unload(self):
        """Unloading a plugin removes its hooks."""
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm.load_plugin("simple_plugin")
        pm.unload_plugin("simple_plugin")
        # After unload, hook should be empty or removed
        hooks = pm.list_hooks()
        if "on_transcribe" in hooks:
            assert hooks["on_transcribe"] == 0


class TestPluginManagerDiscovery:
    """Test plugin discovery from directory."""

    def test_discover_nonexistent_dir(self):
        """Discovery of non-existent directory returns empty list."""
        pm = PluginManager()
        result = pm.discover_plugins("/nonexistent/path/xyz")
        assert result == []

    def test_discover_empty_dir(self):
        """Discovery of empty directory returns empty list."""
        pm = PluginManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pm.discover_plugins(tmpdir)
            assert result == []

    def test_discover_with_plugin_file(self):
        """Discovery finds PluginBase subclass in .py files."""
        pm = PluginManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_code = '''
from vram_core.plugin_manager import PluginBase, PluginInfo

class TestDiscoveryPlugin(PluginBase):
    @property
    def info(self):
        return PluginInfo(name="discovered", version="0.1.0")
'''
            plugin_file = Path(tmpdir) / "discovered_plugin.py"
            plugin_file.write_text(plugin_code, encoding="utf-8")

            discovered = pm.discover_plugins(tmpdir)
            assert "discovered_plugin" in discovered

    def test_discover_skips_underscore_files(self):
        """Discovery skips files starting with underscore."""
        pm = PluginManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "__init__.py").write_text("", encoding="utf-8")
            (Path(tmpdir) / "_private.py").write_text("", encoding="utf-8")
            result = pm.discover_plugins(tmpdir)
            assert result == []


class TestPluginManagerGetAndList:
    """Test plugin querying methods."""

    def test_get_plugin(self):
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm.load_plugin("simple_plugin")
        plugin = pm.get_plugin("simple_plugin")
        assert plugin is not None
        assert isinstance(plugin, SimplePlugin)

    def test_get_plugin_not_loaded(self):
        pm = PluginManager()
        assert pm.get_plugin("ghost") is None

    def test_list_plugins(self):
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm.load_plugin("simple_plugin")
        plugins = pm.list_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "simple_plugin"

    def test_list_plugins_empty(self):
        pm = PluginManager()
        assert pm.list_plugins() == []


class TestDependencyCheck:
    """Test dependency checking."""

    def test_check_missing_dependency(self):
        missing = PluginManager._check_dependencies(["nonexistent_xyz_module"])
        assert "nonexistent_xyz_module" in missing

    def test_check_existing_dependency(self):
        missing = PluginManager._check_dependencies(["json", "os", "sys"])
        assert missing == []

    def test_check_mixed_dependencies(self):
        missing = PluginManager._check_dependencies(["json", "nonexistent_xyz"])
        assert len(missing) == 1


