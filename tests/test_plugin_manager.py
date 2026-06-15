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

import unittest
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


class TestPluginInfo(unittest.TestCase):
    """Test PluginInfo data class."""

    def test_default_values(self):
        info = PluginInfo(name="test")
        self.assertEqual(info.name, "test")
        self.assertEqual(info.version, "0.0.1")
        self.assertEqual(info.author, "unknown")
        self.assertTrue(info.enabled)
        self.assertEqual(info.dependencies, [])
        self.assertEqual(info.hooks, [])

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
        self.assertEqual(info.version, "2.0.0")
        self.assertEqual(len(info.dependencies), 2)
        self.assertFalse(info.enabled)


class TestPluginBase(unittest.TestCase):
    """Test PluginBase abstract class."""

    def test_cannot_instantiate_directly(self):
        """PluginBase cannot be instantiated (abstract)."""
        with self.assertRaises(TypeError):
            PluginBase()

    def test_simple_plugin_info(self):
        p = SimplePlugin()
        self.assertEqual(p.info.name, "simple_plugin")
        self.assertEqual(p.info.version, "1.0.0")

    def test_simple_plugin_lifecycle(self):
        p = SimplePlugin()
        self.assertFalse(p.loaded)
        p.on_load()
        self.assertTrue(p.loaded)
        p.on_unload()
        self.assertTrue(p.unloaded)

    def test_on_transcribe_returns_dict(self):
        p = SimplePlugin()
        result = p.on_transcribe(b"audio", {"text": "hello"})
        self.assertIsNotNone(result)
        self.assertTrue(result["modified"])

    def test_on_diarize_returns_none(self):
        p = SimplePlugin()
        result = p.on_diarize(b"audio", {})
        self.assertIsNone(result)

    def test_on_emotion_returns_none(self):
        p = SimplePlugin()
        result = p.on_emotion(b"audio", {})
        self.assertIsNone(result)

    def test_on_noise_reduce_returns_none(self):
        p = SimplePlugin()
        result = p.on_noise_reduce(b"audio", b"clean")
        self.assertIsNone(result)


class TestPluginManagerInit(unittest.TestCase):
    """Test PluginManager initialization."""

    def test_init_default(self):
        pm = PluginManager()
        self.assertEqual(len(pm._plugins), 0)
        self.assertEqual(len(pm._hooks), 0)

    def test_init_with_plugin_dirs(self):
        pm = PluginManager(plugin_dirs=["/tmp/plugins"])
        self.assertIn("/tmp/plugins", pm._plugin_dirs)


class TestPluginManagerLoad(unittest.TestCase):
    """Test plugin loading."""

    def test_load_direct_instance(self):
        """Test loading by registering directly."""
        pm = PluginManager()
        plugin = SimplePlugin()
        info = plugin.info

        # Manually add to search paths and load
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        result = pm.load_plugin("simple_plugin")
        self.assertTrue(result)
        self.assertTrue(pm.is_loaded("simple_plugin"))

    def test_load_nonexistent_plugin(self):
        """Loading non-existent plugin returns False."""
        pm = PluginManager()
        result = pm.load_plugin("nonexistent")
        self.assertFalse(result)

    def test_load_broken_plugin(self):
        """Loading plugin that raises on_load returns False."""
        pm = PluginManager()
        pm._search_paths.append(("broken_plugin", BrokenPlugin, ""))
        result = pm.load_plugin("broken_plugin")
        self.assertFalse(result)
        self.assertFalse(pm.is_loaded("broken_plugin"))

    def test_load_with_missing_deps(self):
        """Loading plugin with missing dependencies returns False."""
        pm = PluginManager()
        pm._search_paths.append(("dependent_plugin", DependentPlugin, ""))
        result = pm.load_plugin("dependent_plugin")
        self.assertFalse(result)

    def test_is_loaded_false_for_unloaded(self):
        pm = PluginManager()
        self.assertFalse(pm.is_loaded("anything"))


class TestPluginManagerUnload(unittest.TestCase):
    """Test plugin unloading."""

    def test_unload_existing(self):
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm.load_plugin("simple_plugin")
        result = pm.unload_plugin("simple_plugin")
        self.assertTrue(result)
        self.assertFalse(pm.is_loaded("simple_plugin"))

    def test_unload_nonexistent(self):
        pm = PluginManager()
        result = pm.unload_plugin("ghost")
        self.assertFalse(result)

    def test_unload_all(self):
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm._search_paths.append(("no_hook_plugin", NoHookPlugin, ""))
        pm.load_plugin("simple_plugin")
        pm.load_plugin("no_hook_plugin")
        self.assertEqual(len(pm._plugins), 2)
        pm.unload_all()
        self.assertEqual(len(pm._plugins), 0)


class TestPluginManagerHooks(unittest.TestCase):
    """Test hook registration and execution."""

    def test_register_external_hook(self):
        pm = PluginManager()
        callback = MagicMock(return_value="result")
        pm.register_hook("on_test", callback)
        self.assertIn("on_test", pm.list_hooks())
        self.assertEqual(pm.list_hooks()["on_test"], 1)

    def test_execute_hook_calls_callback(self):
        pm = PluginManager()
        callback = MagicMock(return_value="ok")
        pm.register_hook("on_test", callback)
        results = pm.execute_hook("on_test", audio=b"data", result={})
        self.assertEqual(len(results), 1)
        callback.assert_called_once()

    def test_execute_hook_with_no_callbacks(self):
        pm = PluginManager()
        results = pm.execute_hook("nonexistent_hook")
        self.assertEqual(results, [])

    def test_execute_hook_multiple_callbacks(self):
        pm = PluginManager()
        cb1 = MagicMock(return_value="r1")
        cb2 = MagicMock(return_value="r2")
        pm.register_hook("on_test", cb1)
        pm.register_hook("on_test", cb2)
        results = pm.execute_hook("on_test")
        self.assertEqual(len(results), 2)

    def test_execute_hook_callback_error(self):
        """Hook execution continues even if one callback raises."""
        pm = PluginManager()
        bad_cb = MagicMock(side_effect=RuntimeError("fail"))
        good_cb = MagicMock(return_value="ok")
        pm.register_hook("on_test", bad_cb)
        pm.register_hook("on_test", good_cb)
        results = pm.execute_hook("on_test")
        # bad_cb fails silently, good_cb still runs
        self.assertEqual(len(results), 1)

    def test_plugin_hooks_registered_on_load(self):
        """Loading a plugin with hooks registers them."""
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm.load_plugin("simple_plugin")
        hooks = pm.list_hooks()
        self.assertIn("on_transcribe", hooks)

    def test_plugin_hooks_removed_on_unload(self):
        """Unloading a plugin removes its hooks."""
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm.load_plugin("simple_plugin")
        pm.unload_plugin("simple_plugin")
        # After unload, hook should be empty or removed
        hooks = pm.list_hooks()
        if "on_transcribe" in hooks:
            self.assertEqual(hooks["on_transcribe"], 0)


class TestPluginManagerDiscovery(unittest.TestCase):
    """Test plugin discovery from directory."""

    def test_discover_nonexistent_dir(self):
        """Discovery of non-existent directory returns empty list."""
        pm = PluginManager()
        result = pm.discover_plugins("/nonexistent/path/xyz")
        self.assertEqual(result, [])

    def test_discover_empty_dir(self):
        """Discovery of empty directory returns empty list."""
        pm = PluginManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pm.discover_plugins(tmpdir)
            self.assertEqual(result, [])

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
            self.assertIn("discovered_plugin", discovered)

    def test_discover_skips_underscore_files(self):
        """Discovery skips files starting with underscore."""
        pm = PluginManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "__init__.py").write_text("", encoding="utf-8")
            (Path(tmpdir) / "_private.py").write_text("", encoding="utf-8")
            result = pm.discover_plugins(tmpdir)
            self.assertEqual(result, [])


class TestPluginManagerGetAndList(unittest.TestCase):
    """Test plugin querying methods."""

    def test_get_plugin(self):
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm.load_plugin("simple_plugin")
        plugin = pm.get_plugin("simple_plugin")
        self.assertIsNotNone(plugin)
        self.assertIsInstance(plugin, SimplePlugin)

    def test_get_plugin_not_loaded(self):
        pm = PluginManager()
        self.assertIsNone(pm.get_plugin("ghost"))

    def test_list_plugins(self):
        pm = PluginManager()
        pm._search_paths.append(("simple_plugin", SimplePlugin, ""))
        pm.load_plugin("simple_plugin")
        plugins = pm.list_plugins()
        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0].name, "simple_plugin")

    def test_list_plugins_empty(self):
        pm = PluginManager()
        self.assertEqual(pm.list_plugins(), [])


class TestDependencyCheck(unittest.TestCase):
    """Test dependency checking."""

    def test_check_missing_dependency(self):
        missing = PluginManager._check_dependencies(["nonexistent_xyz_module"])
        self.assertIn("nonexistent_xyz_module", missing)

    def test_check_existing_dependency(self):
        missing = PluginManager._check_dependencies(["json", "os", "sys"])
        self.assertEqual(missing, [])

    def test_check_mixed_dependencies(self):
        missing = PluginManager._check_dependencies(["json", "nonexistent_xyz"])
        self.assertEqual(len(missing), 1)


if __name__ == "__main__":
    unittest.main()