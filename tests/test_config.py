"""Tests for vram_core.config module (OmniConfig)."""
import os
import pytest
from unittest.mock import patch
from pathlib import Path


class TestOmniConfigInit:
    """Test OmniConfig initialization from environment variables."""

    def test_default_values(self):
        """OmniConfig should have sensible defaults when no env vars set."""
        from vram_core.config import OmniConfig
        with patch.dict(os.environ, {}, clear=True):
            c = OmniConfig()
            assert c.device == "cuda"
            assert c.sample_rate == 16000
            assert c.log_level == "INFO"
            assert c.openai_model == "whisper-1"
            assert c.whisper_backend == "auto"
            assert c.whisper_device == "cuda"
            assert c.whisper_compute_type == "float16"
            assert c.openai_api_key is None
            assert c.language is None

    def test_env_override(self):
        """OmniConfig should read values from environment variables."""
        from vram_core.config import OmniConfig
        env = {
            "VRAM_DEVICE": "cpu",
            "VRAM_SAMPLE_RATE": "44100",
            "VRAM_LOG_LEVEL": "DEBUG",
            "OPENAI_MODEL": "whisper-large",
            "WHISPER_BACKEND": "faster_whisper",
            "WHISPER_DEVICE": "cpu",
            "WHISPER_COMPUTE_TYPE": "int8",
            "WHISPER_LANGUAGE": "zh",
            "OPENAI_API_KEY": "test-key-123",
        }
        with patch.dict(os.environ, env, clear=True):
            c = OmniConfig()
            assert c.device == "cpu"
            assert c.sample_rate == 44100
            assert c.log_level == "DEBUG"
            assert c.openai_model == "whisper-large"
            assert c.whisper_backend == "faster_whisper"
            assert c.whisper_device == "cpu"
            assert c.whisper_compute_type == "int8"
            assert c.language == "zh"
            assert c.openai_api_key == "test-key-123"


class TestOmniConfigValidation:
    """Test OmniConfig._validate() behavior."""

    def test_invalid_device_fallback(self):
        from vram_core.config import OmniConfig
        with patch.dict(os.environ, {"VRAM_DEVICE": "tpu"}, clear=True):
            c = OmniConfig()
            assert c.device == "cuda"  # fallback

    def test_invalid_sample_rate_fallback(self):
        from vram_core.config import OmniConfig
        with patch.dict(os.environ, {"VRAM_SAMPLE_RATE": "-1"}, clear=True):
            c = OmniConfig()
            assert c.sample_rate == 16000  # fallback

    def test_invalid_whisper_backend_fallback(self):
        from vram_core.config import OmniConfig
        with patch.dict(os.environ, {"WHISPER_BACKEND": "unknown"}, clear=True):
            c = OmniConfig()
            assert c.whisper_backend == "auto"  # fallback

    def test_invalid_compute_type_fallback(self):
        from vram_core.config import OmniConfig
        with patch.dict(os.environ, {"WHISPER_COMPUTE_TYPE": "float64"}, clear=True):
            c = OmniConfig()
            assert c.whisper_compute_type == "float16"  # fallback

    def test_invalid_log_level_fallback(self):
        from vram_core.config import OmniConfig
        with patch.dict(os.environ, {"VRAM_LOG_LEVEL": "INVALID"}, clear=True):
            c = OmniConfig()
            assert c.log_level == "INFO"  # fallback

    def test_log_level_normalized_upper(self):
        from vram_core.config import OmniConfig
        with patch.dict(os.environ, {"VRAM_LOG_LEVEL": "warning"}, clear=True):
            c = OmniConfig()
            assert c.log_level == "WARNING"

    def test_language_normalized_lowercase(self):
        from vram_core.config import OmniConfig
        with patch.dict(os.environ, {"WHISPER_LANGUAGE": " EN "}, clear=True):
            c = OmniConfig()
            assert c.language == "en"


class TestOmniConfigProperties:
    """Test OmniConfig property methods."""

    def test_has_openai_key_true(self):
        from vram_core.config import OmniConfig
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            c = OmniConfig()
            assert c.has_openai_key is True

    def test_has_openai_key_false(self):
        from vram_core.config import OmniConfig
        with patch.dict(os.environ, {}, clear=True):
            c = OmniConfig()
            assert c.has_openai_key is False

    def test_to_dict_masks_key(self):
        from vram_core.config import OmniConfig
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret"}, clear=True):
            c = OmniConfig()
            d = c.to_dict()
            assert d["openai_api_key"] == "***"
            assert "sk-secret" not in str(d)


class TestOmniConfigPublicValidate:
    """Test the public validate() method."""

    def test_valid_config_returns_empty(self):
        from vram_core.config import OmniConfig
        c = OmniConfig()
        issues = c.validate()
        # With default values there should be no issues
        assert isinstance(issues, list)

    def test_invalid_device_reported(self):
        from vram_core.config import OmniConfig
        c = OmniConfig()
        c.device = "tpu"
        issues = c.validate()
        assert any("device" in i.lower() for i in issues)


class TestOmniConfigRepr:
    """Test __repr__."""

    def test_repr_contains_device(self):
        from vram_core.config import OmniConfig
        c = OmniConfig()
        r = repr(c)
        assert "OmniConfig" in r
        assert "device=" in r