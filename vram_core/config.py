"""
vram_core Configuration Management
===================================

Centralized configuration for the vram_core project.
Reads from .env file and environment variables with type validation.

Usage:
    from vram_core.config import config

    # Access configuration
    api_key = config.openai_api_key
    device = config.device
    model_path = config.whisper_model_path

    # Or use as dict
    config_dict = config.to_dict()

Environment Variables:
    OPENAI_API_KEY          - OpenAI API key for Whisper API
    OPENAI_MODEL            - OpenAI Whisper model (default: whisper-1)
    WHISPER_CPP_PATH        - Path to whisper.cpp executable directory
    WHISPER_MODEL_PATH      - Path to GGML model file (e.g., models/ggml-base.bin)
    WHISPER_LANGUAGE        - Default language code (zh, en, ja, etc.)
    WHISPER_BACKEND         - Preferred backend: auto/faster_whisper/whisper_cpp/openai_api
    WHISPER_DEVICE          - Whisper device: cuda/cpu (default: cuda)
    WHISPER_COMPUTE_TYPE    - CTranslate2 compute type: int8/float16/float32
    VRAM_DEVICE             - Compute device: cuda or cpu (default: cuda)
    VRAM_LOG_LEVEL          - Logging level (default: INFO)
    VRAM_SAMPLE_RATE        - Target audio sample rate (default: 16000)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

# Attempt to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    # Look for .env in project root (parent of vram_core/)
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        logger.debug("Loaded .env from %s", _env_path)
    else:
        # Also check current working directory
        _cwd_env = Path.cwd() / ".env"
        if _cwd_env.exists():
            load_dotenv(_cwd_env)
            logger.debug("Loaded .env from %s", _cwd_env)
except ImportError:
    logger.debug(
        "python-dotenv not installed, .env file will not be loaded. "
        "Install with: pip install python-dotenv"
    )


def _get_env(key: str, default: Any = None, cast_type: type = str) -> Any:
    """
    Get environment variable with type casting and default value.

    Args:
        key: Environment variable name.
        default: Default value if not set.
        cast_type: Type to cast value to (str, int, float, bool, Path).

    Returns:
        Cast value or default.
    """
    value = os.environ.get(key)
    if value is None or value.strip() == "":
        return default

    value = value.strip()

    try:
        if cast_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif cast_type == Path:
            return Path(value)
        elif cast_type == int:
            return int(value)
        elif cast_type == float:
            return float(value)
        else:
            return value
    except (ValueError, TypeError) as e:
        logger.warning(
            "Invalid value for %s='%s': %s. Using default: %s", key, value, e, default
        )
        return default


class OmniConfig:
    """
    Centralized configuration for vram_core.

    All configuration values are read from environment variables with
    sensible defaults. Supports .env file loading via python-dotenv.

    Thread-safe: Configuration is read-only after initialization.
    """

    # 鈹€鈹€ OpenAI API Configuration 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    openai_api_key: Optional[str]
    openai_model: str

    # 鈹€鈹€ Whisper.cpp Configuration 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    whisper_cpp_path: Optional[Path]
    whisper_model_path: Optional[Path]

    # 鈹€鈹€ Faster-Whisper Configuration 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    whisper_backend: str
    whisper_device: str
    whisper_compute_type: str

    # 鈹€鈹€ General Settings 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    language: Optional[str]
    device: str
    sample_rate: int
    log_level: str

    def __init__(self):
        """Load all configuration from environment variables."""
        # OpenAI API
        self.openai_api_key = _get_env("OPENAI_API_KEY", default=None, cast_type=str)
        self.openai_model = _get_env(
            "OPENAI_MODEL", default="whisper-1", cast_type=str
        )

        # Whisper.cpp
        self.whisper_cpp_path = _get_env(
            "WHISPER_CPP_PATH", default=None, cast_type=Path
        )
        self.whisper_model_path = _get_env(
            "WHISPER_MODEL_PATH", default=None, cast_type=Path
        )

        # Faster-Whisper
        self.whisper_backend = _get_env(
            "WHISPER_BACKEND", default="auto", cast_type=str
        )
        self.whisper_device = _get_env(
            "WHISPER_DEVICE", default="cuda", cast_type=str
        )
        self.whisper_compute_type = _get_env(
            "WHISPER_COMPUTE_TYPE", default="float16", cast_type=str
        )

        # General
        self.language = _get_env("WHISPER_LANGUAGE", default=None, cast_type=str)
        self.device = _get_env("VRAM_DEVICE", default="cuda", cast_type=str)
        self.sample_rate = _get_env("VRAM_SAMPLE_RATE", default=16000, cast_type=int)
        self.log_level = _get_env("VRAM_LOG_LEVEL", default="INFO", cast_type=str)

        # Validate
        self._validate()

    # ── Class-level allowed values ────────────────────────────────────
    _VALID_DEVICES = ("cuda", "cpu")
    _VALID_BACKENDS = ("auto", "faster_whisper", "whisper_cpp", "openai_api")
    _VALID_COMPUTE_TYPES = ("int8", "float16", "float32")
    _VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    _VALID_SAMPLE_RATES = (8000, 16000, 22050, 44100, 48000)
    _MIN_SAMPLE_RATE = 1000
    _MAX_SAMPLE_RATE = 192000

    def _validate(self):
        """Validate configuration values and fix invalid ones."""
        errors: list = []
        warnings: list = []

        # Device validation
        if self.device not in self._VALID_DEVICES:
            warnings.append(f"Invalid device '{self.device}', falling back to 'cuda'")
            self.device = "cuda"

        # Sample rate validation
        if self.sample_rate <= 0:
            warnings.append(f"Invalid sample_rate {self.sample_rate}, falling back to 16000")
            self.sample_rate = 16000
        elif self.sample_rate < self._MIN_SAMPLE_RATE or self.sample_rate > self._MAX_SAMPLE_RATE:
            warnings.append(
                f"sample_rate {self.sample_rate} outside reasonable range "
                f"[{self._MIN_SAMPLE_RATE}, {self._MAX_SAMPLE_RATE}], "
                "proceeding but may cause issues"
            )

        # Whisper backend validation
        if self.whisper_backend not in self._VALID_BACKENDS:
            warnings.append(
                f"Invalid whisper_backend '{self.whisper_backend}', falling back to 'auto'"
            )
            self.whisper_backend = "auto"

        # Whisper device validation
        if self.whisper_device not in self._VALID_DEVICES:
            warnings.append(
                f"Invalid whisper_device '{self.whisper_device}', falling back to 'cuda'"
            )
            self.whisper_device = "cuda"

        # Whisper compute type validation
        if self.whisper_compute_type not in self._VALID_COMPUTE_TYPES:
            warnings.append(
                f"Invalid whisper_compute_type '{self.whisper_compute_type}', falling back to 'float16'"
            )
            self.whisper_compute_type = "float16"

        # Log level validation
        if self.log_level.upper() not in self._VALID_LOG_LEVELS:
            warnings.append(f"Invalid log_level '{self.log_level}', falling back to INFO")
            self.log_level = "INFO"
        else:
            self.log_level = self.log_level.upper()

        # Whisper model path validation
        if self.whisper_model_path is not None:
            if not self.whisper_model_path.exists():
                warnings.append(
                    f"Whisper model not found: {self.whisper_model_path}. "
                    "Local whisper.cpp transcription may fail."
                )

        # Language validation
        if self.language is not None:
            self.language = self.language.lower().strip()

        # Emit warnings
        for w in warnings:
            logger.warning(w)

    def validate(self) -> list:
        """
        Public validation that returns a list of (non-fatal) issues found.

        Returns:
            List of warning/error strings. Empty list means all OK.
        """
        issues: list = []

        if self.device not in self._VALID_DEVICES:
            issues.append(f"Invalid device: {self.device}")
        if self.whisper_backend not in self._VALID_BACKENDS:
            issues.append(f"Invalid whisper_backend: {self.whisper_backend}")
        if self.whisper_device not in self._VALID_DEVICES:
            issues.append(f"Invalid whisper_device: {self.whisper_device}")
        if self.whisper_compute_type not in self._VALID_COMPUTE_TYPES:
            issues.append(f"Invalid whisper_compute_type: {self.whisper_compute_type}")
        if self.log_level.upper() not in self._VALID_LOG_LEVELS:
            issues.append(f"Invalid log_level: {self.log_level}")
        if self.sample_rate <= 0:
            issues.append(f"Invalid sample_rate: {self.sample_rate}")
        if self.whisper_model_path is not None and not self.whisper_model_path.exists():
            issues.append(f"Whisper model file not found: {self.whisper_model_path}")
        if self.whisper_cpp_path is not None and not self.whisper_cpp_path.exists():
            issues.append(f"whisper.cpp path not found: {self.whisper_cpp_path}")

        return issues

    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.openai_api_key is not None and len(self.openai_api_key) > 0

    @property
    def has_whisper_cpp(self) -> bool:
        """Check if whisper.cpp path is configured and exists."""
        if self.whisper_cpp_path is None:
            return False
        return self.whisper_cpp_path.exists()

    @property
    def has_whisper_model(self) -> bool:
        """Check if whisper model file is configured and exists."""
        if self.whisper_model_path is None:
            return False
        return self.whisper_model_path.exists()

    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA device is requested and potentially available."""
        return self.device == "cuda"

    @property
    def has_faster_whisper(self) -> bool:
        """Check if faster-whisper package is installed."""
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary (masks sensitive values)."""
        return {
            "openai_api_key": "***" if self.has_openai_key else None,
            "openai_model": self.openai_model,
            "whisper_cpp_path": str(self.whisper_cpp_path) if self.whisper_cpp_path else None,
            "whisper_model_path": str(self.whisper_model_path) if self.whisper_model_path else None,
            "whisper_backend": self.whisper_backend,
            "whisper_device": self.whisper_device,
            "whisper_compute_type": self.whisper_compute_type,
            "has_faster_whisper": self.has_faster_whisper,
            "language": self.language,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "log_level": self.log_level,
        }

    def __repr__(self) -> str:
        return (
            f"OmniConfig(device={self.device}, "
            f"sample_rate={self.sample_rate}, "
            f"has_openai_key={self.has_openai_key}, "
            f"has_whisper_cpp={self.has_whisper_cpp})"
        )


# 鈹€鈹€ Global singleton instance 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
config = OmniConfig()


def setup_logging(level: Optional[str] = None):
    """
    Configure logging for the entire vram_core package.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
               If None, uses config.log_level.
    """
    log_level = level or config.log_level
    numeric_level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("vram_core logging initialized at %s level", log_level)
