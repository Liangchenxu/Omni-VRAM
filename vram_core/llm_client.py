"""
LLM Client: Multi-backend LLM integration for meeting analysis
==============================================================

Supports OpenAI API, local llama.cpp, and Chinese LLMs (Qwen/Ernie).
Provides streaming output, token counting, and automatic backend selection.
"""

import logging
import time
import json
import asyncio
from typing import Optional, List, Dict, Any, Generator, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMBackend(Enum):
    """Supported LLM backends."""
    OPENAI = "openai"
    LOCAL_LLAMA = "local_llama"
    QWEN = "qwen"
    ERNIE = "ernie"
    AUTO = "auto"


@dataclass
class LLMConfig:
    """LLM client configuration."""
    backend: LLMBackend = LLMBackend.AUTO
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    # Local llama.cpp settings
    llama_server_url: str = "http://localhost:8081"
    # Qwen settings
    qwen_api_key: Optional[str] = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"
    # Ernie settings
    ernie_api_key: Optional[str] = None
    ernie_secret_key: Optional[str] = None
    # Generation settings
    max_tokens: int = 2048
    temperature: float = 0.3
    top_p: float = 0.9
    timeout: int = 60
    max_context_tokens: int = 128000


@dataclass
class LLMResponse:
    """LLM response wrapper."""
    content: str
    model: str
    backend: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    finish_reason: str = "stop"


@dataclass
class TokenUsage:
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMClient:
    """
    Multi-backend LLM client with automatic backend selection.
    
    Supports:
        - OpenAI API (GPT-4/3.5)
        - Local llama.cpp server
        - Qwen (通义千问) API
        - Ernie (文心一言) API
    
    Features:
        - Streaming output
        - Token counting and limits
        - Automatic retry with fallback
        - Async support
    
    Example:
        >>> client = LLMClient(LLMConfig(backend=LLMBackend.OPENAI))
        >>> response = client.chat("Summarize this meeting...")
        >>> print(response.content)
    """
    
    # Non-retryable error codes/messages
    _NON_RETRYABLE_ERRORS = (401, 403, 404, 422)

    def __init__(self, config: Optional[LLMConfig] = None, max_retries: int = 3, retry_delay: float = 1.0):
        self.config = config or LLMConfig()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._backend = None
        self._client = None
        self._total_usage = TokenUsage()
        self._available: Optional[bool] = None  # Cached availability
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected LLM backend."""
        backend = self.config.backend
        
        if backend == LLMBackend.AUTO:
            backend = self._auto_detect_backend()
        
        if backend == LLMBackend.OPENAI:
            self._init_openai()
        elif backend == LLMBackend.LOCAL_LLAMA:
            self._init_local_llama()
        elif backend == LLMBackend.QWEN:
            self._init_qwen()
        elif backend == LLMBackend.ERNIE:
            self._init_ernie()
        
        self._backend = backend
        logger.info("LLM backend initialized: %s", backend.value)
    
    def _auto_detect_backend(self) -> LLMBackend:
        """Auto-detect available LLM backend."""
        # Try OpenAI first
        if OPENAI_AVAILABLE and self.config.openai_api_key:
            return LLMBackend.OPENAI
        
        # Try Qwen
        if self.config.qwen_api_key:
            return LLMBackend.QWEN
        
        # Try local llama.cpp
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.config.llama_server_url}/health")
            urllib.request.urlopen(req, timeout=2)
            return LLMBackend.LOCAL_LLAMA
        except (ConnectionError, TimeoutError, OSError):
            pass
        
        # Try OpenAI-compatible with environment variable
        import os
        if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            return LLMBackend.OPENAI
        
        logger.warning("No LLM backend available. LLM features will be disabled.")
        return LLMBackend.OPENAI  # Default, will fail gracefully
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required: pip install openai")
        import os
        self._client = openai.OpenAI(
            api_key=self.config.openai_api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=self.config.openai_base_url,
            timeout=self.config.timeout,
        )
    
    def _init_local_llama(self):
        """Initialize local llama.cpp connection."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required for llama.cpp compatibility")
        self._client = openai.OpenAI(
            api_key="not-needed",
            base_url=f"{self.config.llama_server_url}/v1",
            timeout=self.config.timeout,
        )
    
    def _init_qwen(self):
        """Initialize Qwen (通义千问) client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required")
        import os
        self._client = openai.OpenAI(
            api_key=self.config.qwen_api_key or os.environ.get("DASHSCOPE_API_KEY"),
            base_url=self.config.qwen_base_url,
            timeout=self.config.timeout,
        )
    
    def _init_ernie(self):
        """Initialize Ernie (文心一言) client."""
        # Ernie uses OpenAI-compatible API via adapter
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required")
        import os
        self._client = openai.OpenAI(
            api_key=self.config.ernie_api_key or os.environ.get("ERNIE_API_KEY"),
            base_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
            timeout=self.config.timeout,
        )
    
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            prompt: User message
            system_prompt: System prompt for context
            history: Previous conversation history
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            
        Returns:
            LLMResponse with content and metadata
        """
        messages = self._build_messages(prompt, system_prompt, history)
        start_time = time.time()
        
        # Use config defaults if not specified (fix: don't use `or` which
        # would override valid falsy values like temperature=0.0)
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        effective_temperature = temperature if temperature is not None else self.config.temperature

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._get_model_name(),
                    messages=messages,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    top_p=self.config.top_p,
                )

                latency_ms = (time.time() - start_time) * 1000

                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    self._total_usage.prompt_tokens += usage.get("prompt_tokens", 0)
                    self._total_usage.completion_tokens += usage.get("completion_tokens", 0)
                    self._total_usage.total_tokens += usage.get("total_tokens", 0)

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    backend=self._backend.value,
                    usage=usage,
                    latency_ms=latency_ms,
                    finish_reason=response.choices[0].finish_reason or "stop",
                )
            except (RuntimeError, ConnectionError, TimeoutError, OSError, ValueError) as e:
                last_error = e
                # Don't retry non-retryable errors (auth, validation, not found)
                if self._is_non_retryable(e):
                    logger.error(
                        "LLM chat non-retryable error (%s): %s", self._backend.value, e
                    )
                    self._available = False
                    raise
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        "LLM chat failed (%s), attempt %d/%d, retrying in %.1fs: %s",
                        self._backend.value, attempt + 1, self.max_retries, delay, e,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "LLM chat failed after %d attempts (%s): %s",
                        self.max_retries, self._backend.value, e,
                    )
        self._available = False
        raise last_error
    
    def chat_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Stream chat completion response.
        
        Yields:
            Response text chunks
        """
        messages = self._build_messages(prompt, system_prompt, history)
        
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        effective_temperature = temperature if temperature is not None else self.config.temperature

        for attempt in range(self.max_retries):
            try:
                stream = self._client.chat.completions.create(
                    model=self._get_model_name(),
                    messages=messages,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    top_p=self.config.top_p,
                    stream=True,
                )

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # Success, no need to retry
            except (RuntimeError, ConnectionError, TimeoutError, OSError, ValueError) as e:
                if self._is_non_retryable(e):
                    logger.error("LLM stream non-retryable error: %s", e)
                    self._available = False
                    raise
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "LLM stream failed, attempt %d/%d, retrying in %.1fs: %s",
                        attempt + 1, self.max_retries, delay, e,
                    )
                    time.sleep(delay)
                else:
                    logger.error("LLM stream failed after %d attempts: %s", self.max_retries, e)
                    self._available = False
                    raise
    
    async def achat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Async chat completion."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.chat(prompt, system_prompt, history, max_tokens),
        )
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Build message list for API call."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def _get_model_name(self) -> str:
        """Get model name for current backend."""
        if self._backend == LLMBackend.OPENAI:
            return self.config.openai_model
        elif self._backend == LLMBackend.LOCAL_LLAMA:
            return "local"
        elif self._backend == LLMBackend.QWEN:
            return self.config.qwen_model
        elif self._backend == LLMBackend.ERNIE:
            return "ernie-bot-4"
        return "unknown"
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (approximate: 1 token ≈ 4 chars for English, 2 chars for Chinese)."""
        # Simple estimation
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        non_ascii_chars = len(text) - ascii_chars
        return (ascii_chars // 4) + (non_ascii_chars // 2)
    
    @property
    def total_usage(self) -> TokenUsage:
        """Get total token usage."""
        return self._total_usage
    
    @property
    def backend(self) -> LLMBackend:
        """Get current backend."""
        return self._backend
    
    @staticmethod
    def _is_non_retryable(error: Exception) -> bool:
        """Check if an error is non-retryable (auth, validation, etc.)."""
        # Check for HTTP status code on OpenAI API errors
        status = getattr(error, 'status_code', None)
        if status in (401, 403, 404, 422):
            return True
        # Check for common auth error messages
        msg = str(error).lower()
        if any(kw in msg for kw in ('api key', 'unauthorized', 'forbidden', 'invalid_key', 'authentication')):
            return True
        return False

    @property
    def is_available(self) -> bool:
        """Check if LLM is available (uses cached result to avoid repeated calls)."""
        if self._available is not None:
            return self._available
        try:
            response = self.chat("Hi", max_tokens=5)
            self._available = bool(response.content)
            return self._available
        except (RuntimeError, ConnectionError, TimeoutError, OSError, ValueError):
            self._available = False
            return False
