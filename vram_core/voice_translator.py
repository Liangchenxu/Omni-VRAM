"""
Voice Translator Module for vram_core
=======================================

Speech-to-Speech translation pipeline:
Speech 锟?ASR 锟?Text Translation 锟?TTS 锟?Translated Speech

Supports multiple translation backends:
1. **transformers** (MarianMT / NLLB): Local neural translation
2. **deep-translator** (Google/MyMemory): Free API translation

Usage:
    from vram_core.voice_translator import VoiceTranslator

    vt = VoiceTranslator(source_lang="zh", target_lang="en")
    result = vt.translate_audio(audio, sample_rate=16000)
    print(result.translated_text)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

_DEEP_TRANSLATOR_AVAILABLE = False
try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    _DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TranslationResult:
    """Translation result."""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: float = 0.0
    backend_used: str = "unknown"
    audio_original: Optional[np.ndarray] = None
    audio_translated: Optional[np.ndarray] = None
    translated_sample_rate: int = 24000


class VoiceTranslator:
    """
    Speech-to-Speech translation pipeline.

    Combines ASR (Whisper), text translation, and TTS into a single pipeline.

    Args:
        source_lang: Source language code (e.g. "zh", "en", "ja").
        target_lang: Target language code.
        asr_model: Whisper model size for ASR.
        translation_backend: Translation backend ("auto", "marian", "google").

    Usage:
        vt = VoiceTranslator(source_lang="zh", target_lang="en")
        result = vt.translate_audio(audio, sample_rate=16000)
        # result.translated_text, result.audio_translated
    """

    # MarianMT model mappings for common language pairs
    MARIAN_MODELS = {
        ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
        ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
        ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
        ("ja", "en"): "Helsinki-NLP/opus-mt-jap-en",
        ("en", "ko"): "Helsinki-NLP/opus-mt-en-ko",
        ("ko", "en"): "Helsinki-NLP/opus-mt-ko-en",
        ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
        ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
        ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
        ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
        ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
        ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
        ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
        ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    }

    def __init__(
        self,
        source_lang: str = "zh",
        target_lang: str = "en",
        asr_model: str = "base",
        translation_backend: str = "auto",
        whisper_device: Optional[int] = None,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.asr_model_size = asr_model

        self._whisper = None
        self._translator = None
        self._tts = None
        self._backend = "none"
        self._model_cache: Dict[str, object] = {}

        self._init_asr(whisper_device)
        self._init_translator(translation_backend)
        self._init_tts()

    def _init_asr(self, device: Optional[int] = None):
        """Initialize Whisper ASR."""
        try:
            from vram_core.whisper import WhisperBridge
            self._whisper = WhisperBridge(model_size=self.asr_model_size, device_id=device)
            logger.info("ASR initialized: whisper-%s", self.asr_model_size)
        except Exception as e:
            logger.warning("ASR init failed: %s", e)

    def _init_translator(self, backend: str):
        """Initialize translation backend."""
        if backend in ("auto", "marian") and _TRANSFORMERS_AVAILABLE:
            model_key = (self.source_lang, self.target_lang)
            if model_key in self.MARIAN_MODELS:
                self._backend = "marian"
                logger.info("Translation backend: MarianMT for %s锟?s", self.source_lang, self.target_lang)
                return

        if backend in ("auto", "google") and _DEEP_TRANSLATOR_AVAILABLE:
            self._backend = "google"
            logger.info("Translation backend: Google Translate for %s锟?s", self.source_lang, self.target_lang)
            return

        logger.warning("No translation backend available. Install transformers or deep-translator.")

    def _init_tts(self):
        """Initialize TTS for target language."""
        try:
            from vram_core.tts_engine import TTSEngine
            voice_map = TTSEngine.DEFAULT_VOICES
            voice = voice_map.get(self.target_lang)
            self._tts = TTSEngine(voice=voice)
            logger.info("TTS initialized for %s", self.target_lang)
        except Exception as e:
            logger.warning("TTS init failed: %s", e)

    # 鈹€鈹€ Translation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _get_marian_model(self, src: str, tgt: str):
        """Get or load MarianMT model."""
        cache_key = f"{src}_{tgt}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model_name = self.MARIAN_MODELS.get((src, tgt))
        if not model_name:
            raise ValueError(f"No MarianMT model for {src}->{tgt}")

        logger.info("Loading MarianMT: %s", model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        self._model_cache[cache_key] = (model, tokenizer)
        return model, tokenizer

    def translate_text(self, text: str) -> str:
        """
        Translate text from source to target language.

        Args:
            text: Text to translate.

        Returns:
            Translated text string.
        """
        if not text.strip():
            return ""

        if self._backend == "marian":
            return self._translate_marian(text)
        elif self._backend == "google":
            return self._translate_google(text)
        else:
            raise RuntimeError("No translation backend available")

    def _translate_marian(self, text: str) -> str:
        """Translate using MarianMT."""
        model, tokenizer = self._get_marian_model(self.source_lang, self.target_lang)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

    def _translate_google(self, text: str) -> str:
        """Translate using Google/Memory API."""
        try:
            translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
            # Google has character limits, split if needed
            if len(text) > 4500:
                chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
                results = [translator.translate(chunk) for chunk in chunks]
                return "".join(results)
            return translator.translate(text)
        except Exception as e:
            logger.warning("Google translate failed: %s, trying MyMemory", e)
            translator = MyMemoryTranslator(source=self.source_lang, target=self.target_lang)
            return translator.translate(text)

    # 鈹€鈹€ Audio Translation Pipeline 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def translate_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        synthesize_speech: bool = True,
    ) -> TranslationResult:
        """
        Full audio translation pipeline: ASR 锟?Translate 锟?TTS.

        Args:
            audio: Source audio (float32, mono).
            sample_rate: Sample rate.
            synthesize_speech: Whether to generate translated audio via TTS.

        Returns:
            TranslationResult with original text, translated text, and audio.
        """
        # Step 1: ASR
        if self._whisper is None:
            raise RuntimeError("ASR not initialized")

        asr_result = self._whisper.transcribe(audio, sample_rate=sample_rate, language=self.source_lang)
        original_text = asr_result.text if hasattr(asr_result, 'text') else str(asr_result)

        if not original_text.strip():
            return TranslationResult(
                original_text="", translated_text="",
                source_lang=self.source_lang, target_lang=self.target_lang,
            )

        # Step 2: Translate
        translated_text = self.translate_text(original_text)

        # Step 3: TTS (optional)
        translated_audio = None
        translated_sr = 24000
        if synthesize_speech and self._tts and translated_text.strip():
            try:
                tts_result = self._tts.synthesize(translated_text)
                translated_audio = tts_result.audio
                translated_sr = tts_result.sample_rate
            except Exception as e:
                logger.warning("TTS synthesis failed: %s", e)

        return TranslationResult(
            original_text=original_text,
            translated_text=translated_text,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            backend_used=self._backend,
            audio_original=audio,
            audio_translated=translated_audio,
            translated_sample_rate=translated_sr,
        )

    # 鈹€鈹€ Utility 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def available_backends() -> List[str]:
        """List available translation backends."""
        backends = []
        if _TRANSFORMERS_AVAILABLE:
            backends.append("marian")
        if _DEEP_TRANSLATOR_AVAILABLE:
            backends.append("google")
        return backends

    @staticmethod
    def supported_pairs() -> List[tuple]:
        """List supported language pairs for MarianMT."""
        return list(VoiceTranslator.MARIAN_MODELS.keys())

    @property
    def backend(self) -> str:
        return self._backend

    def close(self):
        """Release resources."""
        self._model_cache.clear()
        if self._whisper:
            self._whisper.close()
        if self._tts:
            self._tts.close()

    def __del__(self):
        self.close()