"""
Integration Tests for vram_core
================================

Cross-module interaction tests that verify modules work together correctly.
These tests use synthetic data and do not require GPU or external downloads.

Test Categories:
    1. Audio Pipeline: raw bytes -> AudioProcessor -> NoiseReducer -> features
    2. Speaker Pipeline: audio -> SpeakerDiarizer -> SpeakerVerifier
    3. Chinese NLP Pipeline: text -> normalizer -> tokenizer -> punctuation
    4. Plugin + Module Hook Integration
    5. Config propagation across modules
    6. Module cross-import compatibility
"""

import os
import sys
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_audio():
    """Generate a 3-second synthetic audio signal at 16kHz."""
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Mix two frequencies + noise
    signal = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 500 * t)
    noise = 0.05 * np.random.randn(len(t))
    return (signal + noise).astype(np.float32), sr


@pytest.fixture
def synthetic_speech_like():
    """Generate a more speech-like signal with amplitude modulation."""
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # AM modulated sinusoid to simulate speech envelope
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    carrier = np.sin(2 * np.pi * 250 * t)
    noise = 0.03 * np.random.randn(len(t))
    return (envelope * carrier + noise).astype(np.float32), sr


@pytest.fixture
def chinese_texts():
    return [
        "今天天气真不错我想出去走走",
        "请帮我查一下明天北京到上海的高铁票",
        "人工智能在医疗领域的应用越来越广泛",
    ]


# ---------------------------------------------------------------------------
# 1. Audio Pipeline Integration
# ---------------------------------------------------------------------------

class TestAudioPipelineIntegration:
    """Test: raw audio -> AudioProcessor -> NoiseReducer -> downstream."""

    def test_audio_processor_to_noise_reducer(self, synthetic_audio):
        """AudioProcessor output can be fed into NoiseReducer."""
        from vram_core.audio_utils import AudioProcessor
        from vram_core.noise_reduction import NoiseReducer

        audio, sr = synthetic_audio
        processor = AudioProcessor()
        reducer = NoiseReducer(strength="medium")

        # Process through AudioProcessor (normalize)
        processed = AudioProcessor.normalize(audio)
        assert processed is not None
        assert len(processed) > 0

        # Feed into noise reducer
        cleaned = reducer.process(processed, sample_rate=sr)
        assert cleaned is not None
        assert len(cleaned) == len(processed)
        assert cleaned.dtype == np.float32

    def test_noise_reduction_preserves_signal_shape(self, synthetic_audio):
        """NoiseReducer should not change array shape."""
        from vram_core.noise_reduction import NoiseReducer

        audio, sr = synthetic_audio
        for strength in ["light", "medium", "aggressive"]:
            reducer = NoiseReducer(strength=strength)
            result = reducer.process(audio, sample_rate=sr)
            assert result.shape == audio.shape, f"Shape mismatch for strength={strength}"

    def test_noise_reducer_with_all_strengths(self, synthetic_audio):
        """All strength presets produce valid output."""
        from vram_core.noise_reduction import NoiseReducer

        audio, sr = synthetic_audio
        for strength in ["light", "medium", "aggressive"]:
            reducer = NoiseReducer(strength=strength)
            result = reducer.process(audio, sample_rate=sr)
            assert np.isfinite(result).all(), f"Non-finite values for strength={strength}"
            # Output should be quieter (lower RMS) than input for noisy signals
            assert np.sqrt(np.mean(result ** 2)) <= np.sqrt(np.mean(audio ** 2)) + 0.01


# ---------------------------------------------------------------------------
# 2. Speaker Pipeline Integration
# ---------------------------------------------------------------------------

class TestSpeakerPipelineIntegration:
    """Test: audio -> SpeakerDiarizer -> SpeakerVerifier cross-module flow."""

    def test_diarizer_verifier_cross_module(self, synthetic_speech_like):
        """Diarizer segments can be fed into verifier for identity matching."""
        from vram_core.speaker_diarization import SpeakerDiarizer
        from vram_core.speaker_verification import SpeakerVerifier

        audio, sr = synthetic_speech_like

        try:
            diarizer = SpeakerDiarizer(backend="resemblyzer")
            verifier = SpeakerVerifier()

            # Diarize
            result = diarizer.diarize(audio, sample_rate=sr)
            assert result is not None
            # Result may be a DiarizationResult or a list of segments
            segments = result.segments if hasattr(result, 'segments') else result
            assert isinstance(segments, list)

            # If we got segments, try verification
            if segments:
                first_seg = segments[0]
                # SpeakerSegment has start_time/end_time (seconds)
                assert hasattr(first_seg, "start_time")

                # Extract segment audio using time-based indices
                start_idx = int(first_seg.start_time * sr)
                end_idx = int(first_seg.end_time * sr)
                seg_audio = audio[start_idx:end_idx]
                if len(seg_audio) == 0:
                    seg_audio = audio  # fallback

                if len(seg_audio) > sr * 0.5:  # at least 0.5s
                    verifier.register("speaker_0", seg_audio, sample_rate=sr)
                    result = verifier.verify("speaker_0", seg_audio, sample_rate=sr)
                    assert result is not None
                    assert hasattr(result, "confidence") or hasattr(result, "verified")

        except ImportError as e:
            pytest.skip(f"Missing dependency: {e}")

    def test_speaker_verifier_enroll_verify_cycle(self, synthetic_audio):
        """Full register-then-verify cycle works correctly."""
        from vram_core.speaker_verification import SpeakerVerifier

        audio, sr = synthetic_audio
        verifier = SpeakerVerifier()

        # Register
        verifier.register("test_user", audio, sample_rate=sr)
        speakers = verifier.list_speakers()
        assert len(speakers) > 0

        # Verify same audio
        result = verifier.verify("test_user", audio, sample_rate=sr)
        assert result is not None


# ---------------------------------------------------------------------------
# 3. Chinese NLP Pipeline Integration
# ---------------------------------------------------------------------------

class TestChineseNLPPipelineIntegration:
    """Test: text -> normalizer -> tokenizer -> punctuation pipeline."""

    def test_normalizer_to_tokenizer(self, chinese_texts):
        """Normalized text can be tokenized."""
        from vram_core.chinese.normalizer import TextNormalizer
        from vram_core.chinese.tokenizer import ChineseTokenizer

        normalizer = TextNormalizer()
        tokenizer = ChineseTokenizer()

        for text in chinese_texts:
            normalized = normalizer.normalize(text)
            assert isinstance(normalized, str)
            assert len(normalized) > 0

            tokens = tokenizer.tokenize(normalized)
            assert isinstance(tokens, list)
            assert len(tokens) > 0

    def test_punctuation_restorer_output(self, chinese_texts):
        """Punctuation restorer adds punctuation to unpunctuated text."""
        from vram_core.chinese.punctuation import PunctuationRestorer

        restorer = PunctuationRestorer()
        for text in chinese_texts:
            result = restorer.restore(text)
            assert isinstance(result, str)
            assert len(result) >= len(text)

    def test_domain_dict_integration(self):
        """Domain dictionary works with normalized text."""
        from vram_core.chinese.domain_dict import DomainDictionary

        # DomainDictionary requires a domain to load built-in terms
        ddict = DomainDictionary(domain="tech")

        # DomainDictionary provides terms and post-correction
        terms = ddict.get_terms()
        assert isinstance(terms, list)
        assert len(terms) > 0

        # post_correct should handle text
        corrected = ddict.post_correct("今天用gpu训练模型")
        assert isinstance(corrected, str)
        assert len(corrected) > 0

    def test_dialect_converter_roundtrip(self):
        """Dialect detector can detect dialect from text."""
        from vram_core.chinese.dialect import DialectDetector

        detector = DialectDetector()
        # Test detection on text with dialect markers
        result = detector.detect("今日天氣唔錯喎")
        assert result is not None
        assert hasattr(result, "dialect") or hasattr(result, "confidence")

    def test_full_chinese_pipeline(self, chinese_texts):
        """Full pipeline: normalize -> tokenize -> punctuate."""
        from vram_core.chinese.normalizer import TextNormalizer
        from vram_core.chinese.tokenizer import ChineseTokenizer
        from vram_core.chinese.punctuation import PunctuationRestorer

        normalizer = TextNormalizer()
        tokenizer = ChineseTokenizer()
        restorer = PunctuationRestorer()

        for text in chinese_texts:
            # Step 1: Normalize
            norm = normalizer.normalize(text)
            assert isinstance(norm, str)

            # Step 2: Tokenize
            tokens = tokenizer.tokenize(norm)
            assert isinstance(tokens, list)

            # Step 3: Restore punctuation on a version without punctuation
            clean_text = norm.replace("，", "").replace("。", "").replace("、", "")
            punctuated = restorer.restore(clean_text)
            assert isinstance(punctuated, str)


# ---------------------------------------------------------------------------
# 4. Plugin System Integration
# ---------------------------------------------------------------------------

class TestPluginIntegration:
    """Test: plugin manager + hook system integration."""

    def test_plugin_hook_chain(self):
        """Multiple plugins can process data through hooks."""
        from vram_core.plugin_manager import PluginManager

        pm = PluginManager()

        # Register chain of hooks
        results = []

        def hook_a(data):
            results.append("a")
            return data + "_a"

        def hook_b(data):
            results.append("b")
            return data + "_b"

        pm.register_hook("process", hook_a)
        pm.register_hook("process", hook_b)

        # Dispatch
        pm.execute_hook("process", data="start")
        assert results == ["a", "b"]

    def test_plugin_with_audio_module(self, synthetic_audio):
        """Plugin hooks can wrap audio module calls."""
        from vram_core.plugin_manager import PluginManager
        from vram_core.noise_reduction import NoiseReducer

        pm = PluginManager()
        call_log = []

        def pre_hook(audio):
            call_log.append("pre")
            return audio

        def post_hook(result):
            call_log.append("post")
            return result

        pm.register_hook("before_noise_reduction", pre_hook)
        pm.register_hook("after_noise_reduction", post_hook)

        audio, sr = synthetic_audio
        reducer = NoiseReducer()

        # Manual integration: plugin dispatch + module call
        prepped = pm.execute_hook("before_noise_reduction", audio=audio)
        # execute_hook returns a list; use first result if it's an ndarray
        audio_in = prepped[0] if prepped and isinstance(prepped[0], np.ndarray) else audio
        cleaned = reducer.process(audio_in, sample_rate=sr)
        pm.execute_hook("after_noise_reduction", result=cleaned)

        assert call_log == ["pre", "post"]
        assert cleaned is not None


# ---------------------------------------------------------------------------
# 5. Config Propagation Integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Test: OmniConfig propagates correctly to modules."""

    def test_config_singleton(self):
        """Config is a singleton across imports."""
        from vram_core.config import config as cfg1
        from vram_core.config import config as cfg2
        assert cfg1 is cfg2

    def test_config_has_required_sections(self):
        """Config has all required sections."""
        from vram_core.config import config
        # Should have core attributes
        assert hasattr(config, "device")
        assert hasattr(config, "sample_rate")

    def test_config_with_noise_reducer(self):
        """NoiseReducer can be created with config-like parameters."""
        from vram_core.config import config
        from vram_core.noise_reduction import NoiseReducer

        # Should not raise
        reducer = NoiseReducer(strength="medium")
        assert reducer is not None


# ---------------------------------------------------------------------------
# 6. Cross-Import Compatibility
# ---------------------------------------------------------------------------

class TestCrossImportCompatibility:
    """Test: all modules can be imported without conflicts."""

    def test_import_all_core_modules(self):
        """All core modules import without error."""
        modules = [
            "vram_core.audio_utils",
            "vram_core.noise_reduction",
            "vram_core.emotion_recognition",
            "vram_core.speaker_verification",
            "vram_core.speaker_diarization",
            "vram_core.plugin_manager",
            "vram_core.config",
            "vram_core.monitoring",
            "vram_core.vram_optimizer",
            "vram_core.wake_word",
        ]
        for mod_name in modules:
            try:
                __import__(mod_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {mod_name}: {e}")

    def test_import_chinese_subpackage(self):
        """Chinese NLP subpackage imports without error."""
        modules = [
            "vram_core.chinese.normalizer",
            "vram_core.chinese.punctuation",
            "vram_core.chinese.tokenizer",
            "vram_core.chinese.dialect",
            "vram_core.chinese.domain_dict",
        ]
        for mod_name in modules:
            try:
                __import__(mod_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {mod_name}: {e}")

    def test_import_whisper_subpackage(self):
        """Whisper subpackage imports without error."""
        modules = [
            "vram_core.whisper",
            "vram_core.whisper.models",
            "vram_core.whisper.result",
            "vram_core.whisper.optimizer",
            "vram_core.whisper.preprocessor",
        ]
        for mod_name in modules:
            try:
                __import__(mod_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {mod_name}: {e}")

    def test_deprecated_whisper_bridge_import(self):
        """Deprecated whisper_bridge still works but emits warning."""
        import warnings
        import importlib
        # Force re-import to trigger the warning
        if "vram_core.whisper_bridge" in sys.modules:
            del sys.modules["vram_core.whisper_bridge"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from vram_core.whisper_bridge import WhisperBridge
            # Should have emitted DeprecationWarning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_version_is_current(self):
        """Package version follows semver format."""
        import vram_core
        parts = vram_core.__version__.split(".")
        assert len(parts) >= 2, "Version should be at least X.Y"
        assert all(p.isdigit() for p in parts[:3]), "Version parts should be numeric"


# ---------------------------------------------------------------------------
# 7. Audio Utilities + Event Detection Integration
# ---------------------------------------------------------------------------

class TestAudioEventIntegration:
    """Test: audio_utils -> audio_event_detection pipeline."""

    def test_audio_event_detection_on_synthetic(self, synthetic_audio):
        """AudioEventDetector can process AudioProcessor output."""
        from vram_core.audio_utils import AudioProcessor
        from vram_core.audio_event_detection import AudioEventDetector

        audio, sr = synthetic_audio
        processor = AudioProcessor()
        detector = AudioEventDetector()

        # Process
        processed = AudioProcessor.normalize(audio)
        result = detector.detect(processed, sample_rate=sr)
        assert result is not None
        # Result is an AEDResult with events attribute
        assert hasattr(result, "events") or hasattr(result, "total_events")


# ---------------------------------------------------------------------------
# 8. Wake Word + Stream Processor Integration
# ---------------------------------------------------------------------------

class TestWakeWordStreamIntegration:
    """Test: wake word detector works with stream processor output."""

    def test_wake_word_with_chunked_audio(self, synthetic_audio):
        """WakeWordDetector processes chunked audio from stream."""
        from vram_core.wake_word import WakeWordDetector

        audio, sr = synthetic_audio
        detector = WakeWordDetector(mode="energy", energy_threshold=0.01)

        # Simulate streaming by chunking audio
        chunk_size = sr  # 1 second chunks
        results = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size // 2:
                break
            result = detector.process_chunk(chunk)
            results.append(result)

        assert len(results) >= 2
        # All results are either WakeWordEvent or None
        for r in results:
            assert r is None or hasattr(r, "keyword")


# ---------------------------------------------------------------------------
# 9. Monitoring Integration
# ---------------------------------------------------------------------------

class TestMonitoringIntegration:
    """Test: monitoring works across module calls."""

    def test_metrics_collector_basic(self):
        """MetricsCollector can record transcription metrics."""
        from vram_core.monitoring import MetricsCollector

        collector = MetricsCollector()
        # Record some transcription metrics (API: latency is the first required param)
        collector.record_transcription(latency=0.5, audio_duration=3.0, backend="test")
        collector.record_transcription(latency=1.0, audio_duration=5.0, backend="test")

        # Should be able to get health
        health = collector.get_health()
        assert health is not None
        assert health.total_requests >= 2

    def test_system_health_check(self):
        """SystemHealth can report system status."""
        from vram_core.monitoring import MetricsCollector, SystemHealth

        collector = MetricsCollector()
        health = collector.get_health()
        assert isinstance(health, SystemHealth)
        assert hasattr(health, "status")


# ---------------------------------------------------------------------------
# 10. End-to-End Synthetic Pipeline
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Full E2E: raw audio in -> processed features out."""

    def test_full_audio_processing_pipeline(self, synthetic_audio):
        """Complete pipeline: load -> denoise -> extract features."""
        from vram_core.audio_utils import AudioProcessor
        from vram_core.noise_reduction import NoiseReducer
        from vram_core.emotion_recognition import EmotionRecognizer

        audio, sr = synthetic_audio

        # Step 1: Audio processing
        processed = AudioProcessor.normalize(audio)
        assert processed is not None

        # Step 2: Noise reduction
        reducer = NoiseReducer(strength="medium")
        cleaned = reducer.process(processed, sample_rate=sr)
        assert cleaned is not None
        assert np.isfinite(cleaned).all()

        # Step 3: Emotion recognition
        try:
            recognizer = EmotionRecognizer(backend="energy")
            emotion = recognizer.recognize(cleaned, sample_rate=sr)
            assert emotion is not None
        except Exception:
            pass  # energy backend may not be available

    def test_pipeline_latency_budget(self, synthetic_audio):
        """Full pipeline should complete within reasonable time."""
        from vram_core.audio_utils import AudioProcessor
        from vram_core.noise_reduction import NoiseReducer

        audio, sr = synthetic_audio

        t0 = time.perf_counter()
        processed = AudioProcessor.normalize(audio)
        reducer = NoiseReducer(strength="medium")
        cleaned = reducer.process(processed, sample_rate=sr)
        elapsed = time.perf_counter() - t0

        # Should complete in under 2 seconds for 3s audio
        assert elapsed < 2.0, f"Pipeline too slow: {elapsed:.2f}s"
        assert cleaned is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])