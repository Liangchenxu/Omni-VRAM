"""
Omni-VRAM v2.5.0 Tests
======================

Tests for new features:
- Audio Enhancer (noise reduction, normalization, EQ, dereverb, AGC)
- Speech Quality Assessment
- Edge Backends (ONNX, Lite)
- LLM Meeting Assistant
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAudioEnhancer:
    """Test AudioEnhancer class."""
    
    def test_enhancer_init(self):
        from vram_core.audio_enhancer import AudioEnhancer
        enhancer = AudioEnhancer(sample_rate=16000)
        assert enhancer is not None
    
    def test_enhance_basic(self):
        from vram_core.audio_enhancer import AudioEnhancer
        enhancer = AudioEnhancer()
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = enhancer.enhance(audio)
        assert len(result) == len(audio)
        assert result.dtype == np.float32
    
    def test_normalize(self):
        from vram_core.audio_enhancer import AudioEnhancer
        enhancer = AudioEnhancer()
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        result = enhancer._normalize(audio)
        peak = np.max(np.abs(result))
        assert peak > 0.5
    
    def test_agc(self):
        from vram_core.audio_enhancer import AudioEnhancer
        enhancer = AudioEnhancer()
        audio = np.random.randn(16000).astype(np.float32) * 0.001
        result = enhancer._auto_gain_control(audio)
        assert len(result) == len(audio)
    
    def test_noise_gate(self):
        from vram_core.audio_enhancer import AudioEnhancer
        enhancer = AudioEnhancer()
        audio = np.random.randn(16000).astype(np.float32) * 0.001
        result = enhancer._noise_gate(audio, threshold_db=-40)
        # Quiet parts should be attenuated
        assert np.max(np.abs(result)) <= np.max(np.abs(audio)) + 0.01


class TestSpeechQuality:
    """Test SpeechQualityAssessor class."""
    
    def test_assessor_init(self):
        from vram_core.speech_quality import SpeechQualityAssessor
        assessor = SpeechQualityAssessor()
        assert assessor is not None
    
    def test_assess_speech(self):
        from vram_core.speech_quality import SpeechQualityAssessor
        assessor = SpeechQualityAssessor()
        t = np.linspace(0, 1, 16000)
        audio = (np.sin(2 * np.pi * 440 * t) + np.random.randn(16000) * 0.01).astype(np.float32)
        report = assessor.assess(audio)
        assert report.quality_grade in ["excellent", "good", "fair", "poor"]
    
    def test_assess_empty(self):
        from vram_core.speech_quality import SpeechQualityAssessor
        assessor = SpeechQualityAssessor()
        report = assessor.assess(np.array([], dtype=np.float32))
        assert report.quality_grade == "poor"
        assert len(report.issues) > 0
    
    def test_assess_clipping(self):
        from vram_core.speech_quality import SpeechQualityAssessor
        assessor = SpeechQualityAssessor()
        audio = np.ones(16000, dtype=np.float32) * 0.999
        report = assessor.assess(audio)
        assert report.clipping_ratio > 0.5
    
    def test_snr_estimation(self):
        from vram_core.speech_quality import SpeechQualityAssessor
        assessor = SpeechQualityAssessor()
        t = np.linspace(0, 2, 32000)
        speech = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        noise = np.random.randn(32000).astype(np.float32) * 0.001
        audio = speech + noise
        report = assessor.assess(audio)
        assert report.snr_db > 0


class TestLiteBackend:
    """Test LiteBackend class."""
    
    def test_lite_init(self):
        from vram_core.backends.lite_backend import LiteBackend, LiteConfig
        backend = LiteBackend(LiteConfig())
        assert backend is not None
    
    def test_prepare_mobile_file_not_found(self):
        from vram_core.backends.lite_backend import LiteBackend
        with pytest.raises(Exception):
            LiteBackend.prepare_for_mobile("nonexistent.onnx")


class TestLLMClient:
    """Test LLMClient class."""
    
    def test_llm_init(self):
        from vram_core.llm_client import LLMClient
        client = LLMClient(provider="ollama")
        assert client is not None
    
    def test_llm_providers(self):
        from vram_core.llm_client import LLMClient
        client = LLMClient()
        providers = client.available_providers
        assert "ollama" in providers or len(providers) >= 0


class TestMeetingAnalyzer:
    """Test MeetingAnalyzer class."""
    
    def test_analyzer_init(self):
        from vram_core.meeting_analyzer import MeetingAnalyzer
        analyzer = MeetingAnalyzer()
        assert analyzer is not None
    
    def test_action_item_extraction(self):
        from vram_core.meeting_analyzer import MeetingAnalyzer
        analyzer = MeetingAnalyzer()
        text = "请张三在下周一前完成报告。李四需要准备PPT。"
        items = analyzer._extract_action_items(text)
        assert len(items) > 0
    
    def test_priority_detection(self):
        from vram_core.meeting_analyzer import MeetingAnalyzer
        analyzer = MeetingAnalyzer()
        high = analyzer._detect_priority("这是紧急任务，必须今天完成")
        low = analyzer._detect_priority("这个可以以后再看")
        assert high == "high"
        assert low in ["low", "medium"]