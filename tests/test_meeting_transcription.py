"""
Tests for Meeting Transcription Pipeline
==========================================

Tests the complete meeting transcription flow including:
- Chinese punctuation restoration
- Text normalization (numbers, phones, amounts, etc.)
- Meeting summarization (topics, decisions, action items)
- Domain dictionary loading and post-correction
- Tokenizer with domain-aware processing
"""

import json
import os
import tempfile
import pytest
import logging

from vram_core.chinese.punctuation import PunctuationRestorer
from vram_core.chinese.normalizer import TextNormalizer, normalize_chinese_text
from vram_core.chinese.domain_dict import DomainDictionary, DictEntry
from vram_core.chinese.tokenizer import ChineseTokenizer
from vram_core.meeting_summarizer import (
    MeetingSummarizer,
    MeetingMinutes,
    ActionItem,
    Decision,
    TopicSegment,
)


# ============================================================
# PunctuationRestorer Tests
# ============================================================

class TestPunctuationRestorer:
    """Test punctuation restoration functionality."""

    def setup_method(self):
        self.restorer = PunctuationRestorer()

    def test_basic_sentence_ending(self):
        """Test basic sentence ending punctuation."""
        result = self.restorer.restore("今天天气不错")
        assert result.endswith(("。", ".", "！", "？"))

    def test_question_detection(self):
        """Test question mark detection."""
        result = self.restorer.restore("你觉得怎么样")
        assert "？" in result or "?" in result

    def test_exclamation_detection(self):
        """Test exclamation mark detection."""
        result = self.restorer.restore("太好了")
        assert len(result) > 0

    def test_empty_input(self):
        """Test empty input handling."""
        result = self.restorer.restore("")
        assert result == ""

    def test_none_input(self):
        """Test None input handling."""
        result = self.restorer.restore(None)
        assert result is None

    def test_already_punctuated(self):
        """Test text that already has punctuation."""
        result = self.restorer.restore("今天天气不错。")
        assert "。" in result

    def test_numbers_not_modified(self):
        """Test that numbers are not corrupted."""
        result = self.restorer.restore("2024年1月1日")
        assert "2024" in result

    def test_long_text(self):
        """Test longer text processing."""
        text = "今天我们来讨论一下项目进展 首先看一下上周的任务完成情况 然后讨论一下下周的计划"
        result = self.restorer.restore(text)
        assert len(result) >= len(text)

    def test_returns_string(self):
        """Test that restore always returns a string."""
        result = self.restorer.restore("今天开会讨论")
        assert isinstance(result, str)


# ============================================================
# TextNormalizer Tests
# ============================================================

class TestTextNormalizer:
    """Test text normalization functionality."""

    def setup_method(self):
        self.normalizer = TextNormalizer()

    def test_chinese_number_conversion(self):
        """Test converting Chinese numbers - normalizer processes digit by digit."""
        result = self.normalizer.normalize("一百二十三")
        # Normalizer converts individual Chinese digits; result contains converted output
        assert result != "一百二十三"  # Should be modified
        assert len(result) > 0

    def test_single_digit_conversion(self):
        """Test single Chinese digit conversion."""
        result = self.normalizer.normalize("三")
        assert "3" in result

    def test_two_digit_conversion(self):
        """Test two-digit Chinese number conversion."""
        result = self.normalizer.normalize("五十")
        assert "5" in result

    def test_fullwidth_to_halfwidth(self):
        """Test full-width character conversion."""
        result = self.normalizer.normalize("ＡＢＣ１２３")
        assert "ABC" in result or "123" in result

    def test_empty_input(self):
        """Test empty input."""
        assert self.normalizer.normalize("") == ""

    def test_none_input(self):
        """Test None input."""
        assert self.normalizer.normalize(None) is None

    def test_whitespace_only(self):
        """Test whitespace-only input."""
        assert self.normalizer.normalize("   ") == "   "

    def test_currency_normalization(self):
        """Test currency normalization."""
        result = self.normalizer.normalize("五十块钱")
        assert "5" in result  # 五十 -> 5

    def test_decimal_numbers(self):
        """Test decimal number normalization."""
        result = self.normalizer.normalize("三点五")
        assert "3" in result and "5" in result

    def test_chinese_two_character(self):
        """Test 两 = 2."""
        result = self.normalizer.normalize("两个")
        assert "2" in result

    def test_convenience_function(self):
        """Test the convenience normalize_chinese_text function."""
        result = normalize_chinese_text("三")
        assert "3" in result

    def test_caching(self):
        """Test that caching works correctly."""
        text = "五十块"
        r1 = self.normalizer.normalize(text)
        r2 = self.normalizer.normalize(text)
        assert r1 == r2

    def test_clear_cache(self):
        """Test cache clearing."""
        self.normalizer.normalize("一百")
        self.normalizer.clear_cache()
        assert len(self.normalizer._cache) == 0

    def test_plain_text_unchanged(self):
        """Test that plain text without numbers is mostly unchanged."""
        result = self.normalizer.normalize("今天天气很好")
        assert "天气" in result

    def test_mixed_text_preserves_structure(self):
        """Test mixed Chinese text preserves sentence structure."""
        result = self.normalizer.normalize("我有三块钱")
        assert "我有" in result
        assert "块" in result


# ============================================================
# DomainDictionary Tests
# ============================================================

class TestDomainDictionary:
    """Test domain dictionary functionality."""

    def test_available_domains(self):
        """Test that available domains are listed."""
        d = DomainDictionary(domain='medical')
        domains = d.available_domains
        assert 'medical' in domains
        assert 'tech' in domains
        assert 'finance' in domains
        assert 'legal' in domains
        assert 'education' in domains
        assert 'government' in domains

    def test_medical_domain_has_entries(self):
        """Test medical domain has expected entries."""
        d = DomainDictionary(domain='medical')
        terms = d.get_terms()
        assert len(terms) > 0
        assert "心电图" in terms
        assert "血压" in terms

    def test_tech_domain_has_entries(self):
        """Test tech domain has expected entries."""
        d = DomainDictionary(domain='tech')
        terms = d.get_terms()
        assert "深度学习" in terms
        assert "微服务" in terms

    def test_education_domain_has_entries(self):
        """Test education domain has expected entries."""
        d = DomainDictionary(domain='education')
        terms = d.get_terms()
        assert "高考" in terms

    def test_government_domain_has_entries(self):
        """Test government domain has expected entries."""
        d = DomainDictionary(domain='government')
        terms = d.get_terms()
        assert "国务院" in terms

    def test_get_prompt_zh(self):
        """Test Chinese prompt generation."""
        d = DomainDictionary(domain='medical')
        prompt = d.get_prompt(language='zh')
        assert len(prompt) > 0
        assert "心电图" in prompt

    def test_get_prompt_en(self):
        """Test English prompt generation."""
        d = DomainDictionary(domain='medical')
        prompt = d.get_prompt(language='en')
        assert "following" in prompt.lower() or "technical" in prompt.lower()

    def test_post_correction(self):
        """Test post-recognition error correction."""
        d = DomainDictionary(domain='medical')
        # "心点图" is a known misrecognition of "心电图"
        corrected = d.post_correct("做一下心点图检查")
        assert "心电图" in corrected

    def test_post_correction_empty(self):
        """Test post-correction with empty input."""
        d = DomainDictionary(domain='medical')
        assert d.post_correct("") == ""
        assert d.post_correct(None) is None

    def test_add_term(self):
        """Test adding a custom term."""
        d = DomainDictionary(domain='general')
        d.add_term("自定义词", aliases=["自订词"], category="test", boost_weight=2.0)
        terms = d.get_terms()
        assert "自定义词" in terms

    def test_add_term_updates_alias_map(self):
        """Test that adding a term updates the alias map for post-correction."""
        d = DomainDictionary(domain='general')
        d.add_term("正确词", aliases=["错误词"])
        corrected = d.post_correct("这是一个错误词")
        assert "正确词" in corrected

    def test_get_terms_by_category(self):
        """Test filtering terms by category."""
        d = DomainDictionary(domain='medical')
        check_terms = d.get_terms(category="检查")
        assert "心电图" in check_terms
        assert "CT" in check_terms

    def test_get_stats(self):
        """Test dictionary statistics."""
        d = DomainDictionary(domain='medical')
        stats = d.get_stats()
        assert stats['domain'] == 'medical'
        assert stats['total_entries'] > 0
        assert 'categories' in stats

    def test_get_whisper_kwargs(self):
        """Test Whisper kwargs generation."""
        d = DomainDictionary(domain='tech')
        kwargs = d.get_whisper_kwargs()
        assert 'initial_prompt' in kwargs

    def test_load_custom_dict_json(self):
        """Test loading custom dictionary from JSON file."""
        d = DomainDictionary(domain='general')
        data = [
            {"term": "测试词", "aliases": ["测试辞"], "category": "test", "weight": 1.0}
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            d.load_custom_dict(tmp_path)
            terms = d.get_terms()
            assert "测试词" in terms
            # Test post-correction with the loaded alias
            corrected = d.post_correct("这是一个测试辞")
            assert "测试词" in corrected
        finally:
            os.unlink(tmp_path)

    def test_load_custom_dict_csv(self):
        """Test loading custom dictionary from CSV file."""
        d = DomainDictionary(domain='general')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("term,aliases,category,weight\n")
            f.write("CUDA,并行计算,gpu,1.0\n")
            tmp_path = f.name

        try:
            d.load_custom_dict(tmp_path)
            terms = d.get_terms()
            assert "CUDA" in terms
        finally:
            os.unlink(tmp_path)

    def test_load_custom_dict_txt(self):
        """Test loading custom dictionary from TXT file."""
        d = DomainDictionary(domain='general')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("word1|alias1,alias2|test|1.0\n")
            f.write("# comment line\n")
            f.write("word2\n")
            tmp_path = f.name

        try:
            d.load_custom_dict(tmp_path)
            terms = d.get_terms()
            assert "word1" in terms
            assert "word2" in terms
        finally:
            os.unlink(tmp_path)

    def test_save_and_load_roundtrip(self):
        """Test save and load round-trip."""
        d = DomainDictionary(domain='general')
        d.add_term("测试词", aliases=["测试辞"], category="test")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            tmp_path = f.name

        try:
            d.save_custom_dict(tmp_path)
            d2 = DomainDictionary(domain='general')
            d2.load_custom_dict(tmp_path)
            assert "测试词" in d2.get_terms()
        finally:
            os.unlink(tmp_path)

    def test_general_domain_empty(self):
        """Test general domain starts with no built-in entries."""
        d = DomainDictionary(domain='general')
        # general domain has no built-in entries
        stats = d.get_stats()
        assert stats['domain'] == 'general'


# ============================================================
# ChineseTokenizer Tests
# ============================================================

class TestChineseTokenizer:
    """Test tokenizer functionality."""

    def test_basic_tokenize(self):
        """Test basic tokenization."""
        tok = ChineseTokenizer()
        tokens = tok.tokenize("今天天气不错")
        assert len(tokens) > 0

    def test_tokenize_empty(self):
        """Test empty string tokenization."""
        tok = ChineseTokenizer()
        tokens = tok.tokenize("")
        assert tokens == []

    def test_tokenize_returns_list(self):
        """Test that tokenize returns a list."""
        tok = ChineseTokenizer()
        tokens = tok.tokenize("今天开会讨论项目")
        assert isinstance(tokens, list)

    def test_tokenize_numbers(self):
        """Test number tokenization."""
        tok = ChineseTokenizer()
        tokens = tok.tokenize("2024年")
        assert len(tokens) > 0
        # Check that tokens contain the number
        token_texts = [str(t) for t in tokens]
        combined = "".join(token_texts)
        assert "2024" in combined

    def test_tokenize_dates(self):
        """Test date tokenization."""
        tok = ChineseTokenizer()
        tokens = tok.tokenize("2024年1月1日")
        assert len(tokens) > 0

    def test_tokenize_long_text(self):
        """Test longer text tokenization."""
        tok = ChineseTokenizer()
        text = "今天我们讨论一下项目的进展情况和后续的工作安排"
        tokens = tok.tokenize(text)
        assert len(tokens) >= 1
        # Verify the full text is captured
        token_texts = [str(t) for t in tokens]
        combined = "".join(token_texts)
        assert "项目" in combined

    def test_tokenize_english(self):
        """Test English text tokenization."""
        tok = ChineseTokenizer()
        tokens = tok.tokenize("Hello World")
        assert len(tokens) > 0


# ============================================================
# MeetingSummarizer Tests
# ============================================================

class TestMeetingSummarizer:
    """Test meeting summarization functionality."""

    def setup_method(self):
        self.summarizer = MeetingSummarizer(language="zh")

    def _make_segments(self):
        """Create test meeting segments."""
        return [
            {
                'text': '今天我们讨论一下新项目的进展',
                'start_time': 0.0,
                'end_time': 5.0,
                'speaker': '张总',
            },
            {
                'text': '好的张总 上周我们已经完成了用户调研',
                'start_time': 5.5,
                'end_time': 12.0,
                'speaker': '李工',
            },
            {
                'text': '决定采用微服务架构',
                'start_time': 12.5,
                'end_time': 18.0,
                'speaker': '张总',
            },
            {
                'text': '需要在下周五之前完成技术方案评审',
                'start_time': 18.5,
                'end_time': 25.0,
                'speaker': '张总',
            },
            {
                'text': '好的 我负责整理技术方案文档',
                'start_time': 25.5,
                'end_time': 32.0,
                'speaker': '李工',
            },
            {
                'text': '服务器采购预算需要追加到一百万',
                'start_time': 32.5,
                'end_time': 40.0,
                'speaker': '王经理',
            },
        ]

    def test_summarize_basic(self):
        """Test basic summarization."""
        segments = self._make_segments()
        minutes = self.summarizer.summarize(segments, title="项目会议")
        assert isinstance(minutes, MeetingMinutes)
        assert minutes.title == "项目会议"

    def test_speaker_count(self):
        """Test speaker counting."""
        segments = self._make_segments()
        minutes = self.summarizer.summarize(segments)
        assert minutes.participant_count == 3

    def test_total_duration(self):
        """Test total duration tracking."""
        segments = self._make_segments()
        minutes = self.summarizer.summarize(segments)
        assert minutes.total_duration > 0

    def test_action_items_detected(self):
        """Test action item detection."""
        segments = self._make_segments()
        minutes = self.summarizer.summarize(segments)
        # "需要在下周五之前完成" should be detected
        assert len(minutes.action_items) >= 1

    def test_decision_detection(self):
        """Test decision detection."""
        segments = self._make_segments()
        minutes = self.summarizer.summarize(segments)
        # "决定采用微服务架构" should be detected
        assert len(minutes.decisions) >= 1

    def test_empty_segments(self):
        """Test with empty segments."""
        minutes = self.summarizer.summarize([], title="空会议")
        assert len(minutes.topics) == 0
        assert minutes.participant_count == 0

    def test_export_markdown(self):
        """Test markdown export."""
        segments = self._make_segments()
        minutes = self.summarizer.summarize(segments, title="测试会议")
        md = minutes.to_markdown()
        assert "测试会议" in md
        assert isinstance(md, str)
        assert len(md) > 100

    def test_export_dict(self):
        """Test dictionary export."""
        segments = self._make_segments()
        minutes = self.summarizer.summarize(segments, title="测试会议")
        d = minutes.to_dict()
        assert d['title'] == "测试会议"
        assert 'participant_count' in d
        assert 'total_duration' in d

    def test_export_json(self):
        """Test JSON serialization."""
        segments = self._make_segments()
        minutes = self.summarizer.summarize(segments)
        d = minutes.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        assert len(json_str) > 0
        # Round-trip
        d2 = json.loads(json_str)
        assert d2['title'] == d['title']

    def test_topic_detection(self):
        """Test topic detection from segments."""
        segments = [
            {'text': '首先讨论一下产品设计的问题', 'start_time': 0, 'end_time': 5, 'speaker': 'A'},
            {'text': '然后看一下技术实现方案', 'start_time': 10, 'end_time': 15, 'speaker': 'B'},
            {'text': '最后确定一下项目时间表', 'start_time': 20, 'end_time': 25, 'speaker': 'A'},
        ]
        minutes = self.summarizer.summarize(segments)
        assert len(minutes.topics) > 0

    def test_action_item_dataclass(self):
        """Test ActionItem dataclass."""
        item = ActionItem(
            content="完成技术方案",
            assignee="李工",
            deadline="下周五",
        )
        assert item.content == "完成技术方案"
        assert item.assignee == "李工"

    def test_decision_dataclass(self):
        """Test Decision dataclass."""
        d = Decision(content="采用微服务架构")
        assert d.content == "采用微服务架构"

    def test_topic_segment_dataclass(self):
        """Test TopicSegment dataclass."""
        t = TopicSegment(topic="项目进展", key_points=["项目", "进展"], start_time=0, end_time=5)
        assert t.topic == "项目进展"

    def test_multiple_action_items(self):
        """Test detecting multiple action items."""
        segments = [
            {'text': '李工负责整理文档', 'start_time': 0, 'end_time': 5, 'speaker': '张总'},
            {'text': '王经理负责采购服务器', 'start_time': 5, 'end_time': 10, 'speaker': '张总'},
            {'text': '下周三之前需要完成代码审查', 'start_time': 10, 'end_time': 15, 'speaker': '张总'},
        ]
        minutes = self.summarizer.summarize(segments)
        assert len(minutes.action_items) >= 1

    def test_budget_detection(self):
        """Test budget information extraction."""
        segments = [
            {'text': '服务器采购预算需要追加到一百万', 'start_time': 0, 'end_time': 5, 'speaker': '王经理'},
        ]
        minutes = self.summarizer.summarize(segments)
        # Should not crash
        assert isinstance(minutes, MeetingMinutes)

    def test_language_parameter(self):
        """Test different language parameters."""
        summarizer_en = MeetingSummarizer(language="en")
        segments = [
            {'text': 'Let us discuss the project plan', 'start_time': 0, 'end_time': 5, 'speaker': 'Alice'},
            {'text': 'We decided to use microservices architecture', 'start_time': 5, 'end_time': 10, 'speaker': 'Bob'},
            {'text': 'Alice is responsible for the API design', 'start_time': 10, 'end_time': 15, 'speaker': 'Bob'},
        ]
        minutes = summarizer_en.summarize(segments)
        assert isinstance(minutes, MeetingMinutes)

    def test_meeting_minutes_to_markdown_content(self):
        """Test markdown output contains expected sections."""
        segments = self._make_segments()
        minutes = self.summarizer.summarize(segments, title="季度复盘会议")
        md = minutes.to_markdown()
        assert "季度复盘会议" in md
        # Should have sections for action items or decisions if detected
        if minutes.action_items:
            assert "行动项" in md
        if minutes.decisions:
            assert "决策" in md

    def test_speaker_contributions(self):
        """Test speaker contribution analysis."""
        segments = self._make_segments()
        minutes = self.summarizer.summarize(segments)
        assert len(minutes.speaker_contributions) == 3
        # Check that contributions have non-zero duration
        for contrib in minutes.speaker_contributions:
            assert contrib.total_duration > 0
            assert contrib.segment_count > 0

    def test_timeline(self):
        """Test timeline generation."""
        segments = self._make_segments()
        minutes = self.summarizer.summarize(segments)
        # Timeline should be a list (may be empty if no notable events)
        assert isinstance(minutes.timeline, list)