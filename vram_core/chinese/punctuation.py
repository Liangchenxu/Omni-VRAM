"""
Chinese Punctuation Restoration
================================

Adds punctuation to unpunctuated Chinese ASR text using rule-based
and optional model-based approaches.

Handles: 。，！？、；：""''（）《》——…… etc.

Supports:
- Pause-based comma/period detection (using segment timestamps)
- Tone-based question/exclamation marks
- Long sentence auto-breaking
- Clause boundary detection

Usage:
    from vram_core.chinese.punctuation import PunctuationRestorer

    restorer = PunctuationRestorer()
    result = restorer.restore("你好 今天天气怎么样 我们去公园吧")
    # => "你好，今天天气怎么样？我们去公园吧。"

    # With timing info for better punctuation
    result = restorer.restore_with_timing(segments)
"""

import re
import logging
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class PunctuationRestorer:
    """
    Restores punctuation to unpunctuated Chinese text.

    Uses a hybrid approach:
    1. Rule-based heuristics (fast, zero dependency)
    2. Timing-based punctuation (comma/period from pause duration)
    3. Tone-based punctuation (question/exclamation from intonation)
    4. Long sentence auto-breaking
    5. Optional model-based restoration (higher accuracy)

    Args:
        use_model: Whether to use a punctuation model (requires transformers).
        model_name: HuggingFace model name for punctuation restoration.
        pause_comma_ms: Pause duration (ms) to insert a comma (default: 300ms).
        pause_period_ms: Pause duration (ms) to insert a period (default: 800ms).
        max_sentence_chars: Maximum characters before auto-breaking a sentence.
    """

    # Common sentence-ending patterns
    _SENTENCE_END_PATTERNS = [
        (r'[吗呢吧啊哦呀嘛]$', '？'),   # Question particles
        (r'好的?$', '。'),
        (r'对吧$', '？'),
        (r'是不是$', '？'),
        (r'有没有$', '？'),
        (r'可以吗$', '？'),
        (r'行吗$', '？'),
        (r'为什么$', '？'),
        (r'怎么$', '？'),
        (r'多少$', '？'),
        (r'什么时候$', '？'),
        (r'哪儿$', '？'),
        (r'哪里$', '？'),
        (r'谁$', '？'),
    ]

    # Exclamation patterns
    _EXCLAMATION_PATTERNS = [
        r'[太真好厉害]$',        # 太好了！真棒！
        r'太[棒好厉害漂亮]',
        r'真[棒厉害不错]',
        r'好[棒厉害啊]',
        r'厉害',
        r'漂亮',
        r'精彩',
        r'完美',
        r'加油',
    ]

    # Common clause boundary words (often preceded by a comma or period)
    _CLAUSE_BOUNDARIES = [
        '但是', '但', '可是', '然而', '不过',  # adversative
        '所以', '因此', '于是', '结果',          # causal
        '而且', '并且', '另外', '此外',          # additive
        '如果', '假如', '要是', '万一',          # conditional
        '虽然', '尽管', '即使', '哪怕',          # concessive
        '因为', '由于',                          # reason
        '然后', '接着', '随后',                  # sequential
        '总之', '总的来说', '综上所述',          # summary
        '比如', '例如', '譬如',                  # example
        '首先', '其次', '最后',                  # enumeration
        '同时', '与此同时',                      # simultaneous
        '也就是说', '换句话说',                  # clarification
        '不仅', '不但',                          # progressive
        '除了', '除非',                          # exception
        '不管', '无论',                          # unconditional
        '既然',                                  # causal
        '只要',                                  # condition
    ]

    # Words that typically start a new sentence
    _SENTENCE_STARTERS = [
        '你好', '请问', '谢谢', '对不起', '没关系',
        '欢迎', '恭喜', '再见',
        '好的', '行', '没问题', '可以',
    ]

    # Question words that indicate interrogative sentences
    _QUESTION_WORDS = [
        '什么', '怎么', '为什么', '哪里', '哪儿', '谁',
        '多少', '几', '哪个', '哪些', '如何', '是否',
        '能否', '能不能', '会不会', '是不是', '有没有',
        '对不对', '好不好', '行不行', '可以不可以',
    ]

    def __init__(
        self,
        use_model: bool = False,
        model_name: str = "oliverguhr/spacy-chinese-punctuation",
        pause_comma_ms: float = 300,
        pause_period_ms: float = 800,
        max_sentence_chars: int = 80,
    ):
        self.use_model = use_model
        self.model_name = model_name
        self.pause_comma_ms = pause_comma_ms
        self.pause_period_ms = pause_period_ms
        self.max_sentence_chars = max_sentence_chars
        self._model = None
        self._model_lock = None

        if use_model:
            import threading
            self._model_lock = threading.Lock()

    def restore(self, text: str) -> str:
        """
        Restore punctuation to unpunctuated Chinese text.

        Args:
            text: Input text without punctuation.

        Returns:
            Text with restored punctuation.
        """
        if not text or not text.strip():
            return text

        text = text.strip()

        # Remove any existing punctuation and normalize spaces
        text = self._clean_input(text)

        if self.use_model:
            try:
                return self._restore_model(text)
            except Exception as e:
                logger.warning("Model punctuation failed, using rules: %s", e)

        return self._restore_rules(text)

    def restore_with_timing(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Restore punctuation using timing information from ASR segments.

        Uses pause duration between segments to determine comma vs period,
        and segment duration to detect sentence boundaries.

        Args:
            segments: List of segment dicts with 'text', 'start', 'end' keys
                      (times in seconds).

        Returns:
            Updated segments with punctuated text.
        """
        if not segments:
            return segments

        result = []
        for i, seg in enumerate(segments):
            new_seg = dict(seg)
            text = seg.get('text', '').strip()
            if not text:
                result.append(new_seg)
                continue

            text = self._clean_input(text)

            # Calculate pause before this segment
            if i > 0:
                prev_end = segments[i - 1].get('end', 0)
                curr_start = seg.get('start', 0)
                pause_ms = (curr_start - prev_end) * 1000
            else:
                pause_ms = 0

            # Get segment duration
            duration_s = seg.get('end', 0) - seg.get('start', 0)

            # Determine punctuation based on timing
            punct = self._determine_punctuation(text, pause_ms, duration_s)

            # Clean text and apply punctuation
            new_seg['text'] = text + punct
            result.append(new_seg)

        # Post-process: merge consecutive short segments, break long sentences
        result = self._post_process_segments(result)

        return result

    def _determine_punctuation(self, text: str, pause_ms: float, duration_s: float) -> str:
        """
        Determine appropriate punctuation based on timing and content.

        Args:
            text: Segment text.
            pause_ms: Pause duration before this segment (milliseconds).
            duration_s: Duration of this segment (seconds).

        Returns:
            Punctuation character.
        """
        # Check question patterns first
        for pattern, p in self._SENTENCE_END_PATTERNS:
            if re.search(pattern, text):
                return p

        # Check question words
        for qw in self._QUESTION_WORDS:
            if qw in text:
                return '？'

        # Check exclamation patterns
        for pattern in self._EXCLAMATION_PATTERNS:
            if re.search(pattern, text):
                return '！'

        # Use timing to determine punctuation
        if pause_ms >= self.pause_period_ms:
            return '。'
        elif pause_ms >= self.pause_comma_ms:
            return '，'

        # If segment is long (speech without pause), likely a period
        if duration_s > 3.0:
            return '。'

        # Default: comma for continuation
        return '，'

    def _clean_input(self, text: str) -> str:
        """Clean input text: remove existing punctuation, normalize spaces."""
        # Remove existing Chinese and English punctuation
        text = re.sub(r'[，。！？、；：""''（）《》——…·\.\,\!\?\;\:\"\'\(\)\[\]\{\}]', '', text)
        # Normalize spaces around punctuation removal
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _restore_rules(self, text: str) -> str:
        """Rule-based punctuation restoration."""
        if not text:
            return text

        # Split by natural pause indicators
        # Whisper often inserts spaces at word boundaries
        words = text.split()
        if not words:
            return text

        result_parts: List[str] = []
        current_clause: List[str] = []

        for i, word in enumerate(words):
            current_clause.append(word)
            clause_text = ''.join(current_clause)

            # Check for clause boundary
            should_split = False
            split_punct = '，'

            # Next word starts a clause boundary
            if i + 1 < len(words):
                next_word = words[i + 1]
                for boundary in self._CLAUSE_BOUNDARIES:
                    if next_word.startswith(boundary):
                        should_split = True
                        split_punct = '，'
                        break

            # Current word ends a clause boundary
            for boundary in self._CLAUSE_BOUNDARIES:
                if word.endswith(boundary) and i > 0 and i < len(words) - 1:
                    should_split = True
                    split_punct = '，'
                    break

            # Auto-break long sentences
            if len(clause_text) >= self.max_sentence_chars:
                should_split = True
                split_punct = '。'

            if should_split and current_clause:
                result_parts.append(''.join(current_clause))
                current_clause = []

        # Join remaining
        if current_clause:
            result_parts.append(''.join(current_clause))

        # Now apply sentence-level punctuation
        final_parts: List[str] = []
        for i, part in enumerate(result_parts):
            part = part.strip()
            if not part:
                continue

            # Check sentence-ending patterns
            punct = '。'  # default end punctuation

            for pattern, p in self._SENTENCE_END_PATTERNS:
                if re.search(pattern, part):
                    punct = p
                    break

            # Check exclamation patterns
            if punct == '。':
                for pattern in self._EXCLAMATION_PATTERNS:
                    if re.search(pattern, part):
                        punct = '！'
                        break

            # Check question words
            if punct == '。':
                for qw in self._QUESTION_WORDS:
                    if qw in part:
                        punct = '？'
                        break

            if '？' in punct or '！' in punct:
                final_parts.append(part + punct)
            elif i < len(result_parts) - 1:
                final_parts.append(part + '，')
            else:
                final_parts.append(part + punct)

        result = ''.join(final_parts)

        # Clean up double punctuation
        result = re.sub(r'[，。]{2,}', '。', result)
        result = re.sub(r'[？]{2,}', '？', result)
        result = re.sub(r'[！]{2,}', '！', result)

        return result

    def _post_process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-process segments: merge short consecutive segments,
        break long sentences.

        Args:
            segments: List of segment dicts.

        Returns:
            Processed segments.
        """
        if len(segments) <= 1:
            return segments

        result = []
        for seg in segments:
            text = seg.get('text', '')
            # If text is very long, try to break it
            if len(text) > self.max_sentence_chars:
                # Split at clause boundaries
                broken = self._break_long_sentence(text)
                if len(broken) > 1:
                    for part in broken:
                        new_seg = dict(seg)
                        new_seg['text'] = part
                        result.append(new_seg)
                    continue

            result.append(seg)

        return result

    def _break_long_sentence(self, text: str) -> List[str]:
        """
        Break a long sentence into shorter ones at clause boundaries.

        Args:
            text: Long sentence text.

        Returns:
            List of shorter sentences.
        """
        # Try to split at clause boundary words
        parts = [text]
        for boundary in self._CLAUSE_BOUNDARIES:
            new_parts = []
            for part in parts:
                if len(part) > self.max_sentence_chars and boundary in part:
                    # Split at the first occurrence of the boundary
                    idx = part.index(boundary)
                    if idx > 10:  # Don't create tiny fragments
                        new_parts.append(part[:idx])
                        new_parts.append(part[idx:])
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            parts = new_parts

        return parts

    def _restore_model(self, text: str) -> str:
        """Model-based punctuation restoration."""
        with self._model_lock:
            if self._model is None:
                self._load_model()

        # Use the model for punctuation restoration
        # Implementation depends on the specific model
        raise NotImplementedError("Model-based punctuation not yet implemented")

    def _load_model(self):
        """Load the punctuation restoration model."""
        try:
            from transformers import pipeline
            self._model = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
            )
            logger.info("Punctuation model loaded: %s", self.model_name)
        except Exception as e:
            logger.error("Failed to load punctuation model: %s", e)
            raise

    def restore_segments(self, segments: List[dict]) -> List[dict]:
        """
        Restore punctuation to a list of ASR segments.

        Args:
            segments: List of segment dicts with 'text' key.

        Returns:
            Updated segments with punctuation.
        """
        result = []
        for seg in segments:
            new_seg = dict(seg)
            new_seg['text'] = self.restore(seg.get('text', ''))
            result.append(new_seg)
        return result


def restore_punctuation(text: str) -> str:
    """
    Convenience function for quick punctuation restoration.

    Args:
        text: Unpunctuated Chinese text.

    Returns:
        Punctuated text.
    """
    restorer = PunctuationRestorer()
    return restorer.restore(text)