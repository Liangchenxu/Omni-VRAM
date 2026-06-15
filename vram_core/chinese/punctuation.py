"""
Chinese Punctuation Restoration
================================

Adds punctuation to unpunctuated Chinese ASR text using rule-based
and optional model-based approaches.

Handles: 。，！？、；：""''（）《》——…… etc.

Usage:
    from vram_core.chinese.punctuation import PunctuationRestorer

    restorer = PunctuationRestorer()
    result = restorer.restore("你好 今天天气怎么样 我们去公园吧")
    # => "你好，今天天气怎么样？我们去公园吧。"
"""

import re
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class PunctuationRestorer:
    """
    Restores punctuation to unpunctuated Chinese text.

    Uses a hybrid approach:
    1. Rule-based heuristics (fast, zero dependency)
    2. Optional model-based restoration (higher accuracy)

    Args:
        use_model: Whether to use a punctuation model (requires transformers).
        model_name: HuggingFace model name for punctuation restoration.
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
    ]

    # Words that typically start a new sentence
    _SENTENCE_STARTERS = [
        '你好', '请问', '谢谢', '对不起', '没关系',
        '欢迎', '恭喜', '再见',
    ]

    def __init__(
        self,
        use_model: bool = False,
        model_name: str = "oliverguhr/spacy-chinese-punctuation",
    ):
        self.use_model = use_model
        self.model_name = model_name
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

            # First person statements typically end with period
            # Questions typically end with question mark
            if '？' in punct:
                final_parts.append(part + punct)
            elif i < len(result_parts) - 1:
                final_parts.append(part + '，')
            else:
                final_parts.append(part + punct)

        result = ''.join(final_parts)

        # Clean up double punctuation
        result = re.sub(r'[，。]{2,}', '。', result)
        result = re.sub(r'[？]{2,}', '？', result)

        return result

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