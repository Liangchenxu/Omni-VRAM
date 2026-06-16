"""
Chinese Tokenizer / Word Segmentation
=======================================

Optimized Chinese word segmentation for ASR post-processing.

Provides jieba-based segmentation with domain-aware dictionaries
and custom tokenization for mixed Chinese-English text.

Usage:
    from vram_core.chinese.tokenizer import ChineseTokenizer

    tokenizer = ChineseTokenizer()
    tokens = tokenizer.tokenize("使用GPU进行深度学习训练")
    # => ['使用', 'GPU', '进行', '深度学习', '训练']

    keywords = tokenizer.extract_keywords("心电图检查显示窦性心律")
    # => ['心电图', '窦性心律', '检查', '显示']
"""

import re
import logging
from typing import Optional, List, Dict, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Token:
    """A token from Chinese text segmentation."""
    word: str
    start: int       # Character offset in source text
    end: int         # Character offset end (exclusive)
    pos: str         # Part-of-speech tag (if available)
    is_chinese: bool # Whether this is a Chinese token


class ChineseTokenizer:
    """
    Chinese word segmentation optimized for ASR output.

    Features:
    - jieba-based segmentation with domain dictionaries
    - Mixed Chinese-English tokenization
    - Keyword extraction (TF-IDF based)
    - Custom dictionary support

    Args:
        use_jieba: Whether to use jieba (if available). Falls back to
                   character-based segmentation if not.
        domain_dicts: List of domain dictionary file paths to load.
    """

    # Common stop words for keyword extraction
    _STOP_WORDS = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
        '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
        '你', '会', '着', '没有', '看', '好', '自己', '这', '他', '她',
        '吗', '把', '那', '它', '被', '从', '对', '但', '以', '可以',
        '这个', '那个', '什么', '怎么', '如果', '因为', '所以', '但是',
        '然后', '而且', '或者', '虽然', '还是', '已经', '可能', '应该',
        '可以', '需要', '不是', '没有', '已经', '非常', '比较', '一些',
        '这个', '那个', '这些', '那些', '这样', '那样', '这里', '那里',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'shall',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'and', 'or', 'but', 'not', 'no', 'so', 'if', 'it', 'its',
    }

    # POS patterns for keyword importance
    _IMPORTANT_POS = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn', 'eng', 'j'}

    # Proper noun patterns (common company/org names, brand names)
    _PROPER_NOUNS = {
        'OpenAI', 'Google', 'Microsoft', 'Apple', 'Meta', 'Amazon',
        'NVIDIA', 'Intel', 'AMD', 'Tesla', 'Baidu', 'Alibaba', 'Tencent',
        'ByteDance', 'Huawei', 'Xiaomi', 'JD', 'Meituan', 'Didi',
        'ChatGPT', 'GPT-4', 'GPT-3', 'Claude', 'Gemini', 'Llama',
        'PyTorch', 'TensorFlow', 'Kubernetes', 'Docker', 'Linux',
        'Python', 'JavaScript', 'TypeScript', 'Rust', 'Go', 'Java',
        'React', 'Vue', 'Angular', 'Node.js', 'Redis', 'MongoDB',
        'PostgreSQL', 'MySQL', 'Elasticsearch', 'Kafka', 'Spark',
        'GitHub', 'GitLab', 'Slack', 'Zoom', 'Teams',
    }

    # Number unit patterns for proper tokenization
    _NUMBER_UNITS = {
        '年', '月', '日', '号', '时', '分', '秒',
        '元', '美元', '欧元', '英镑', '日元',
        '米', '公里', '千米', '厘米', '毫米',
        '公斤', '千克', '克', '吨',
        '升', '毫升',
        '度', '摄氏度', '华氏度',
        '倍', '次', '个', '只', '条', '张', '本', '台', '套',
        '万', '亿', '%',
    }

    def __init__(
        self,
        use_jieba: bool = True,
        domain_dicts: Optional[List[str]] = None,
        custom_proper_nouns: Optional[Set[str]] = None,
    ):
        self._use_jieba = use_jieba
        self._jieba = None
        self._jieba_loaded = False
        self._proper_nouns: Set[str] = set(self._PROPER_NOUNS)
        if custom_proper_nouns:
            self._proper_nouns.update(custom_proper_nouns)

        if use_jieba:
            self._try_load_jieba()

        if domain_dicts:
            for dict_path in domain_dicts:
                self.load_dict(dict_path)

    def _try_load_jieba(self):
        """Try to import and initialize jieba."""
        if self._jieba_loaded:
            return
        try:
            import jieba
            import jieba.posseg as pseg
            jieba.setLogLevel(logging.WARNING)
            self._jieba = jieba
            self._pseg = pseg
            self._jieba_loaded = True
            logger.info("jieba loaded for Chinese tokenization")
        except ImportError:
            logger.info("jieba not available, using character-based segmentation")
            self._jieba_loaded = False

    def load_dict(self, dict_path: str):
        """
        Load a jieba custom dictionary.

        Args:
            dict_path: Path to dictionary file.
        """
        if not self._jieba_loaded:
            self._try_load_jieba()
        if self._jieba:
            try:
                self._jieba.load_userdict(dict_path)
                logger.info("Loaded custom dictionary: %s", dict_path)
            except Exception as e:
                logger.warning("Failed to load dictionary %s: %s", dict_path, e)

    def add_word(self, word: str, freq: Optional[int] = None, pos: Optional[str] = None):
        """
        Add a custom word to the tokenizer.

        Args:
            word: Word to add.
            freq: Word frequency (higher = more likely to be segmented as a word).
            pos: Part-of-speech tag.
        """
        if self._jieba_loaded:
            if freq and pos:
                self._jieba.add_word(word, freq, pos)
            elif freq:
                self._jieba.add_word(word, freq)
            else:
                self._jieba.add_word(word)

    def tokenize(self, text: str) -> List[Token]:
        """
        Segment Chinese text into tokens.

        Args:
            text: Input text.

        Returns:
            List of Token objects.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        if self._jieba_loaded:
            return self._tokenize_jieba(text)
        return self._tokenize_basic(text)

    def _tokenize_jieba(self, text: str) -> List[Token]:
        """Tokenize using jieba with POS tagging."""
        tokens = []
        words = self._pseg.cut(text)
        offset = 0

        for word, pos in words:
            start = text.find(word, offset)
            if start == -1:
                start = offset
            end = start + len(word)

            is_chinese = bool(re.search(r'[\u4e00-\u9fff]', word))

            tokens.append(Token(
                word=word,
                start=start,
                end=end,
                pos=pos,
                is_chinese=is_chinese,
            ))
            offset = end

        return tokens

    def _tokenize_basic(self, text: str) -> List[Token]:
        """
        Basic tokenization without jieba.

        Splits on:
        - Chinese character sequences (grouped)
        - English/number sequences (grouped, with special handling)
        - Punctuation (individual)
        - Proper nouns (kept as single tokens)
        """
        tokens = []
        # First pass: find proper nouns and mark them
        remaining = text
        offset = 0

        # Sort proper nouns by length (longest first)
        sorted_nouns = sorted(self._proper_nouns, key=len, reverse=True)

        # Build pattern for proper nouns
        if sorted_nouns:
            noun_pattern = '|'.join(re.escape(n) for n in sorted_nouns)
            noun_regex = re.compile(f'({noun_pattern})', re.IGNORECASE)
        else:
            noun_regex = None

        # Main tokenization pattern
        pattern = re.compile(
            r'([\u4e00-\u9fff]+)'           # Chinese characters
            r'|([a-zA-Z][a-zA-Z0-9.+-]*)'   # English words (may contain digits, dots, etc.)
            r'|([0-9]+(?:\.[0-9]+)?)'        # Numbers (including decimals)
            r'|(\s+)'                         # Whitespace
            r'|(.)'                           # Other (punctuation etc.)
        )

        # First, try to identify proper nouns in the text
        proper_noun_spans = []
        if noun_regex:
            for m in noun_regex.finditer(text):
                proper_noun_spans.append((m.start(), m.end(), m.group(0)))

        # Tokenize with proper noun awareness
        used_spans: Set[Tuple[int, int]] = set()

        for pn_start, pn_end, pn_text in proper_noun_spans:
            if (pn_start, pn_end) not in used_spans:
                is_chinese = bool(re.search(r'[\u4e00-\u9fff]', pn_text))
                tokens.append(Token(
                    word=pn_text,
                    start=pn_start,
                    end=pn_end,
                    pos='nz',  # proper noun POS
                    is_chinese=is_chinese,
                ))
                used_spans.add((pn_start, pn_end))

        for match in pattern.finditer(text):
            start = match.start()
            end = match.end()

            # Skip if this span is part of a proper noun
            if any(s <= start < e for s, e, _ in proper_noun_spans):
                continue

            word = match.group(0)
            is_chinese = bool(match.group(1))
            is_number = bool(match.group(3))

            pos = 'm' if is_number else 'x'  # 'm' = numeral in Chinese POS

            tokens.append(Token(
                word=word,
                start=start,
                end=end,
                pos=pos,
                is_chinese=is_chinese,
            ))

        # Sort tokens by position
        tokens.sort(key=lambda t: t.start)
        return tokens

    def tokenize_numbers_with_units(self, text: str) -> List[str]:
        """
        Tokenize text keeping number+unit combinations as single tokens.

        Examples:
            "35000元" → ["35000元"]
            "2024年1月" → ["2024年", "1月"]
            "100公里" → ["100公里"]

        Args:
            text: Input text.

        Returns:
            List of token strings.
        """
        unit_pattern = '|'.join(re.escape(u) for u in self._NUMBER_UNITS)
        pattern = re.compile(
            r'(\d+(?:\.\d+)?(?:' + unit_pattern + r'))'  # number+unit
            r'|(\d+(?:\.\d+)?)'                           # standalone number
            r'|([\u4e00-\u9fff]+)'                        # Chinese text
            r'|([a-zA-Z][a-zA-Z0-9.+-]*)'                # English words
            r'|(.)'                                       # other
        )

        result = []
        for match in pattern.finditer(text):
            result.append(match.group(0))
        return result

    def add_proper_noun(self, noun: str):
        """
        Add a proper noun to the recognition set.

        Args:
            noun: Proper noun to add (e.g., company name, person name).
        """
        self._proper_nouns.add(noun)
        # Also add to jieba if available
        if self._jieba_loaded:
            self._jieba.add_word(noun, freq=10000, pos='nz')

    def add_proper_nouns(self, nouns: List[str]):
        """
        Add multiple proper nouns.

        Args:
            nouns: List of proper nouns.
        """
        for noun in nouns:
            self.add_proper_noun(noun)

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract keywords from Chinese text.

        Uses jieba's TF-IDF algorithm if available, otherwise
        uses simple frequency-based extraction.

        Args:
            text: Input text.
            top_k: Number of keywords to return.

        Returns:
            List of keyword strings, sorted by importance.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        if self._jieba_loaded:
            try:
                import jieba.analyse
                keywords = jieba.analyse.extract_tags(
                    text, topK=top_k, withWeight=False
                )
                return keywords
            except Exception as e:
                logger.warning("jieba keyword extraction failed: %s", e)

        return self._extract_keywords_basic(text, top_k)

    def _extract_keywords_basic(self, text: str, top_k: int) -> List[str]:
        """Basic keyword extraction by frequency and POS filtering."""
        tokens = self.tokenize(text)

        # Filter: keep only meaningful tokens
        candidates = []
        for token in tokens:
            word = token.word.strip()
            if not word or len(word) < 2:
                continue
            if word.lower() in self._STOP_WORDS:
                continue
            if token.pos in self._IMPORTANT_POS or (not token.is_chinese and len(word) >= 2):
                candidates.append(word)

        # Count frequencies
        freq: Dict[str, int] = {}
        for word in candidates:
            freq[word] = freq.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_k]]

    def segment_for_display(self, text: str, separator: str = ' ') -> str:
        """
        Segment text and join with separators (for display/visualization).

        Args:
            text: Input text.
            separator: Separator between tokens.

        Returns:
            Segmented text string.
        """
        tokens = self.tokenize(text)
        words = [t.word for t in tokens if t.word.strip()]
        return separator.join(words)

    def get_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Extract character n-grams from text.

        Args:
            text: Input text.
            n: N-gram size.

        Returns:
            List of n-gram strings.
        """
        # Remove spaces and non-Chinese chars for n-gram extraction
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        if len(chinese_chars) < n:
            return []

        ngrams = []
        for i in range(len(chinese_chars) - n + 1):
            ngrams.append(''.join(chinese_chars[i:i + n]))
        return ngrams


def segment_chinese(text: str) -> List[str]:
    """
    Convenience function for quick Chinese segmentation.

    Args:
        text: Chinese text.

    Returns:
        List of word strings.
    """
    tokenizer = ChineseTokenizer()
    return [t.word for t in tokenizer.tokenize(text) if t.word.strip()]