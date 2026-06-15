"""
Chinese Language Processing Module
====================================

Provides Chinese-specific NLP utilities for ASR post-processing:

- punctuation: Punctuation restoration for unpunctuated text
- normalizer: Number, date, and unit normalization
- dialect: Dialect detection and mapping
- domain_dict: Domain-specific dictionary and hotword boosting
- tokenizer: Optimized Chinese word segmentation
"""

from vram_core.chinese.punctuation import PunctuationRestorer, restore_punctuation
from vram_core.chinese.normalizer import TextNormalizer, normalize_chinese_text
from vram_core.chinese.dialect import DialectDetector, DialectInfo
from vram_core.chinese.domain_dict import DomainDictionary, DictEntry, get_domain_prompt
from vram_core.chinese.tokenizer import ChineseTokenizer, Token, segment_chinese

__all__ = [
    "PunctuationRestorer",
    "restore_punctuation",
    "TextNormalizer",
    "normalize_chinese_text",
    "DialectDetector",
    "DialectInfo",
    "DomainDictionary",
    "DictEntry",
    "get_domain_prompt",
    "ChineseTokenizer",
    "Token",
    "segment_chinese",
]
