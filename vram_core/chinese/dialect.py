"""
Chinese Dialect Detection
==========================

Detects and maps Chinese dialect features in ASR output to improve
recognition accuracy. Primarily targets Mandarin with awareness of
Cantonese, Wu, Min, Hakka dialects.

Usage:
    from vram_core.chinese.dialect import DialectDetector

    detector = DialectDetector()
    info = detector.detect("今日天气好靓啊")
    # => DialectInfo(dialect='cantonese', confidence=0.8, ...)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set

logger = logging.getLogger(__name__)


@dataclass
class DialectInfo:
    """Detection result for a dialect."""
    dialect: str  # 'mandarin', 'cantonese', 'wu', 'min', 'hakka', 'mixed'
    confidence: float  # 0.0 - 1.0
    features_found: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class DialectDetector:
    """
    Detects Chinese dialect features in text.

    This is a rule-based detector that looks for dialect-specific
    vocabulary, particles, and grammatical patterns.

    Useful for:
    - Choosing the right Whisper language code (zh vs yue)
    - Applying dialect-specific post-processing
    - Providing user feedback on recognition quality
    """

    # Dialect-specific vocabulary
    _CANTONESE_FEATURES = {
        '嘅': 'possessive particle',
        '啲': 'plural/diminutive',
        '咁': 'so/such',
        '喺': 'at/in',
        '嚟': 'come',
        '咗': 'completed aspect',
        '緊': 'progressive aspect',
        '冇': 'not have',
        '唔': 'not',
        '佢': 'he/she',
        '我哋': 'we',
        '你哋': 'you (pl)',
        '佢哋': 'they',
        '靓': 'beautiful',
        '嘢': 'thing',
        '食': 'eat',
        '饮': 'drink',
        '返工': 'go to work',
        '返学': 'go to school',
        '乜嘢': 'what',
        '点解': 'why',
        '几时': 'when',
        '边度': 'where',
        '今日': 'today',
        '听日': 'tomorrow',
        '琴日': 'yesterday',
        '好嘢': 'great',
        '唔该': 'please/thanks',
        '多谢': 'thanks',
        '早晨': 'good morning',
        '拜拜': 'bye bye',
    }

    _WU_FEATURES = {
        '侬': 'you',
        '阿拉': 'we',
        '啥': 'what',
        '勿': 'don\'t',
        '勿要': 'don\'t',
        '辰光': 'time',
        '物事': 'thing',
        '老卵': 'awesome',
        '嗲': 'good',
        '戆': 'stupid',
        '弄堂': 'alley',
        '阿拉': 'we/us',
        '噶': 'particle',
        '嘞': 'particle',
        '咾': 'particle',
    }

    _MIN_FEATURES = {
        '汝': 'you',
        '伊': 'he/she',
        '阮': 'we',
        '恁': 'you (pl)',
        '厝': 'house',
        '厝边': 'neighbor',
        '歹势': 'sorry',
        '多谢': 'thanks',
        '好势': 'good',
        '水': 'beautiful',
        '冻': 'cold',
        '烧': 'hot',
        '呷': 'eat',
        '啉': 'drink',
    }

    _HAKKA_FEATURES = {
        '涯': 'I',
        '你': 'you',
        '佢': 'he/she',
        '涯兜人': 'we',
        '毋': 'not',
        '系': 'is',
        '食': 'eat',
        '行': 'walk',
        '靓': 'beautiful',
        '细人仔': 'children',
        '阿婆': 'grandmother',
        '阿公': 'grandfather',
    }

    # Dialect-specific particles (sentence-final)
    _CANTONESE_PARTICLES = {'啊', '呀', '嘅', '咩', '喎', '嘞', '啦', '啫', '㗎', '咋', '呢', '嘛'}
    _WU_PARTICLES = {'嘞', '咾', '噶', '伐', '啦', '呀'}

    # Whisper language code mapping
    WHISPER_LANG_CODES = {
        'mandarin': 'zh',
        'cantonese': 'yue',
        'wu': 'zh',  # No separate Whisper code
        'min': 'zh',
        'hakka': 'zh',
        'mixed': 'zh',
    }

    def __init__(self):
        pass

    def detect(self, text: str) -> DialectInfo:
        """
        Detect the dialect of the input text.

        Args:
            text: Chinese text (from ASR output).

        Returns:
            DialectInfo with detected dialect and confidence.
        """
        if not text or not text.strip():
            return DialectInfo(dialect='mandarin', confidence=0.5)

        text = text.strip()
        scores: Dict[str, float] = {
            'mandarin': 0.0,
            'cantonese': 0.0,
            'wu': 0.0,
            'min': 0.0,
            'hakka': 0.0,
        }
        features: Dict[str, List[str]] = {d: [] for d in scores}

        # Check each dialect
        for word, desc in self._CANTONESE_FEATURES.items():
            if word in text:
                scores['cantonese'] += 1.0
                features['cantonese'].append(f"{word}({desc})")

        for word, desc in self._WU_FEATURES.items():
            if word in text:
                scores['wu'] += 1.0
                features['wu'].append(f"{word}({desc})")

        for word, desc in self._MIN_FEATURES.items():
            if word in text:
                scores['min'] += 1.0
                features['min'].append(f"{word}({desc})")

        for word, desc in self._HAKKA_FEATURES.items():
            if word in text:
                scores['hakka'] += 1.0
                features['hakka'].append(f"{word}({desc})")

        # Check sentence-final particles
        if text:
            last_char = text[-1]
            if last_char in self._CANTONESE_PARTICLES:
                scores['cantonese'] += 0.5
                features['cantonese'].append(f"particle:{last_char}")
            if last_char in self._WU_PARTICLES:
                scores['wu'] += 0.5
                features['wu'].append(f"particle:{last_char}")

        # Mandarin gets a base score (default)
        scores['mandarin'] = 0.5

        # Find winner
        max_score = max(scores.values())
        if max_score <= 0.5:
            # No strong dialect signal → Mandarin
            return DialectInfo(
                dialect='mandarin',
                confidence=0.7,
                features_found=[],
                suggestions=[],
            )

        # Find the dialect with highest score
        best_dialect = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best_dialect] / total if total > 0 else 0.5
        confidence = min(confidence, 1.0)

        # Generate suggestions
        suggestions = self._get_suggestions(best_dialect)

        return DialectInfo(
            dialect=best_dialect,
            confidence=confidence,
            features_found=features[best_dialect],
            suggestions=suggestions,
        )

    def get_whisper_lang_code(self, dialect: str) -> str:
        """
        Get the appropriate Whisper language code for a dialect.

        Args:
            dialect: Detected dialect name.

        Returns:
            Whisper language code (e.g., 'zh', 'yue').
        """
        return self.WHISPER_LANG_CODES.get(dialect, 'zh')

    def _get_suggestions(self, dialect: str) -> List[str]:
        """Get suggestions based on detected dialect."""
        suggestions = {
            'cantonese': [
                "检测到粤语特征，建议使用 Whisper language='yue' 以获得更好的识别效果",
                "粤语文本建议使用繁体中文显示",
            ],
            'wu': [
                "检测到吴语特征，建议使用 Whisper language='zh'",
                "吴语识别准确率可能较低，建议使用更大的模型",
            ],
            'min': [
                "检测到闽语特征，建议使用 Whisper language='zh'",
                "闽语有多种变体，识别效果可能有限",
            ],
            'hakka': [
                "检测到客家话特征，建议使用 Whisper language='zh'",
            ],
        }
        return suggestions.get(dialect, [])

    def map_to_whisper_lang(self, text: str) -> str:
        """
        Detect dialect and return the appropriate Whisper language code.

        Args:
            text: Chinese text.

        Returns:
            Whisper language code.
        """
        info = self.detect(text)
        return self.get_whisper_lang_code(info.dialect)