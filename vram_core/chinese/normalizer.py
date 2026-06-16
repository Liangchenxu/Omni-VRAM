"""
Chinese Text Normalizer
========================

Normalizes Chinese ASR output by converting:
- Spoken numbers → digits (一百二十三 → 123)
- Spoken dates → date format (二零二四年一月一日 → 2024年1月1日)
- Spoken units → standard units (三千米 → 3km)
- Spoken money → currency format (五十块钱 → 50元)
- Full-width → half-width characters

Usage:
    from vram_core.chinese.normalizer import TextNormalizer

    normalizer = TextNormalizer()
    result = normalizer.normalize("我有一百二十三块钱")
    # => "我有123元"
"""

import re
import logging
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Normalizes Chinese text from ASR output.

    Handles number, date, time, currency, and unit normalization.

    Args:
        normalize_numbers: Convert spoken numbers to digits.
        normalize_dates: Convert spoken dates to date format.
        normalize_units: Convert spoken units to standard format.
        normalize_currency: Convert spoken money to currency format.
        fullwidth_to_halfwidth: Convert full-width to half-width chars.
    """

    # Chinese digit mapping
    _DIGITS = {
        '零': 0, '〇': 0, '一': 1, '壹': 1,
        '二': 2, '两': 2, '贰': 2, '貳': 2,
        '三': 3, '叁': 3, '參': 3,
        '四': 4, '肆': 4,
        '五': 5, '伍': 5,
        '六': 6, '陆': 6, '陸': 6,
        '七': 7, '柒': 7,
        '八': 8, '捌': 8,
        '九': 9, '玖': 9,
        '十': 10, '拾': 10,
        '百': 100, '佰': 100,
        '千': 1000, '仟': 1000,
        '万': 10000, '萬': 10000,
        '亿': 100000000, '億': 100000000,
    }

    # Number words that appear in Whisper output
    _NUMBER_WORDS = {
        '十一': '11', '十二': '12', '十三': '13', '十四': '14',
        '十五': '15', '十六': '16', '十七': '17', '十八': '18',
        '十九': '19', '二十': '20', '三十': '30', '四十': '40',
        '五十': '50', '六十': '60', '七十': '70', '八十': '80',
        '九十': '90', '一百': '100', '一千': '1000', '一万': '10000',
    }

    # Unit mappings
    _UNITS = {
        '米': 'm', '千米': 'km', '公里': 'km', '厘米': 'cm',
        '毫米': 'mm', '英里': 'mi', '英尺': 'ft', '英寸': 'in',
        '公斤': 'kg', '千克': 'kg', '克': 'g', '毫克': 'mg',
        '吨': 't', '磅': 'lb',
        '升': 'L', '毫升': 'mL',
        '度': '°', '摄氏度': '°C', '华氏度': '°F',
        '瓦': 'W', '千瓦': 'kW',
        '赫兹': 'Hz', '千赫': 'kHz', '兆赫': 'MHz',
        '字节': 'B', '千字节': 'KB', '兆字节': 'MB',
        '吉字节': 'GB', '太字节': 'TB',
        '秒': 's', '分钟': 'min', '小时': 'h',
        '天': 'd', '周': 'w', '月': '月', '年': '年',
        '百分号': '%',
    }

    # Currency patterns
    _CURRENCY = {
        '块钱': '元', '块': '元', '元': '元',
        '角': '角', '毛': '角',
        '分': '分',
        '美元': '美元', '美金': '美元',
        '欧元': '欧元', '英镑': '英镑',
        '日元': '日元', '韩元': '韩元',
    }

    # Large amount units (万/亿)
    _AMOUNT_UNITS = {
        '万': 10000,
        '萬': 10000,
        '亿': 100000000,
        '億': 100000000,
    }

    def __init__(
        self,
        normalize_numbers: bool = True,
        normalize_dates: bool = True,
        normalize_units: bool = True,
        normalize_currency: bool = True,
        normalize_phone: bool = True,
        normalize_id_number: bool = True,
        normalize_amounts: bool = True,
        fullwidth_to_halfwidth: bool = True,
    ):
        self.normalize_numbers = normalize_numbers
        self.normalize_dates = normalize_dates
        self.normalize_units = normalize_units
        self.normalize_currency = normalize_currency
        self.normalize_phone = normalize_phone
        self.normalize_id_number = normalize_id_number
        self.normalize_amounts = normalize_amounts
        self.fullwidth_to_halfwidth = fullwidth_to_halfwidth
        self._cache: dict = {}  # Simple LRU-style cache

    def normalize(self, text: str) -> str:
        """
        Normalize Chinese text.

        Args:
            text: Input text from ASR.

        Returns:
            Normalized text.
        """
        if not text or not text.strip():
            return text

        text = text.strip()

        # Check cache for repeated strings
        cache_key = text
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.fullwidth_to_halfwidth:
            text = self._fw_to_hw(text)

        if self.normalize_numbers:
            text = self._normalize_numbers(text)

        if self.normalize_dates:
            text = self._normalize_dates(text)

        if self.normalize_currency:
            text = self._normalize_currency(text)

        if self.normalize_units:
            text = self._normalize_units(text)

        if self.normalize_phone:
            text = self._normalize_phone_numbers(text)

        if self.normalize_id_number:
            text = self._normalize_id_numbers(text)

        if self.normalize_amounts:
            text = self._normalize_amounts(text)

        # Cache result (keep cache bounded)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[cache_key] = text

        return text

    def clear_cache(self):
        """Clear normalization cache."""
        self._cache.clear()

    def _fw_to_hw(self, text: str) -> str:
        """Convert full-width characters to half-width."""
        result = []
        for char in text:
            code = ord(char)
            # Full-width ASCII chars: 0xFF01-0xFF5E → 0x0021-0x007E
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            # Full-width space
            elif code == 0x3000:
                result.append(' ')
            else:
                result.append(char)
        return ''.join(result)

    def _normalize_numbers(self, text: str) -> str:
        """
        Convert Chinese number words to digits.

        Examples:
            一百二十三 → 123
            三千五百 → 3500
            二点五 → 2.5
        """
        # Pattern: Chinese number sequences
        # Match sequences of Chinese number characters
        number_chars = set(self._DIGITS.keys()) | {'点', '两'}

        result = []
        i = 0
        while i < len(text):
            # Try to match a Chinese number sequence
            if text[i] in number_chars:
                num_str = ''
                while i < len(text) and text[i] in number_chars:
                    num_str += text[i]
                    i += 1
                # Try to convert
                converted = self._chinese_to_number(num_str)
                if converted is not None:
                    result.append(str(converted))
                else:
                    result.append(num_str)
            else:
                result.append(text[i])
                i += 1

        return ''.join(result)

    def _chinese_to_number(self, text: str) -> Optional[float]:
        """
        Convert a Chinese number string to a number.

        Args:
            text: Chinese number text (e.g., "一百二十三").

        Returns:
            Number as float, or None if conversion fails.
        """
        if not text:
            return None

        # Handle decimal point
        if '点' in text:
            parts = text.split('点', 1)
            integer_part = self._chinese_integer_to_int(parts[0]) if parts[0] else 0
            if integer_part is None:
                return None
            # Decimal part: each digit is read individually
            decimal_str = ''
            for ch in parts[1]:
                if ch in self._DIGITS:
                    decimal_str += str(self._DIGITS[ch])
                else:
                    break
            if decimal_str:
                return float(f"{integer_part}.{decimal_str}")
            return float(integer_part)

        result = self._chinese_integer_to_int(text)
        return float(result) if result is not None else None

    def _chinese_integer_to_int(self, text: str) -> Optional[int]:
        """Convert Chinese integer text to int."""
        if not text:
            return None

        # Special case: just a single unit like "十" means 10
        if text == '十':
            return 10
        if text == '百':
            return 100
        if text == '千':
            return 1000
        if text == '万':
            return 10000

        total = 0
        current = 0
        wan_part = 0  # For 万 (ten-thousand) section

        for ch in text:
            if ch not in self._DIGITS:
                return None

            val = self._DIGITS[ch]

            if ch in ('万', '萬'):
                wan_part = (current + (wan_part if wan_part else 0)) * 10000
                current = 0
            elif ch in ('亿', '億'):
                # Not commonly needed but handle it
                total += (wan_part + current) * 100000000
                wan_part = 0
                current = 0
            elif val >= 10:
                if current == 0:
                    current = 1  # "十" at start means 10, "百" means 100, etc.
                current *= val
            else:
                current = val

        return total + wan_part + current

    def _normalize_dates(self, text: str) -> str:
        """
        Normalize spoken dates.

        Examples:
            二零二四年一月一日 → 2024年1月1日
            二零二四年十二月三十一号 → 2024年12月31号
        """
        # Date pattern: YYYY年M月D日/号
        # First normalize the numbers in date context
        date_pattern = re.compile(
            r'([零〇一二两三四五六七八十壹贰叁肆伍陆柒捌玖拾百]+)年'
            r'([零〇一二两三四五六七八十壹贰叁肆伍陆柒捌玖拾百]+)月'
            r'([零〇一二两三四五六七八十壹贰叁肆伍陆柒捌玖拾百]+)[日号]'
        )

        def replace_date(match):
            year_str = match.group(1)
            month_str = match.group(2)
            day_str = match.group(3)

            year = self._date_digits_to_str(year_str)
            month = self._chinese_to_number(month_str)
            day = self._chinese_to_number(day_str)

            if year and month and day:
                return f"{year}年{int(month)}月{int(day)}日"
            return match.group(0)

        text = date_pattern.sub(replace_date, text)

        # Also handle time: 下午三点十五分
        time_pattern = re.compile(
            r'([上下]午)?([零〇一二两三四五六七八九十]+)点'
            r'([零〇一二两三四五六七八九十]+)?分?'
            r'([零〇一二两三四五六七八九十]+)?秒?'
        )

        def replace_time(match):
            period = match.group(1) or ''
            hour_str = match.group(2)
            minute_str = match.group(3) or ''
            second_str = match.group(4) or ''

            hour = self._chinese_to_number(hour_str)
            result = f"{period}{int(hour)}点"
            if minute_str:
                minute = self._chinese_to_number(minute_str)
                if minute:
                    result += f"{int(minute)}分"
            if second_str:
                second = self._chinese_to_number(second_str)
                if second:
                    result += f"{int(second)}秒"
            return result

        text = time_pattern.sub(replace_time, text)

        return text

    def _date_digits_to_str(self, chinese_digits: str) -> Optional[str]:
        """Convert year digits: 二零二四 → '2024'."""
        digit_map = {
            '零': '0', '〇': '0',
            '一': '1', '二': '2', '两': '2',
            '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '九': '9',
        }
        result = ''
        for ch in chinese_digits:
            if ch in digit_map:
                result += digit_map[ch]
            else:
                return None
        return result if result else None

    def _normalize_currency(self, text: str) -> str:
        """
        Normalize spoken currency.

        Examples:
            五十块钱 → 50元
            三百二十美元 → 320美元
        """
        # Pattern: number + currency word
        currency_words = '|'.join(re.escape(k) for k in self._CURRENCY.keys())
        pattern = re.compile(
            r'([零〇一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億点]+)'
            r'(' + currency_words + r')'
        )

        def replace_currency(match):
            num_str = match.group(1)
            unit = match.group(2)
            num = self._chinese_to_number(num_str)
            if num is not None:
                replacement = self._CURRENCY[unit]
                if num == int(num):
                    return f"{int(num)}{replacement}"
                return f"{num}{replacement}"
            return match.group(0)

        return pattern.sub(replace_currency, text)

    def _normalize_units(self, text: str) -> str:
        """
        Normalize spoken units.

        Examples:
            三千米 → 3km
            五十公斤 → 50kg
        """
        # Sort by length (longest first) to match multi-char units first
        sorted_units = sorted(self._UNITS.keys(), key=len, reverse=True)
        unit_words = '|'.join(re.escape(k) for k in sorted_units)
        pattern = re.compile(
            r'([零〇一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億点]+)'
            r'(' + unit_words + r')'
        )

        def replace_unit(match):
            num_str = match.group(1)
            unit = match.group(2)
            num = self._chinese_to_number(num_str)
            if num is not None:
                replacement = self._UNITS[unit]
                if num == int(num):
                    return f"{int(num)}{replacement}"
                return f"{num}{replacement}"
            return match.group(0)

        return pattern.sub(replace_unit, text)


    def _normalize_phone_numbers(self, text: str) -> str:
        """
        Normalize spoken phone numbers.

        Handles:
        - Chinese spoken digits → phone number format (一三八零零一二三四五六 → 13800123456)
        - Common phone number patterns (11-digit mobile, 8-digit landline)
        - Spoken with 四位 区号 patterns

        Examples:
            一三八零零一二三四五六 → 13800123456
            零一零八八八八九九九九 → 010-88889999
        """
        # Pattern: sequences of spoken single digits that form phone numbers
        # First, convert sequences of single Chinese digits (一三八零零一二三四五六)
        digit_map = {
            '零': '0', '〇': '0', '一': '1', '壹': '1',
            '二': '2', '两': '2', '贰': '2',
            '三': '3', '叁': '3',
            '四': '4', '肆': '4',
            '五': '5', '伍': '5',
            '六': '6', '陆': '6',
            '七': '7', '柒': '7',
            '八': '8', '捌': '8',
            '九': '9', '玖': '9',
        }

        # Match sequences of single-digit Chinese characters (11 digits = mobile, 7-8 digits = landline)
        digit_chars = set(digit_map.keys())

        result = []
        i = 0
        while i < len(text):
            if text[i] in digit_chars:
                # Collect consecutive single-digit characters
                digit_str = ''
                j = i
                while j < len(text) and text[j] in digit_chars:
                    digit_str += digit_map[text[j]]
                    j += 1

                # Check if this looks like a phone number
                if len(digit_str) == 11 and digit_str[0] == '1':
                    # Mobile number: 1xx-xxxx-xxxx
                    result.append(f"{digit_str[:3]}-{digit_str[3:7]}-{digit_str[7:]}")
                    i = j
                elif 10 <= len(digit_str) <= 12 and digit_str[0] == '0':
                    # Landline with area code: 0xx-xxxx-xxxx
                    area_len = 3 if digit_str[1:3] in ('10',) else 4
                    if area_len == 3 and len(digit_str) >= 10:
                        result.append(f"{digit_str[:3]}-{digit_str[3:]}")
                    elif area_len == 4 and len(digit_str) >= 11:
                        result.append(f"{digit_str[:4]}-{digit_str[4:]}")
                    else:
                        result.append(digit_str)
                    i = j
                elif 7 <= len(digit_str) <= 8:
                    # Landline without area code
                    result.append(digit_str)
                    i = j
                else:
                    # Not a phone number, keep original
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1

        return ''.join(result)

    def _normalize_id_numbers(self, text: str) -> str:
        """
        Normalize spoken ID card numbers (身份证号码).

        Handles 18-digit Chinese ID numbers spoken digit by digit.
        The spoken form uses single Chinese digits for each position.

        Examples:
            一二三四五六七八九零一二三四五六七八X → 123456789012345678X
        """
        digit_map = {
            '零': '0', '〇': '0', '一': '1', '壹': '1',
            '二': '2', '两': '2', '贰': '2',
            '三': '3', '叁': '3',
            '四': '4', '肆': '4',
            '五': '5', '伍': '5',
            '六': '6', '陆': '6',
            '七': '7', '柒': '7',
            '八': '8', '捌': '8',
            '九': '9', '玖': '9',
        }

        # Context clues for ID numbers
        id_contexts = ['身份证', '身份号', '证件号', 'ID', 'id号', '身份证号']

        result = []
        i = 0
        while i < len(text):
            # Check if there's an ID number context
            found_context = False
            for ctx in id_contexts:
                if text[i:i + len(ctx)] == ctx:
                    found_context = True
                    result.append(ctx)
                    i += len(ctx)
                    # Skip optional 是/为/冒号
                    while i < len(text) and text[i] in '是为：:号码':
                        result.append(text[i])
                        i += 1
                    # Now collect the ID number digits
                    id_digits = ''
                    j = i
                    while j < len(text) and (text[j] in digit_map or text[j] in 'xX'):
                        if text[j] in 'xX':
                            id_digits += 'X'
                        else:
                            id_digits += digit_map[text[j]]
                        j += 1

                    if len(id_digits) == 18:
                        # Format: 6-8-3-1 (area-birthday-seq-check)
                        result.append(f"{id_digits[:6]}-{id_digits[6:14]}-{id_digits[14:17]}-{id_digits[17]}")
                        i = j
                    else:
                        result.append(id_digits)
                        i = j
                    break

            if not found_context:
                result.append(text[i])
                i += 1

        return ''.join(result)

    def _normalize_amounts(self, text: str) -> str:
        """
        Normalize spoken large amounts with 万/亿.

        Handles:
        - 三万五千 → 35000
        - 两点五亿 → 2.5亿
        - 一百二十万 → 1200000

        Examples:
            三点五万 → 3.5万
            两亿三千万 → 2.3亿
            一百二十万 → 120万
        """
        # Pattern: number + 万/亿
        amount_units = '|'.join(re.escape(k) for k in self._AMOUNT_UNITS.keys())
        pattern = re.compile(
            r'([零〇一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億点]+)'
            r'(' + amount_units + r')'
        )

        def replace_amount(match):
            num_str = match.group(1)
            unit = match.group(2)
            num = self._chinese_to_number(num_str)
            if num is not None:
                unit_value = self._AMOUNT_UNITS[unit]
                # If the number already includes the unit multiplier, divide it out
                # e.g., "三万五千" → chinese_to_number returns 35000, unit is 万(10000)
                # We want to display "3.5万"
                if unit_value <= abs(num) and unit_value > 1:
                    # The number already includes the unit, format as X万/X亿
                    actual = num / unit_value
                    if actual == int(actual):
                        return f"{int(actual)}{unit}"
                    return f"{actual:.1f}{unit}".rstrip('0').rstrip('.')
                else:
                    # Simple case: number * unit
                    total = num * unit_value
                    if total == int(total):
                        return f"{int(total)}"
                    return f"{total}"
            return match.group(0)

        return pattern.sub(replace_amount, text)


def normalize_chinese_text(text: str) -> str:
    """
    Convenience function for quick text normalization.

    Args:
        text: Chinese ASR text.

    Returns:
        Normalized text.
    """
    normalizer = TextNormalizer()
    return normalizer.normalize(text)