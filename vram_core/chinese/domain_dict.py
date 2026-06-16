"""
Chinese Domain Dictionary
==========================

Provides domain-specific vocabulary and hotword boosting for improving
ASR accuracy in specialized fields (medical, legal, tech, finance, etc.).

Uses Whisper's hotword/prompt mechanism to bias recognition towards
domain-specific terms.

Usage:
    from vram_core.chinese.domain_dict import DomainDictionary

    domain = DomainDictionary(domain="medical")
    prompt = domain.get_prompt()
    # => "以下词汇可能出现：心电图、血压、CT、核磁共振..."

    corrected = domain.post_correct("心点图检查显示正常")
    # => "心电图检查显示正常"
"""

import re
import logging
from typing import Optional, List, Dict, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DictEntry:
    """A domain dictionary entry."""
    term: str           # Standard form
    aliases: List[str]  # Common misrecognitions
    category: str       # Sub-category within domain
    boost_weight: float = 1.0  # Priority weight for hotword boosting


class DomainDictionary:
    """
    Domain-specific vocabulary for Chinese ASR improvement.

    Provides:
    1. Hotword lists for Whisper prompt engineering
    2. Post-recognition error correction
    3. Custom dictionary support

    Built-in domains:
    - medical: 医疗
    - legal: 法律
    - tech: 科技
    - finance: 金融
    - education: 教育
    - general: 通用
    """

    # Medical domain
    _MEDICAL: List[DictEntry] = [
        DictEntry("心电图", ["心点图", "新电图", "心电徒"], "检查"),
        DictEntry("血压", ["血鸭", "学压"], "体征"),
        DictEntry("CT", ["西提", "sei踢", "西梯"], "检查"),
        DictEntry("核磁共振", ["核词共振", "和磁共振", "核兹共振"], "检查"),
        DictEntry("B超", ["必超", "笔超"], "检查"),
        DictEntry("心率", ["新率"], "体征"),
        DictEntry("血糖", ["学糖"], "体征"),
        DictEntry("胆固醇", ["胆固醇", "但固醇"], "化验"),
        DictEntry("甘油三酯", ["甘有三酯", "干油三酯"], "化验"),
        DictEntry("白细胞", ["白西包", "白细泡"], "化验"),
        DictEntry("红细胞", ["红西包", "红细泡"], "化验"),
        DictEntry("血小板", ["学小板"], "化验"),
        DictEntry("甲状腺", ["甲壮腺"], "内分泌"),
        DictEntry("胰岛素", ["夷岛素", "胰到素"], "内分泌"),
        DictEntry("抗生素", ["抗生束"], "药物"),
        DictEntry("阿莫西林", ["阿莫西临"], "药物"),
        DictEntry("布洛芬", ["布落分", "布罗芬"], "药物"),
        DictEntry("头孢", ["头饱", "透孢"], "药物"),
        DictEntry("静脉注射", ["静卖注射"], "治疗"),
        DictEntry("手术", ["收术"], "治疗"),
        DictEntry("麻醉", ["麻最", "马醉"], "治疗"),
        DictEntry("病理", ["病利"], "诊断"),
        DictEntry("恶性肿瘤", ["恶性种瘤"], "诊断"),
        DictEntry("良性肿瘤", ["良性种瘤"], "诊断"),
        DictEntry("幽门螺杆菌", ["幽门罗杆菌", "优门螺杆菌"], "检验"),
    ]

    # Tech domain
    _TECH: List[DictEntry] = [
        DictEntry("API", ["诶派哎", "API"], "接口"),
        DictEntry("GPU", ["鸡屁优", "GPU"], "硬件"),
        DictEntry("CPU", ["西屁优", "CPU"], "硬件"),
        DictEntry("深度学习", ["深度学席", "深度雪习"], "AI"),
        DictEntry("神经网络", ["神经网落", "生经网络"], "AI"),
        DictEntry("机器学习", ["机器学席"], "AI"),
        DictEntry("大模型", ["大魔型", "打模型"], "AI"),
        DictEntry("微调", ["微条", "维调"], "AI"),
        DictEntry("推理", ["推力"], "AI"),
        DictEntry("向量数据库", ["向量数具库"], "数据库"),
        DictEntry("容器", ["容易"], "云原生"),
        DictEntry("微服务", ["微福物", "微服物"], "架构"),
        DictEntry("负载均衡", ["在副均衡", "负在均衡"], "网络"),
        DictEntry("带宽", ["带快"], "网络"),
        DictEntry("延迟", ["沿迟"], "性能"),
        DictEntry("吞吐量", ["吞土量"], "性能"),
        DictEntry("缓存", ["换存"], "存储"),
        DictEntry("消息队列", ["消息对列"], "中间件"),
        DictEntry("中间件", ["中间见"], "中间件"),
        DictEntry("分布式", ["分步式"], "架构"),
        DictEntry("高可用", ["搞可用"], "架构"),
        DictEntry("Kubernetes", ["酷伯奈替斯", "k八s"], "云原生"),
        DictEntry("Docker", ["道客", "到克"], "云原生"),
        DictEntry("数据库", ["数具库", "数距库"], "存储"),
        DictEntry("MySQL", ["买色扣", "my sequel"], "数据库"),
        DictEntry("Redis", ["瑞迪斯", "热迪斯"], "数据库"),
        DictEntry("Nginx", ["恩静克斯", "engine x"], "Web"),
        DictEntry("前端", ["前短"], "开发"),
        DictEntry("后端", ["后短"], "开发"),
        DictEntry("算法", ["算发"], "计算机科学"),
        DictEntry("编译器", ["编义器", "变异器"], "工具"),
    ]

    # Finance domain
    _FINANCE: List[DictEntry] = [
        DictEntry("收益率", ["受益率", "收亿率"], "投资"),
        DictEntry("市盈率", ["是盈率", "市赢率"], "股票"),
        DictEntry("市净率", ["是净率"], "股票"),
        DictEntry("涨停", ["张停"], "股票"),
        DictEntry("跌停", ["碟停", "叠停"], "股票"),
        DictEntry("基金", ["机金"], "投资"),
        DictEntry("债券", ["寨卷", "债卷"], "投资"),
        DictEntry("期货", ["欺货"], "衍生品"),
        DictEntry("期权", ["弃权"], "衍生品"),
        DictEntry("杠杆", ["杠干", "刚杆"], "风险"),
        DictEntry("风控", ["风空"], "风险"),
        DictEntry("合规", ["和归"], "监管"),
        DictEntry("监管", ["坚管"], "监管"),
        DictEntry("资管", ["资官"], "管理"),
        DictEntry("信托", ["新拖"], "金融"),
        DictEntry("融资租赁", ["融资租恁"], "金融"),
        DictEntry("不良资产", ["不凉资产"], "银行"),
        DictEntry("拨备", ["波备", "播备"], "银行"),
        DictEntry("资本充足率", ["资本充组率"], "银行"),
        DictEntry("流动性", ["留动性"], "风险"),
    ]

    # Legal domain
    _LEGAL: List[DictEntry] = [
        DictEntry("诉讼", ["速颂", "素讼"], "程序"),
        DictEntry("仲裁", ["中财", "重裁"], "程序"),
        DictEntry("被告", ["倍告"], "当事人"),
        DictEntry("原告", ["元告"], "当事人"),
        DictEntry("上诉", ["上束", "商诉"], "程序"),
        DictEntry("判决", ["判绝", "盼决"], "裁判"),
        DictEntry("调解", ["条解"], "程序"),
        DictEntry("合同", ["和同"], "文书"),
        DictEntry("违约", ["威约", "维约"], "责任"),
        DictEntry("赔偿", ["陪偿", "培偿"], "责任"),
        DictEntry("知识产权", ["只是产权", "知识产全"], "权利"),
        DictEntry("专利", ["转利"], "知识产权"),
        DictEntry("商标", ["伤标"], "知识产权"),
        DictEntry("著作权", ["做着权", "著做权"], "知识产权"),
        DictEntry("法人", ["发人"], "主体"),
        DictEntry("有限责任", ["有限责认"], "公司"),
        DictEntry("刑事责任", ["行是责任"], "刑法"),
        DictEntry("民事责任", ["民是责任"], "民法"),
        DictEntry("行政处罚", ["行政出发"], "行政"),
        DictEntry("强制执行", ["强制执形"], "执行"),
        DictEntry("不动产", ["不动颤"], "物权"),
        DictEntry("抵押权", ["低押权"], "担保"),
    ]

    # Education domain
    _EDUCATION: List[DictEntry] = [
        DictEntry("高考", ["高靠"], "考试"),
        DictEntry("考研", ["考验"], "考试"),
        DictEntry("研究生", ["研究身"], "学历"),
        DictEntry("博士", ["搏士"], "学历"),
        DictEntry("硕士", ["硕是"], "学历"),
        DictEntry("学士", ["学是"], "学历"),
        DictEntry("毕业论文", ["毕业轮文"], "学术"),
        DictEntry("答辩", ["达辩"], "学术"),
        DictEntry("选课", ["选可"], "教务"),
        DictEntry("学分", ["雪分"], "教务"),
        DictEntry("绩点", ["积点", "机点"], "教务"),
        DictEntry("辅导员", ["辅导元"], "教职"),
        DictEntry("副教授", ["副叫授"], "教职"),
        DictEntry("院士", ["院是"], "学术"),
        DictEntry("实验室", ["实验时"], "科研"),
        DictEntry("课题", ["棵题"], "科研"),
        DictEntry("学术论文", ["学术轮文"], "学术"),
        DictEntry("综合素质", ["综合素治"], "评价"),
        DictEntry("义务教育", ["义务教欲"], "政策"),
    ]

    # Government/Politics domain
    _GOVERNMENT: List[DictEntry] = [
        DictEntry("国务院", ["国务愿"], "政府"),
        DictEntry("发改委", ["发改为"], "政府"),
        DictEntry("财政部", ["财政步"], "政府"),
        DictEntry("住建部", ["住建步"], "政府"),
        DictEntry("公安部", ["公安步"], "政府"),
        DictEntry("监察委", ["监查委"], "政府"),
        DictEntry("人大", ["认大"], "机关"),
        DictEntry("政协", ["正协"], "机关"),
        DictEntry("党纪", ["当纪"], "党建"),
        DictEntry("巡视", ["寻视"], "纪检"),
        DictEntry("通报", ["通抱"], "政务"),
        DictEntry("批示", ["批是"], "政务"),
        DictEntry("基层治理", ["基层之理"], "治理"),
        DictEntry("营商环境", ["营业环竟"], "经济"),
        DictEntry("供给侧改革", ["供给策改革"], "政策"),
        DictEntry("放管服", ["放管福"], "改革"),
        DictEntry("乡村振兴", ["乡存振兴"], "政策"),
        DictEntry("一带一路", ["一戴一路"], "政策"),
        DictEntry("高质量发展", ["搞质量发展"], "政策"),
        DictEntry("共同富裕", ["共同付裕"], "政策"),
    ]

    # Domain registry
    _DOMAINS: Dict[str, List[DictEntry]] = {
        'medical': _MEDICAL,
        'tech': _TECH,
        'finance': _FINANCE,
        'legal': _LEGAL,
        'education': _EDUCATION,
        'government': _GOVERNMENT,
    }

    def __init__(
        self,
        domain: str = 'general',
        custom_dict: Optional[List[DictEntry]] = None,
        max_prompt_terms: int = 50,
    ):
        """
        Initialize domain dictionary.

        Args:
            domain: Domain name ('medical', 'tech', 'finance', 'legal', 'general').
            custom_dict: Additional custom dictionary entries.
            max_prompt_terms: Maximum number of terms to include in prompt.
        """
        self.domain = domain
        self.max_prompt_terms = max_prompt_terms

        # Build active dictionary
        self._entries: List[DictEntry] = []
        if domain in self._DOMAINS:
            self._entries.extend(self._DOMAINS[domain])
        if custom_dict:
            self._entries.extend(custom_dict)

        # Build alias → standard form mapping for post-correction
        self._alias_map: Dict[str, str] = {}
        for entry in self._entries:
            for alias in entry.aliases:
                if alias != entry.term:
                    self._alias_map[alias] = entry.term

    @property
    def available_domains(self) -> List[str]:
        """List of available built-in domains."""
        return list(self._DOMAINS.keys())

    def get_prompt(self, language: str = 'zh') -> str:
        """
        Generate a Whisper prompt with domain-specific terms.

        The prompt biases Whisper towards recognizing domain vocabulary.

        Args:
            language: Language code.

        Returns:
            Prompt string for Whisper initial_prompt parameter.
        """
        if not self._entries:
            return ""

        # Sort by boost weight (highest first) and take top N
        sorted_entries = sorted(
            self._entries, key=lambda e: e.boost_weight, reverse=True
        )[:self.max_prompt_terms]

        terms = [e.term for e in sorted_entries]

        if language == 'zh':
            return "以下专业词汇可能出现：" + "、".join(terms)
        else:
            return "The following technical terms may appear: " + ", ".join(terms)

    def post_correct(self, text: str) -> str:
        """
        Post-correct common ASR errors using domain dictionary.

        Args:
            text: ASR output text.

        Returns:
            Corrected text.
        """
        if not text or not self._alias_map:
            return text

        result = text
        # Sort by alias length (longest first) to avoid partial matches
        sorted_aliases = sorted(self._alias_map.keys(), key=len, reverse=True)

        for alias in sorted_aliases:
            if alias in result:
                result = result.replace(alias, self._alias_map[alias])

        return result

    def get_terms(self, category: Optional[str] = None) -> List[str]:
        """
        Get list of domain terms.

        Args:
            category: Optional category filter.

        Returns:
            List of term strings.
        """
        if category:
            return [e.term for e in self._entries if e.category == category]
        return [e.term for e in self._entries]

    def add_term(
        self,
        term: str,
        aliases: Optional[List[str]] = None,
        category: str = 'custom',
        boost_weight: float = 1.0,
    ):
        """
        Add a custom term to the dictionary.

        Args:
            term: Standard term form.
            aliases: Common misrecognitions.
            category: Category name.
            boost_weight: Priority weight.
        """
        entry = DictEntry(
            term=term,
            aliases=aliases or [],
            category=category,
            boost_weight=boost_weight,
        )
        self._entries.append(entry)

        # Update alias map
        for alias in entry.aliases:
            if alias != term:
                self._alias_map[alias] = term

    def load_custom_dict(self, filepath: str):
        """
        Load a custom dictionary from a file.

        Supported formats:
        - .txt: One entry per line:
            standard_term|alias1,alias2|category|weight
        - .json: JSON array of objects:
            [{"term": "...", "aliases": [...], "category": "...", "weight": 1.0}, ...]
        - .csv: CSV with header (term,aliases,category,weight)

        Args:
            filepath: Path to dictionary file.
        """
        import os
        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.json':
            self._load_json_dict(filepath)
        elif ext == '.csv':
            self._load_csv_dict(filepath)
        else:
            self._load_txt_dict(filepath)

    def _load_txt_dict(self, filepath: str):
        """Load pipe-delimited text dictionary."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('|')
                    if len(parts) >= 1:
                        term = parts[0].strip()
                        aliases = []
                        category = 'custom'
                        weight = 1.0
                        if len(parts) >= 2 and parts[1].strip():
                            aliases = [a.strip() for a in parts[1].split(',')]
                        if len(parts) >= 3 and parts[2].strip():
                            category = parts[2].strip()
                        if len(parts) >= 4:
                            try:
                                weight = float(parts[3].strip())
                            except ValueError:
                                pass
                        self.add_term(term, aliases, category, weight)
            logger.info("Loaded custom text dictionary from %s", filepath)
        except Exception as e:
            logger.error("Failed to load custom text dictionary: %s", e)

    def _load_json_dict(self, filepath: str):
        """Load JSON format dictionary."""
        import json
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and 'term' in entry:
                        self.add_term(
                            term=entry['term'],
                            aliases=entry.get('aliases', []),
                            category=entry.get('category', 'custom'),
                            boost_weight=entry.get('weight', 1.0),
                        )
            logger.info("Loaded custom JSON dictionary from %s (%d entries)", filepath, len(data))
        except Exception as e:
            logger.error("Failed to load custom JSON dictionary: %s", e)

    def _load_csv_dict(self, filepath: str):
        """Load CSV format dictionary."""
        import csv
        try:
            with open(filepath, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    if 'term' in row and row['term'].strip():
                        aliases = []
                        if row.get('aliases', '').strip():
                            aliases = [a.strip() for a in row['aliases'].split(',')]
                        self.add_term(
                            term=row['term'].strip(),
                            aliases=aliases,
                            category=row.get('category', 'custom').strip() or 'custom',
                            boost_weight=float(row.get('weight', 1.0) or 1.0),
                        )
                        count += 1
            logger.info("Loaded custom CSV dictionary from %s (%d entries)", filepath, count)
        except Exception as e:
            logger.error("Failed to load custom CSV dictionary: %s", e)

    def load_dicts_from_directory(self, directory: str):
        """
        Load all dictionary files from a directory.

        Supports .txt, .json, .csv files.

        Args:
            directory: Path to directory containing dictionary files.
        """
        import os
        if not os.path.isdir(directory):
            logger.warning("Dictionary directory not found: %s", directory)
            return

        count = 0
        for filename in sorted(os.listdir(directory)):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[1].lower()
                if ext in ('.txt', '.json', '.csv'):
                    self.load_custom_dict(filepath)
                    count += 1

        logger.info("Loaded %d dictionary files from %s", count, directory)

    def save_custom_dict(self, filepath: str):
        """
        Save current dictionary entries to a file.

        Supports .json and .txt formats.

        Args:
            filepath: Path to save file.
        """
        import json
        import os
        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext == '.json':
                data = [
                    {
                        'term': e.term,
                        'aliases': e.aliases,
                        'category': e.category,
                        'weight': e.boost_weight,
                    }
                    for e in self._entries
                ]
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("# term|aliases|category|weight\n")
                    for e in self._entries:
                        aliases_str = ','.join(e.aliases)
                        f.write(f"{e.term}|{aliases_str}|{e.category}|{e.boost_weight}\n")
            logger.info("Saved dictionary to %s (%d entries)", filepath, len(self._entries))
        except Exception as e:
            logger.error("Failed to save dictionary: %s", e)

    def get_stats(self) -> dict:
        """
        Get dictionary statistics.

        Returns:
            Dict with entry count, category count, etc.
        """
        categories: Dict[str, int] = {}
        for entry in self._entries:
            categories[entry.category] = categories.get(entry.category, 0) + 1

        return {
            'domain': self.domain,
            'total_entries': len(self._entries),
            'total_aliases': len(self._alias_map),
            'categories': categories,
        }

    def get_whisper_kwargs(self) -> dict:
        """
        Get kwargs to pass to WhisperBridge.transcribe().

        Returns:
            Dict with 'initial_prompt' key.
        """
        prompt = self.get_prompt()
        if prompt:
            return {'initial_prompt': prompt}
        return {}


def get_domain_prompt(domain: str) -> str:
    """
    Convenience function to get a domain prompt string.

    Args:
        domain: Domain name.

    Returns:
        Prompt string.
    """
    d = DomainDictionary(domain=domain)
    return d.get_prompt()