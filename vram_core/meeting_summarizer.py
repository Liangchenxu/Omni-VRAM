"""
Meeting Summarizer
===================

AI-powered meeting summarization module that generates structured meeting
minutes from transcription segments.

Features:
- Extract action items, decisions, and key topics
- Generate structured meeting minutes
- Identify speakers and their contributions
- Timeline-based summary with timestamps

Usage:
    from vram_core.meeting_summarizer import MeetingSummarizer

    summarizer = MeetingSummarizer()
    minutes = summarizer.summarize(segments)
    print(minutes.summary)
    print(minutes.action_items)
"""

import re
import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ActionItem:
    """An action item extracted from meeting."""
    content: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    priority: str = "normal"  # low, normal, high
    timestamp: float = 0.0


@dataclass
class Decision:
    """A decision made in the meeting."""
    content: str
    context: str = ""
    timestamp: float = 0.0


@dataclass
class TopicSegment:
    """A topic discussed in the meeting."""
    topic: str
    start_time: float = 0.0
    end_time: float = 0.0
    key_points: List[str] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)


@dataclass
class SpeakerContribution:
    """Summary of a speaker's contribution."""
    speaker: str
    total_duration: float = 0.0
    segment_count: int = 0
    key_topics: List[str] = field(default_factory=list)


@dataclass
class MeetingMinutes:
    """Structured meeting minutes."""
    title: str = ""
    summary: str = ""
    topics: List[TopicSegment] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    speaker_contributions: List[SpeakerContribution] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    total_duration: float = 0.0
    participant_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'title': self.title,
            'summary': self.summary,
            'total_duration': self.total_duration,
            'participant_count': self.participant_count,
            'topics': [
                {
                    'topic': t.topic,
                    'start_time': t.start_time,
                    'end_time': t.end_time,
                    'key_points': t.key_points,
                    'speakers': t.speakers,
                }
                for t in self.topics
            ],
            'action_items': [
                {
                    'content': a.content,
                    'assignee': a.assignee,
                    'deadline': a.deadline,
                    'priority': a.priority,
                    'timestamp': a.timestamp,
                }
                for a in self.action_items
            ],
            'decisions': [
                {
                    'content': d.content,
                    'context': d.context,
                    'timestamp': d.timestamp,
                }
                for d in self.decisions
            ],
            'speaker_contributions': [
                {
                    'speaker': s.speaker,
                    'total_duration': s.total_duration,
                    'segment_count': s.segment_count,
                    'key_topics': s.key_topics,
                }
                for s in self.speaker_contributions
            ],
            'timeline': self.timeline,
        }

    def to_markdown(self) -> str:
        """Generate markdown formatted meeting minutes."""
        lines = []

        if self.title:
            lines.append(f"# {self.title}\n")
        else:
            lines.append("# 会议纪要\n")

        # Basic info
        duration_min = self.total_duration / 60
        lines.append(f"**会议时长**: {duration_min:.1f} 分钟")
        lines.append(f"**参与人数**: {self.participant_count}\n")

        # Summary
        if self.summary:
            lines.append("## 摘要\n")
            lines.append(f"{self.summary}\n")

        # Topics
        if self.topics:
            lines.append("## 讨论议题\n")
            for i, topic in enumerate(self.topics, 1):
                time_str = self._format_time(topic.start_time)
                lines.append(f"### {i}. {topic.topic} [{time_str}]\n")
                if topic.key_points:
                    for point in topic.key_points:
                        lines.append(f"- {point}")
                if topic.speakers:
                    lines.append(f"\n*参与者*: {', '.join(topic.speakers)}")
                lines.append("")

        # Decisions
        if self.decisions:
            lines.append("## 决策事项\n")
            for i, dec in enumerate(self.decisions, 1):
                time_str = self._format_time(dec.timestamp)
                lines.append(f"{i}. **{dec.content}** [{time_str}]")
                if dec.context:
                    lines.append(f"   - 背景: {dec.context}")
            lines.append("")

        # Action items
        if self.action_items:
            lines.append("## 行动项\n")
            lines.append("| 序号 | 内容 | 负责人 | 截止时间 | 优先级 |")
            lines.append("|------|------|--------|----------|--------|")
            for i, item in enumerate(self.action_items, 1):
                assignee = item.assignee or "待定"
                deadline = item.deadline or "待定"
                priority_map = {'high': '🔴 高', 'normal': '🟡 中', 'low': '🟢 低'}
                priority = priority_map.get(item.priority, '🟡 中')
                lines.append(f"| {i} | {item.content} | {assignee} | {deadline} | {priority} |")
            lines.append("")

        # Speaker contributions
        if self.speaker_contributions:
            lines.append("## 发言统计\n")
            lines.append("| 发言人 | 发言次数 | 时长 | 关键话题 |")
            lines.append("|--------|----------|------|----------|")
            for s in self.speaker_contributions:
                duration_str = f"{s.total_duration:.0f}秒"
                topics_str = ', '.join(s.key_topics[:3]) if s.key_topics else '-'
                lines.append(
                    f"| {s.speaker} | {s.segment_count} | {duration_str} | {topics_str} |"
                )
            lines.append("")

        return '\n'.join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"


class MeetingSummarizer:
    """
    Meeting summarization engine.

    Extracts structured information from meeting transcription segments.

    Args:
        language: Language code ('zh' or 'en').
        min_segment_duration: Minimum segment duration to consider.
    """

    # Action item trigger phrases (Chinese)
    _ACTION_TRIGGERS_ZH = [
        r'请.{1,10}(完成|提交|准备|跟进|处理|确认|安排|联系|跟进)',
        r'(需要|必须|务必|记得).{1,15}(完成|提交|准备|跟进|处理)',
        r'(下一步|接下来|后续).{1,15}(做|进行|推进|落实)',
        r'(.{2,8})负责.{1,20}',
        r'(截止|deadline|期限).{1,10}',
        r'(尽快|马上|立刻|今天|明天|本周|下周).{1,15}(完成|提交|发送|处理)',
        r'大家.{1,5}(记住|注意|留意)',
        r'(任务|待办|TODO|todo).{1,20}',
    ]

    # Decision trigger phrases (Chinese)
    _DECISION_TRIGGERS_ZH = [
        r'(决定|确定|确认|同意|通过|批准)',
        r'(最终|最后).{1,10}(选择|方案|决定)',
        r'(一致|大家|全体).{1,5}(同意|通过|认为|决定)',
        r'(方案|计划|策略)是',
        r'(结论|结果)是',
        r'就这样(定了|决定|办)',
    ]

    # Topic change indicators
    _TOPIC_TRIGGERS_ZH = [
        r'(下一个|接下来|另外|还有|关于|说到|谈到|关于)',
        r'(第[一二三四五六七八九十]+[个点项议题])',
        r'(问题|议题|话题|方面)',
        r'(我们来|现在来|下面).{0,5}(讨论|谈谈|说说|看看)',
    ]

    # Action triggers (English)
    _ACTION_TRIGGERS_EN = [
        r'(please|kindly)\s+\w+\s+(finish|submit|prepare|follow.up|process)',
        r'(need|must|should)\s+(to\s+)?\w+\s+(complete|submit|prepare)',
        r'(action\s+item|todo|task)',
        r'(\w+)\s+(is\s+)?responsible\s+for',
        r'(deadline|due\s+date|by\s+end\s+of)',
        r'(asap|urgently|immediately)',
    ]

    # Decision triggers (English)
    _DECISION_TRIGGERS_EN = [
        r'(decided|agreed|approved|confirmed)',
        r'(the\s+decision\s+is|we\s+decide)',
        r'(consensus|unanimously)',
        r'(it\s+is\s+agreed|it\s+was\s+decided)',
    ]

    def __init__(
        self,
        language: str = 'zh',
        min_segment_duration: float = 1.0,
    ):
        self.language = language
        self.min_segment_duration = min_segment_duration

        if language == 'zh':
            self._action_triggers = [
                re.compile(p) for p in self._ACTION_TRIGGERS_ZH
            ]
            self._decision_triggers = [
                re.compile(p) for p in self._DECISION_TRIGGERS_ZH
            ]
            self._topic_triggers = [
                re.compile(p) for p in self._TOPIC_TRIGGERS_ZH
            ]
        else:
            self._action_triggers = [
                re.compile(p, re.IGNORECASE) for p in self._ACTION_TRIGGERS_EN
            ]
            self._decision_triggers = [
                re.compile(p, re.IGNORECASE) for p in self._DECISION_TRIGGERS_EN
            ]
            self._topic_triggers = []

    def summarize(
        self,
        segments: List[Any],
        title: Optional[str] = None,
    ) -> MeetingMinutes:
        """
        Generate meeting minutes from transcription segments.

        Args:
            segments: List of transcription segments. Each segment should have
                     attributes: text, start_time, end_time, speaker (optional).
            title: Optional meeting title.

        Returns:
            MeetingMinutes object with structured meeting data.
        """
        if not segments:
            return MeetingMinutes(title=title or "空会议")

        # Normalize segments
        norm_segments = self._normalize_segments(segments)

        # Calculate total duration
        total_duration = 0.0
        if norm_segments:
            total_duration = norm_segments[-1]['end_time']

        # Extract components
        action_items = self._extract_action_items(norm_segments)
        decisions = self._extract_decisions(norm_segments)
        topics = self._extract_topics(norm_segments)
        speaker_contributions = self._analyze_speakers(norm_segments)
        timeline = self._build_timeline(norm_segments)

        # Generate summary text
        summary = self._generate_summary(norm_segments, topics, decisions)

        # Count unique participants
        speakers = set()
        for seg in norm_segments:
            if seg.get('speaker'):
                speakers.add(seg['speaker'])

        minutes = MeetingMinutes(
            title=title or self._generate_title(topics),
            summary=summary,
            topics=topics,
            action_items=action_items,
            decisions=decisions,
            speaker_contributions=speaker_contributions,
            timeline=timeline,
            total_duration=total_duration,
            participant_count=len(speakers),
        )

        logger.info(
            "Meeting summarized: %d topics, %d action items, %d decisions, %d speakers",
            len(topics), len(action_items), len(decisions), len(speakers),
        )

        return minutes

    def _normalize_segments(self, segments: List[Any]) -> List[Dict]:
        """Convert segments to normalized dict format."""
        result = []
        for seg in segments:
            if isinstance(seg, dict):
                result.append({
                    'text': seg.get('text', ''),
                    'start_time': float(seg.get('start_time', seg.get('start', 0))),
                    'end_time': float(seg.get('end_time', seg.get('end', 0))),
                    'speaker': seg.get('speaker', seg.get('speaker_id', None)),
                })
            else:
                result.append({
                    'text': getattr(seg, 'text', ''),
                    'start_time': float(getattr(seg, 'start_time', getattr(seg, 'start', 0))),
                    'end_time': float(getattr(seg, 'end_time', getattr(seg, 'end', 0))),
                    'speaker': getattr(seg, 'speaker', getattr(seg, 'speaker_id', None)),
                })
        return result

    def _extract_action_items(self, segments: List[Dict]) -> List[ActionItem]:
        """Extract action items from segments."""
        items = []
        for seg in segments:
            text = seg['text']
            for trigger in self._action_triggers:
                if trigger.search(text):
                    # Try to extract assignee
                    assignee = self._extract_assignee(text)
                    deadline = self._extract_deadline(text)
                    priority = self._assess_priority(text)

                    items.append(ActionItem(
                        content=text.strip(),
                        assignee=assignee,
                        deadline=deadline,
                        priority=priority,
                        timestamp=seg['start_time'],
                    ))
                    break  # One match per segment is enough

        return items

    def _extract_decisions(self, segments: List[Dict]) -> List[Decision]:
        """Extract decisions from segments."""
        decisions = []
        for seg in segments:
            text = seg['text']
            for trigger in self._decision_triggers:
                if trigger.search(text):
                    decisions.append(Decision(
                        content=text.strip(),
                        timestamp=seg['start_time'],
                    ))
                    break

        return decisions

    def _extract_topics(self, segments: List[Dict]) -> List[TopicSegment]:
        """Extract and group topics from segments."""
        if not segments:
            return []

        topics: List[TopicSegment] = []
        current_topic = TopicSegment(
            topic="",
            start_time=segments[0]['start_time'],
        )
        topic_texts: List[str] = []
        topic_speakers: set = set()

        for seg in segments:
            text = seg['text']

            # Check for topic change
            is_topic_change = False
            for trigger in self._topic_triggers:
                if trigger.search(text):
                    is_topic_change = True
                    break

            # Also consider long silence as topic change
            if current_topic.end_time > 0:
                gap = seg['start_time'] - current_topic.end_time
                if gap > 30:  # 30 second silence
                    is_topic_change = True

            if is_topic_change and topic_texts:
                # Save current topic
                current_topic.topic = self._summarize_topic(topic_texts)
                current_topic.key_points = self._extract_key_points(topic_texts)
                current_topic.speakers = list(topic_speakers)
                topics.append(current_topic)

                # Start new topic
                current_topic = TopicSegment(
                    topic="",
                    start_time=seg['start_time'],
                )
                topic_texts = []
                topic_speakers = set()

            topic_texts.append(text)
            current_topic.end_time = seg['end_time']
            if seg.get('speaker'):
                topic_speakers.add(seg['speaker'])

        # Save last topic
        if topic_texts:
            current_topic.topic = self._summarize_topic(topic_texts)
            current_topic.key_points = self._extract_key_points(topic_texts)
            current_topic.speakers = list(topic_speakers)
            topics.append(current_topic)

        return topics

    def _analyze_speakers(self, segments: List[Dict]) -> List[SpeakerContribution]:
        """Analyze speaker contributions."""
        speaker_data: Dict[str, Dict] = {}

        for seg in segments:
            speaker = seg.get('speaker') or 'unknown'
            if speaker not in speaker_data:
                speaker_data[speaker] = {
                    'total_duration': 0.0,
                    'segment_count': 0,
                    'texts': [],
                }
            speaker_data[speaker]['total_duration'] += seg['end_time'] - seg['start_time']
            speaker_data[speaker]['segment_count'] += 1
            speaker_data[speaker]['texts'].append(seg['text'])

        contributions = []
        for speaker, data in speaker_data.items():
            # Extract key topics from speaker's text
            all_text = ' '.join(data['texts'])
            key_topics = self._extract_speaker_topics(all_text)

            contributions.append(SpeakerContribution(
                speaker=speaker,
                total_duration=data['total_duration'],
                segment_count=data['segment_count'],
                key_topics=key_topics,
            ))

        # Sort by total duration (most active first)
        contributions.sort(key=lambda c: c.total_duration, reverse=True)
        return contributions

    def _build_timeline(self, segments: List[Dict]) -> List[Dict[str, Any]]:
        """Build a meeting timeline with key events."""
        timeline = []

        for seg in segments:
            text = seg['text']
            event_type = 'speech'

            # Classify event type
            for trigger in self._action_triggers:
                if trigger.search(text):
                    event_type = 'action_item'
                    break

            if event_type == 'speech':
                for trigger in self._decision_triggers:
                    if trigger.search(text):
                        event_type = 'decision'
                        break

            if event_type == 'speech':
                for trigger in self._topic_triggers:
                    if trigger.search(text):
                        event_type = 'topic_change'
                        break

            if event_type != 'speech':  # Only include notable events
                timeline.append({
                    'time': seg['start_time'],
                    'type': event_type,
                    'text': text,
                    'speaker': seg.get('speaker'),
                })

        return timeline

    def _generate_summary(
        self,
        segments: List[Dict],
        topics: List[TopicSegment],
        decisions: List[Decision],
    ) -> str:
        """Generate a brief meeting summary."""
        parts = []

        # Overview
        duration_min = 0.0
        if segments:
            duration_min = (segments[-1]['end_time'] - segments[0]['start_time']) / 60

        speakers = set(seg.get('speaker') for seg in segments if seg.get('speaker'))

        if self.language == 'zh':
            parts.append(f"本次会议时长约{duration_min:.0f}分钟")
            if speakers:
                parts.append(f"，共{len(speakers)}位参会者")

            if topics:
                topic_names = [t.topic for t in topics if t.topic][:5]
                if topic_names:
                    parts.append(f"。主要讨论了{'、'.join(topic_names)}等议题")

            if decisions:
                parts.append(f"，形成了{len(decisions)}项决策")
        else:
            parts.append(f"This meeting lasted approximately {duration_min:.0f} minutes")
            if speakers:
                parts.append(f" with {len(speakers)} participants")
            if topics:
                topic_names = [t.topic for t in topics if t.topic][:5]
                if topic_names:
                    parts.append(f". Main topics discussed: {', '.join(topic_names)}")
            if decisions:
                parts.append(f", with {len(decisions)} decisions made")

        return ''.join(parts) + '。' if self.language == 'zh' else ''.join(parts) + '.'

    def _generate_title(self, topics: List[TopicSegment]) -> str:
        """Generate a meeting title from topics."""
        if not topics:
            return "会议纪要" if self.language == 'zh' else "Meeting Minutes"

        # Use the first topic as title basis
        main_topic = topics[0].topic
        if main_topic:
            if self.language == 'zh':
                return f"会议纪要 - {main_topic}"
            return f"Meeting Minutes - {main_topic}"

        return "会议纪要" if self.language == 'zh' else "Meeting Minutes"

    def _summarize_topic(self, texts: List[str]) -> str:
        """Create a short topic summary from text segments."""
        combined = ' '.join(texts)

        # Try to find the most representative sentence
        sentences = re.split(r'[。！？.!?\n]', combined)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 4]

        if sentences:
            # Use the first meaningful sentence, truncated
            topic = sentences[0]
            if len(topic) > 30:
                topic = topic[:30] + '...'
            return topic

        # Fallback: use first 30 chars
        if len(combined) > 30:
            return combined[:30] + '...'
        return combined

    def _extract_key_points(self, texts: List[str]) -> List[str]:
        """Extract key points from topic texts."""
        combined = ' '.join(texts)
        sentences = re.split(r'[。！？.!?\n]', combined)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 6]

        # Return up to 5 key points
        return sentences[:5]

    def _extract_speaker_topics(self, text: str) -> List[str]:
        """Extract key topics from a speaker's text."""
        # Simple keyword extraction
        if self.language == 'zh':
            # Find Chinese word sequences of 2-6 characters
            words = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
        else:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Count frequency
        freq: Dict[str, int] = {}
        stop_words = {'这个', '那个', '就是', '可以', '应该', '不是', '没有',
                      'the', 'and', 'for', 'that', 'this', 'with', 'from'}

        for w in words:
            if w not in stop_words and len(w) >= 2:
                freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:3]]

    def _extract_assignee(self, text: str) -> Optional[str]:
        """Try to extract the person responsible for an action item."""
        # Pattern: XX负责... or 请XX...
        if self.language == 'zh':
            patterns = [
                r'([\u4e00-\u9fff]{2,4})(?:负责|来办|来做|跟进|处理)',
                r'请([\u4e00-\u9fff]{2,4})(?:完成|提交|准备)',
                r'(?:由|让)([\u4e00-\u9fff]{2,4})(?:负责|处理|跟进)',
            ]
        else:
            patterns = [
                r'(\w+)\s+(?:is\s+)?responsible\s+for',
                r'assign(?:ed)?\s+to\s+(\w+)',
                r'(\w+)\s+should\s+(?:handle|take\s+care\s+of)',
            ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    def _extract_deadline(self, text: str) -> Optional[str]:
        """Try to extract deadline from text."""
        if self.language == 'zh':
            patterns = [
                r'(?:截止|deadline|期限|之前|之前完成)[：:]?\s*(.+?)(?:[,，。\.]|$)',
                r'(\d{1,2}月\d{1,2}[日号](?:之前|前)?)',
                r'((?:今天|明天|后天|本周|下周|月底|年底)(?:之前|前)?)',
                r'((?:这周|下周|本月|下月)\s*(?:内|之前|前)?)',
            ]
        else:
            patterns = [
                r'(?:deadline|due|by)\s+(.+?)(?:[,\.]|$)',
                r'((?:today|tomorrow|this\s+week|next\s+week|end\s+of\s+\w+))',
            ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    def _assess_priority(self, text: str) -> str:
        """Assess priority level of an action item."""
        if self.language == 'zh':
            high_keywords = ['紧急', '立刻', '马上', '尽快', 'ASAP', 'asap', '立即', '今天内']
            low_keywords = ['有空', '方便时', '不急', '慢慢', '后续', '以后']
        else:
            high_keywords = ['urgent', 'asap', 'immediately', 'critical', 'priority']
            low_keywords = ['when possible', 'no rush', 'low priority', 'eventually']

        text_lower = text.lower()
        for kw in high_keywords:
            if kw in text_lower:
                return 'high'
        for kw in low_keywords:
            if kw in text_lower:
                return 'low'
        return 'normal'


def summarize_meeting(segments: List[Any], title: Optional[str] = None) -> MeetingMinutes:
    """
    Convenience function for quick meeting summarization.

    Args:
        segments: Transcription segments.
        title: Optional meeting title.

    Returns:
        MeetingMinutes object.
    """
    summarizer = MeetingSummarizer()
    return summarizer.summarize(segments, title=title)