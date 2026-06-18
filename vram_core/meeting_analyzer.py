"""
Meeting Analyzer: LLM-powered meeting analysis with topic detection,
sentiment tracking, participation analysis, and action item extraction.
"""

import logging
import time
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TopicSegment:
    """Detected topic in a meeting."""
    title: str
    start_time: float
    end_time: float
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    speaker_contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class SentimentPoint:
    """Sentiment at a point in the meeting."""
    timestamp: float
    speaker: str = ""
    sentiment: str = "neutral"  # positive/negative/neutral
    score: float = 0.0
    text: str = ""


@dataclass
class ParticipationStats:
    """Speaker participation statistics."""
    speaker_id: str
    total_time: float = 0.0
    turn_count: int = 0
    word_count: int = 0
    question_count: int = 0
    interruption_count: int = 0
    participation_pct: float = 0.0


@dataclass
class ActionItem:
    """Extracted action item."""
    description: str
    assignee: str = ""
    deadline: str = ""
    priority: str = "medium"
    context: str = ""


@dataclass
class Decision:
    """Extracted decision."""
    content: str
    context: str = ""
    proposer: str = ""
    timestamp: float = 0.0


@dataclass
class MeetingTimeline:
    """Timeline of meeting events."""
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_event(self, event_type: str, timestamp: float, content: str, speaker: str = ""):
        self.events.append({
            "type": event_type,
            "timestamp": timestamp,
            "content": content,
            "speaker": speaker,
        })


@dataclass
class MeetingAnalysis:
    """Complete meeting analysis result."""
    topics: List[TopicSegment] = field(default_factory=list)
    sentiments: List[SentimentPoint] = field(default_factory=list)
    participation: List[ParticipationStats] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    timeline: MeetingTimeline = field(default_factory=MeetingTimeline)
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    overall_sentiment: str = "neutral"
    duration_seconds: float = 0.0
    analysis_time_ms: float = 0.0


class MeetingAnalyzer:
    """
    LLM-powered meeting analyzer with topic detection, sentiment analysis,
    participation tracking, and action item extraction.
    
    Features:
        - Topic detection and segmentation
        - Sentiment analysis per speaker
        - Participation analysis
        - Decision tracking
        - Action item extraction
        - Timeline generation
        - LLM-enhanced analysis (optional)
    
    Example:
        >>> analyzer = MeetingAnalyzer()
        >>> result = analyzer.analyze(transcript, speaker_segments=segments)
        >>> print(result.summary)
        >>> for item in result.action_items:
        ...     print(f"{item.assignee}: {item.description}")
    """
    
    # Action item keywords
    ACTION_KEYWORDS = [
        "需要", "应该", "必须", "负责", "跟进", "完成", "提交", "准备",
        "need", "should", "must", "responsible", "follow up", "complete",
        "deadline", "before", "by", "assign", "todo", "action",
        "will do", "going to", "plan to", "next step",
    ]
    
    # Decision keywords
    DECISION_KEYWORDS = [
        "决定", "确定", "同意", "通过", "批准", "确认",
        "decide", "agreed", "approved", "confirmed", "resolved",
        "we will", "let's go", "final decision", "it's settled",
    ]
    
    # Question patterns
    QUESTION_PATTERNS = re.compile(r'[?？]|^(what|why|how|when|where|who|which|is|are|do|does|can|could|would|should)\b', re.IGNORECASE)
    
    def __init__(self, llm_client=None):
        """
        Initialize meeting analyzer.
        
        Args:
            llm_client: Optional LLMClient for enhanced analysis
        """
        self._llm = llm_client
    
    def analyze(
        self,
        transcript: str,
        speaker_segments: Optional[List[Dict[str, Any]]] = None,
        sample_rate: int = 16000,
    ) -> MeetingAnalysis:
        """
        Perform comprehensive meeting analysis.
        
        Args:
            transcript: Full meeting transcript text
            speaker_segments: List of {speaker, text, start_time, end_time}
            sample_rate: Audio sample rate (for time calculations)
            
        Returns:
            MeetingAnalysis with all analysis results
        """
        start_time = time.time()
        result = MeetingAnalysis()
        
        # Default segments if not provided
        if not speaker_segments:
            speaker_segments = self._segment_by_speaker(transcript)
        
        # Calculate duration
        if speaker_segments:
            result.duration_seconds = max(s.get("end_time", 0) for s in speaker_segments)
        
        # 1. Participation analysis
        result.participation = self._analyze_participation(speaker_segments, transcript)
        
        # 2. Topic detection
        result.topics = self._detect_topics(speaker_segments, transcript)
        
        # 3. Sentiment analysis
        result.sentiments = self._analyze_sentiment(speaker_segments)
        
        # 4. Action item extraction
        result.action_items = self._extract_action_items(speaker_segments)
        
        # 5. Decision extraction
        result.decisions = self._extract_decisions(speaker_segments)
        
        # 6. Timeline generation
        result.timeline = self._generate_timeline(speaker_segments, result)
        
        # 7. Key points extraction
        result.key_points = self._extract_key_points(transcript, speaker_segments)
        
        # 8. Overall sentiment
        if result.sentiments:
            scores = [s.score for s in result.sentiments]
            avg = sum(scores) / len(scores)
            result.overall_sentiment = "positive" if avg > 0.2 else "negative" if avg < -0.2 else "neutral"
        
        # 9. LLM-enhanced summary (if available)
        if self._llm:
            result.summary = self._generate_llm_summary(transcript, result)
        else:
            result.summary = self._generate_basic_summary(transcript, result)
        
        result.analysis_time_ms = (time.time() - start_time) * 1000
        logger.info("Meeting analysis completed in %.1fms", result.analysis_time_ms)
        
        return result
    
    def _segment_by_speaker(self, transcript: str) -> List[Dict[str, Any]]:
        """Segment transcript into speaker turns."""
        segments = []
        lines = transcript.split('\n')
        current_speaker = "Speaker_1"
        current_text = []
        time_offset = 0.0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for speaker labels: "Speaker: text" or "[Speaker] text"
            speaker_match = re.match(r'^(?:\[([^\]]+)\]|([^:：]+))[:：]\s*(.+)', line)
            if speaker_match:
                if current_text:
                    text = ' '.join(current_text)
                    duration = max(len(text) * 0.05, 1.0)
                    segments.append({
                        "speaker": current_speaker,
                        "text": text,
                        "start_time": time_offset,
                        "end_time": time_offset + duration,
                    })
                    time_offset += duration
                    current_text = []
                
                current_speaker = (speaker_match.group(1) or speaker_match.group(2) or "").strip()
                current_text.append(speaker_match.group(3))
            else:
                current_text.append(line)
        
        # Last segment
        if current_text:
            text = ' '.join(current_text)
            duration = max(len(text) * 0.05, 1.0)
            segments.append({
                "speaker": current_speaker,
                "text": text,
                "start_time": time_offset,
                "end_time": time_offset + duration,
            })
        
        return segments or [{"speaker": "Unknown", "text": transcript, "start_time": 0, "end_time": len(transcript) * 0.05}]
    
    def _analyze_participation(
        self, segments: List[Dict[str, Any]], transcript: str
    ) -> List[ParticipationStats]:
        """Analyze speaker participation."""
        stats = defaultdict(lambda: ParticipationStats(speaker_id=""))
        total_time = 0
        
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            duration = seg.get("end_time", 0) - seg.get("start_time", 0)
            text = seg.get("text", "")
            words = text.split()
            
            s = stats[speaker]
            s.speaker_id = speaker
            s.total_time += duration
            s.turn_count += 1
            s.word_count += len(words)
            
            if self.QUESTION_PATTERNS.search(text):
                s.question_count += 1
            
            total_time += duration
        
        # Calculate percentages
        result = []
        for speaker, s in stats.items():
            s.participation_pct = (s.total_time / total_time * 100) if total_time > 0 else 0
            result.append(s)
        
        return sorted(result, key=lambda x: x.total_time, reverse=True)
    
    def _detect_topics(
        self, segments: List[Dict[str, Any]], transcript: str
    ) -> List[TopicSegment]:
        """Detect topics using keyword-based segmentation."""
        if not segments:
            return []
        
        # Simple topic detection using text similarity between segments
        topics = []
        current_topic_text = []
        current_start = segments[0].get("start_time", 0)
        current_keywords = set()
        
        # Use simple keyword extraction
        all_words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', transcript.lower())
        word_freq = defaultdict(int)
        for w in all_words:
            if len(w) > 1:
                word_freq[w] += 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30]
        
        topic_idx = 0
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            current_topic_text.append(text)
            
            # Extract keywords from current segment
            seg_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower()))
            current_keywords.update(seg_words & {k[0] for k in top_keywords})
            
            # Check if we should start a new topic (every 5 segments or at natural breaks)
            if (i + 1) % 5 == 0 or i == len(segments) - 1:
                topic_text = ' '.join(current_topic_text)
                topic_keywords = list(current_keywords)[:5]
                
                title = topic_keywords[0] if topic_keywords else f"Topic {topic_idx + 1}"
                
                topics.append(TopicSegment(
                    title=title,
                    start_time=current_start,
                    end_time=seg.get("end_time", 0),
                    summary=topic_text[:200] + "..." if len(topic_text) > 200 else topic_text,
                    keywords=topic_keywords,
                ))
                
                topic_idx += 1
                current_topic_text = []
                current_keywords = set()
                if i + 1 < len(segments):
                    current_start = segments[i + 1].get("start_time", 0)
        
        return topics
    
    def _analyze_sentiment(self, segments: List[Dict[str, Any]]) -> List[SentimentPoint]:
        """Analyze sentiment using keyword-based approach."""
        # Sentiment lexicons
        positive_words = {
            "好", "对", "是", "同意", "不错", "优秀", "棒", "赞", "支持", "成功",
            "good", "great", "excellent", "agree", "yes", "right", "perfect",
            "wonderful", "fantastic", "amazing", "love", "happy", "success",
        }
        negative_words = {
            "不", "没", "差", "错", "问题", "困难", "失败", "担心", "反对", "拒绝",
            "bad", "no", "wrong", "problem", "fail", "issue", "concern",
            "worried", "difficult", "terrible", "hate", "angry", "error",
        }
        
        sentiments = []
        for seg in segments:
            text = seg.get("text", "").lower()
            words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text))
            
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)
            total = pos_count + neg_count
            
            if total == 0:
                sentiment = "neutral"
                score = 0.0
            else:
                score = (pos_count - neg_count) / total
                if score > 0.2:
                    sentiment = "positive"
                elif score < -0.2:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
            
            sentiments.append(SentimentPoint(
                timestamp=seg.get("start_time", 0),
                speaker=seg.get("speaker", ""),
                sentiment=sentiment,
                score=score,
                text=seg.get("text", "")[:100],
            ))
        
        return sentiments
    
    def _extract_action_items(self, segments: List[Dict[str, Any]]) -> List[ActionItem]:
        """Extract action items from transcript."""
        items = []
        seen = set()
        
        for seg in segments:
            text = seg.get("text", "")
            sentences = re.split(r'[。.！!？?]', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 5:
                    continue
                
                # Check for action keywords
                lower = sentence.lower()
                has_action = any(kw in lower for kw in self.ACTION_KEYWORDS)
                
                if has_action:
                    # Avoid duplicates
                    key = sentence[:50]
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    # Try to extract assignee
                    assignee = seg.get("speaker", "")
                    assignee_match = re.search(r'(\w+)\s*(负责|请|needs to|should|will)', sentence)
                    if assignee_match:
                        assignee = assignee_match.group(1)
                    
                    # Determine priority
                    priority = "high" if any(w in lower for w in ["紧急", "立即", "asap", "urgent", "immediately"]) else "medium"
                    
                    items.append(ActionItem(
                        description=sentence,
                        assignee=assignee,
                        priority=priority,
                        context=text[:200],
                    ))
        
        return items
    
    def _extract_decisions(self, segments: List[Dict[str, Any]]) -> List[Decision]:
        """Extract decisions from transcript."""
        decisions = []
        seen = set()
        
        for seg in segments:
            text = seg.get("text", "")
            sentences = re.split(r'[。.！!？?]', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 5:
                    continue
                
                lower = sentence.lower()
                has_decision = any(kw in lower for kw in self.DECISION_KEYWORDS)
                
                if has_decision:
                    key = sentence[:50]
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    decisions.append(Decision(
                        content=sentence,
                        context=text[:200],
                        proposer=seg.get("speaker", ""),
                        timestamp=seg.get("start_time", 0),
                    ))
        
        return decisions
    
    def _generate_timeline(
        self, segments: List[Dict[str, Any]], analysis: MeetingAnalysis
    ) -> MeetingTimeline:
        """Generate meeting timeline."""
        timeline = MeetingTimeline()
        
        # Add speaker turns
        for seg in segments:
            timeline.add_event(
                "speech",
                seg.get("start_time", 0),
                seg.get("text", "")[:100],
                seg.get("speaker", ""),
            )
        
        # Add topic changes
        for topic in analysis.topics:
            timeline.add_event(
                "topic_change",
                topic.start_time,
                f"Topic: {topic.title}",
            )
        
        # Add decisions
        for decision in analysis.decisions:
            timeline.add_event(
                "decision",
                decision.timestamp,
                decision.content,
                decision.proposer,
            )
        
        timeline.events.sort(key=lambda e: e["timestamp"])
        return timeline
    
    def _extract_key_points(
        self, transcript: str, segments: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract key points from transcript."""
        key_points = []
        
        # Find sentences with high keyword density
        all_words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', transcript.lower())
        word_freq = defaultdict(int)
        for w in all_words:
            if len(w) > 1:
                word_freq[w] += 1
        
        important_words = {k for k, v in word_freq.items() if v >= 3}
        
        for seg in segments:
            text = seg.get("text", "")
            sentences = re.split(r'[。.！!？?；;]', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', sentence.lower()))
                importance = len(words & important_words)
                
                if importance >= 3:
                    key_points.append(sentence)
        
        # Return top key points, deduplicated
        unique_points = list(dict.fromkeys(key_points))
        return unique_points[:10]
    
    def _generate_llm_summary(
        self, transcript: str, analysis: MeetingAnalysis
    ) -> str:
        """Generate summary using LLM."""
        try:
            prompt = f"""请对以下会议记录进行专业摘要，包括：
1. 会议主题概述
2. 关键讨论点
3. 主要决策
4. 行动事项

会议记录：
{transcript[:3000]}

请用中文输出简洁专业的摘要。"""
            
            response = self._llm.chat(
                prompt=prompt,
                system_prompt="你是一个专业的会议记录分析师。请提供简洁、结构化的会议摘要。",
                max_tokens=500,
            )
            return response.content
        except (RuntimeError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning("LLM summary generation failed: %s", e)
            return self._generate_basic_summary(transcript, analysis)
    
    def _generate_basic_summary(
        self, transcript: str, analysis: MeetingAnalysis
    ) -> str:
        """Generate basic summary without LLM."""
        parts = []
        
        # Duration
        mins = analysis.duration_seconds / 60
        parts.append(f"会议时长: {mins:.1f} 分钟")
        
        # Participants
        speakers = [p.speaker_id for p in analysis.participation]
        parts.append(f"参与人: {', '.join(speakers)} ({len(speakers)} 人)")
        
        # Topics
        if analysis.topics:
            topic_names = [t.title for t in analysis.topics[:5]]
            parts.append(f"讨论话题: {', '.join(topic_names)}")
        
        # Decisions
        if analysis.decisions:
            parts.append(f"决策数量: {len(analysis.decisions)} 项")
            for d in analysis.decisions[:3]:
                parts.append(f"  - {d.content}")
        
        # Action items
        if analysis.action_items:
            parts.append(f"行动项: {len(analysis.action_items)} 项")
            for a in analysis.action_items[:3]:
                parts.append(f"  - [{a.assignee}] {a.description}")
        
        # Sentiment
        parts.append(f"整体氛围: {analysis.overall_sentiment}")
        
        return '\n'.join(parts)