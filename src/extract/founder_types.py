from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime

class DataType(Enum):
    FACTUAL = "FACTUAL"
    SPECULATIVE = "SPECULATIVE"
    INFERRED = "INFERRED"

class EventType(Enum):
    FOUNDING = "FOUNDING"
    PIVOT = "PIVOT"
    CRISIS = "CRISIS"
    SUCCESS = "SUCCESS"
    OTHER = "OTHER"

class TraitCategory(Enum):
    TRAIT = "TRAIT"
    SKILL = "SKILL"

class TraitOrigin(Enum):
    INNATE = "INNATE"
    DEVELOPED = "DEVELOPED"

@dataclass
class SourcedItem:
    """Base class for items that have source attribution and confidence"""
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass
class Era:
    start: str
    end: Optional[str] = None

@dataclass
class BasicInfo:
    name: str
    domain: List[str]
    financial_background: Optional[str] = None
    era: Optional[Era] = None

@dataclass
class EarlyLifeDetail:
    detail: str
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass(frozen=True)
class TraitChange:
    trait: str
    change: str

    def __hash__(self):
        return hash((self.trait, self.change))

@dataclass(frozen=True)
class TimelineEvent:
    date: str
    event_type: EventType
    description: str
    emotional_context: Optional[str] = None
    trait_changes: tuple[TraitChange, ...] = field(default_factory=tuple)
    alternative_paths: tuple[str, ...] = field(default_factory=tuple)
    external_triggers: tuple[str, ...] = field(default_factory=tuple)
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

    def __post_init__(self):
        if isinstance(self.trait_changes, list):
            object.__setattr__(self, 'trait_changes', tuple(self.trait_changes))
        if isinstance(self.alternative_paths, list):
            object.__setattr__(self, 'alternative_paths', tuple(self.alternative_paths))
        if isinstance(self.external_triggers, list):
            object.__setattr__(self, 'external_triggers', tuple(self.external_triggers))

    def __hash__(self):
        return hash((
            self.date,
            self.event_type,
            self.description,
            self.emotional_context,
            self.trait_changes,
            self.alternative_paths,
            self.external_triggers,
            self.source,
            self.confidence,
            self.data_type
        ))

@dataclass
class Example:
    description: str
    source: Optional[str] = None

@dataclass
class TraitEvolution:
    period: str
    change: str
    trigger: Optional[str] = None

@dataclass
class BlindSpot:
    description: str
    overcome_strategy: Optional[str] = None
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass
class Trait:
    trait: str
    category: TraitCategory
    origin: TraitOrigin
    description: str
    examples: List[Example] = field(default_factory=list)
    evolution: List[TraitEvolution] = field(default_factory=list)
    blind_spots: List[BlindSpot] = field(default_factory=list)
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass
class Belief:
    belief: str
    rationale: Optional[str] = None
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass
class Quote:
    quote: str
    source: str
    context: Optional[str] = None

@dataclass
class Contradiction:
    description: str
    timeline_ref: Optional[str] = None

@dataclass
class Philosophy:
    principle: str
    quotes: List[Quote] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass
class Failure:
    event: str
    date: str
    description: str
    lessons_learned: List[str] = field(default_factory=list)
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass
class KeyDecision:
    decision: str
    date: str
    context: str
    impact: str
    reasoning: Optional[str] = None
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass
class Connection:
    name: str
    relationship: str
    impact: str
    period: Optional[str] = None
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass
class Habit:
    habit: str
    description: str
    impact: Optional[str] = None
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass
class UniqueApproach:
    approach_name: str
    description: str
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass
class Anecdote:
    title: str
    narrative: str
    moral: Optional[str] = None
    timeline_ref: Optional[str] = None
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL

@dataclass
class StressStrategy:
    strategy: str
    context: str
    effectiveness: Optional[str] = None

@dataclass
class LeadershipTrait:
    trait: str
    description: str
    examples: List[str] = field(default_factory=list)

@dataclass
class EmotionalIntelligence:
    source: str = ""
    confidence: float = 0.0
    data_type: DataType = DataType.FACTUAL
    stress_management: List[StressStrategy] = field(default_factory=list)
    leadership_style: List[LeadershipTrait] = field(default_factory=list)

@dataclass
class Metadata:
    last_updated: datetime
    version: str
    sources: List[str] = field(default_factory=list)

@dataclass
class Founder:
    """Main class representing a founder's complete profile"""
    basic_info: BasicInfo
    early_life: List[EarlyLifeDetail]
    timeline: List[TimelineEvent]
    traits: List[Trait]
    beliefs: List[Belief]
    philosophies: List[Philosophy]
    failures: List[Failure]
    key_decisions: List[KeyDecision]
    mentors_and_network: List[Connection]
    habits: List[Habit]
    unique_approaches: List[UniqueApproach]
    anecdotes: List[Anecdote]
    emotional_intelligence: EmotionalIntelligence
    metadata: Metadata

    def to_json(self) -> dict:
        """Convert to JSON-serializable dict"""
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, (list, tuple)):  # Handle both lists and tuples
                return [serialize(item) for item in obj]
            elif isinstance(obj, (TraitChange, TimelineEvent, Example, TraitEvolution,
                                BlindSpot, Quote, Contradiction, StressStrategy,
                                LeadershipTrait, Era)):  # Add all custom classes
                return {k: serialize(v) for k, v in obj.__dict__.items()}
            elif hasattr(obj, '__dict__'):
                return {k: serialize(v) for k, v in obj.__dict__.items()}
            return obj

        return serialize(self)

    @classmethod
    def from_json(cls, data: dict) -> "Founder":
        """Create from JSON data"""
        # TODO: Implement parsing of JSON into Founder instance
        pass