"""
Data schemas and typing structures for EduMCQ.
Provides formal type definitions for questions, distractors, and evaluation metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DistractorOption:
    text: str
    similarity_score: float
    plausibility: str
    nli_verdict: Optional[str] = None
    contradiction_score: Optional[float] = None
    entailment_score: Optional[float] = None
    neutral_score: Optional[float] = None
    rationale: Optional[str] = None


@dataclass
class QuestionItem:
    id: int
    question: str
    options: List[str]
    correct_option_letter: str
    correct_option_index: int
    correct_answer: str
    distractors: List[Dict[str, Any]]
    option_explanations: Dict[str, str]
    bloom_level: str
    difficulty_tier: str
    difficulty_index: float
    generation_method: str
    context_sentence: str
    explanation: str


@dataclass
class QuizEvaluationMetrics:
    total_questions: int
    mean_distractor_plausibility: float
    mean_distractor_diversity: float
    mean_readability_score: float
    quiz_quality_index: float
    quality_rating: str
