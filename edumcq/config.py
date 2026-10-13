"""
Configuration parameters for EduMCQ pipeline.
Centralizes all threshold hyperparameters, model identifiers, and taxonomy mappings.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict


@dataclass
class PipelineConfig:
    """Hyperparameters and model settings for the MCQ generation pipeline."""

    # Embedding Model Settings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # NLI Model Settings
    nli_model_name: str = "cross-encoder/nli-distilroberta-base"
    nli_entailment_threshold: float = 0.60  # Rejection threshold for false-positive distractors

    # Distractor Similarity Calibration Windows [min_similarity, max_similarity]
    similarity_windows: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "easy": (0.18, 0.50),
        "medium": (0.35, 0.75),
        "hard": (0.55, 0.88)
    })

    # Concept Extraction Filters
    min_phrase_len: int = 3
    max_phrase_words: int = 4
    min_sentence_chars: int = 25
    min_sentence_words: int = 5

    # Bloom's Taxonomy Weights
    bloom_level_weights: Dict[str, float] = field(default_factory=lambda: {
        "Remembering": 1.0,
        "Understanding": 2.0,
        "Applying": 3.0,
        "Analyzing": 4.0
    })


DEFAULT_CONFIG = PipelineConfig()
