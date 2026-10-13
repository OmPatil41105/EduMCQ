"""
Evaluation & Quality Scoring Module for Educational MCQs.
Computes objective NLP assessment metrics:
- Distractor Plausibility Score (DPS)
- Distractor Diversity Index (DDI)
- Fluency & Readability Index
- Overall Quiz Quality Score (QQM)
"""

import re
from typing import List, Dict, Any
import numpy as np


class MCQEvaluator:
    """Evaluates question quality, distractor plausibility, and linguistic metrics."""

    def __init__(self, distractor_generator=None):
        self.distractor_gen = distractor_generator

    def evaluate_quiz(self, mcqs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate evaluation metrics across an entire generated quiz."""
        if not mcqs:
            return {
                "total_questions": 0,
                "mean_plausibility_score": 0.0,
                "mean_diversity_index": 0.0,
                "mean_readability_score": 0.0,
                "overall_quality_rating": "N/A"
            }

        plausibility_scores = []
        diversity_indices = []
        readability_scores = []

        for item in mcqs:
            distractors = item.get("distractors", [])
            answer = item.get("correct_answer", "")
            question = item.get("question", "")

            # 1. Distractor Plausibility
            sims = [d.get("similarity_score", 0.5) for d in distractors if isinstance(d, dict)]
            if sims:
                plausibility_scores.append(float(np.mean(sims)))
            else:
                plausibility_scores.append(0.5)

            # 2. Distractor Diversity (Pairwise distance between distractors)
            d_texts = [d.get("text", "") if isinstance(d, dict) else str(d) for d in distractors]
            d_texts = [t for t in d_texts if t]
            div = self._calculate_diversity(d_texts)
            diversity_indices.append(div)

            # 3. Readability score of question stem
            readability = self._flesch_reading_ease(question)
            readability_scores.append(readability)

        mean_dps = float(np.mean(plausibility_scores)) if plausibility_scores else 0.0
        mean_ddi = float(np.mean(diversity_indices)) if diversity_indices else 0.0
        mean_read = float(np.mean(readability_scores)) if readability_scores else 0.0

        # Composite Quality Index (0-100)
        # Optimal DPS: ~0.55-0.70; High DDI: >0.50; Good Readability: 50-80
        quality_score = (
            (min(1.0, mean_dps / 0.65) * 40.0) +
            (min(1.0, mean_ddi / 0.60) * 30.0) +
            (min(1.0, mean_read / 70.0) * 30.0)
        )
        quality_score = min(100.0, max(10.0, quality_score))

        rating = "Excellent" if quality_score >= 80 else ("Good" if quality_score >= 65 else "Acceptable")

        return {
            "total_questions": len(mcqs),
            "mean_distractor_plausibility": round(mean_dps, 3),
            "mean_distractor_diversity": round(mean_ddi, 3),
            "mean_readability_score": round(mean_read, 1),
            "quiz_quality_index": round(quality_score, 1),
            "quality_rating": rating
        }

    def _calculate_diversity(self, texts: List[str]) -> float:
        """Compute average pairwise distance between distractor texts."""
        if len(texts) < 2:
            return 1.0

        if self.distractor_gen and self.distractor_gen.embedder:
            try:
                embs = self.distractor_gen.embedder.encode(texts, normalize_embeddings=True)
                n = len(texts)
                dists = []
                for i in range(n):
                    for j in range(i + 1, n):
                        cos_sim = float(np.dot(embs[i], embs[j]))
                        dists.append(1.0 - cos_sim)
                return float(np.mean(dists)) if dists else 0.5
            except Exception:
                pass

        # Lexical fallback diversity (Jaccard dissimilarity)
        n = len(texts)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                s1 = set(texts[i].lower().split())
                s2 = set(texts[j].lower().split())
                union = s1.union(s2)
                inter = s1.intersection(s2)
                jaccard = len(inter) / len(union) if union else 1.0
                dists.append(1.0 - jaccard)
        return float(np.mean(dists)) if dists else 0.5

    def _flesch_reading_ease(self, text: str) -> float:
        """Flesch-Kincaid Reading Ease approximation."""
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return 70.0

        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        num_sentences = max(1, len(sentences))

        def count_syllables(word):
            w = word.lower()
            count = len(re.findall(r'[aeiouy]+', w))
            if w.endswith('e') and not w.endswith('le') and len(w) > 2:
                count = max(1, count - 1)
            return max(1, count)

        total_syllables = sum(count_syllables(w) for w in words)
        words_per_sent = len(words) / num_sentences
        syllables_per_word = total_syllables / len(words)

        score = 206.835 - (1.015 * words_per_sent) - (84.6 * syllables_per_word)
        return max(0.0, min(100.0, score))
