"""
Bloom's Taxonomy Cognitive Classification Module.
Classifies generated assessment questions into pedagogical cognitive levels:
- Level 1: Remembering (Recall facts & basic definitions)
- Level 2: Understanding (Explain concepts & describe processes)
- Level 3: Applying (Execute procedures & utilize concepts)
- Level 4: Analyzing (Differentiate, organize & relate components)
Provides difficulty scoring and cognitive indicators.
"""

import re
from typing import Dict, Any, List


class BloomsClassifier:
    """Classifies questions into Bloom's Taxonomy cognitive domains and computes difficulty."""

    def __init__(self):
        # Flexible regex patterns supporting standard verb inflections (-s, -ed, -ing, -ation)
        self.taxonomy_keywords = {
            "Remembering": [
                r"\bidentif\w*", r"\bname\w*", r"\brecall\w*", r"\bdefin\w*", r"\bstate\w*",
                r"\bwhich term\b", r"\bwhat is\b", r"\bwhat are\b", r"\blabel\w*", r"\blist\w*",
                r"\bcompletes the statement\b", r"\bwhich concept\b"
            ],
            "Understanding": [
                r"\bdescrib\w*", r"\bexplain\w*", r"\bsummariz\w*", r"\binterpret\w*",
                r"\bwhich describes\b", r"\bwhat role\b", r"\bwhy does\b", r"\bclarif\w*",
                r"\bwhich statement\b", r"\bhow does\b", r"\bfunction of\b", r"\bpurpose of\b"
            ],
            "Applying": [
                r"\buse\w*", r"\bexecut\w*", r"\bappl\w*", r"\bimplement\w*", r"\bsolv\w*",
                r"\bcalculat\w*", r"\bdemonstrat\w*", r"\billustrat\w*"
            ],
            "Analyzing": [
                r"\bdifferentiat\w*", r"\bdistinguish\w*", r"\bcompar\w*", r"\bcontrast\w*",
                r"\bexamin\w*", r"\banalyz\w*", r"\bcategoriz\w*", r"\bwhich differs\b"
            ]
        }

    def classify_question(self, question: str, target_answer: str, mean_distractor_sim: float = 0.5) -> Dict[str, Any]:
        """Classify a question stem into a Bloom's level and compute composite difficulty."""
        q_lower = question.lower()

        scores = {level: 0 for level in self.taxonomy_keywords}
        for level, patterns in self.taxonomy_keywords.items():
            for pat in patterns:
                matches = re.findall(pat, q_lower)
                scores[level] += len(matches)

        # Prioritize higher cognitive orders if matched
        # (e.g. if 'explains' is matched, Understanding > Remembering default)
        max_level = "Remembering"
        max_score = 0
        for level in ["Analyzing", "Applying", "Understanding", "Remembering"]:
            if scores[level] > max_score:
                max_score = scores[level]
                max_level = level

        if max_score == 0:
            if "why" in q_lower or "how" in q_lower:
                max_level = "Understanding"
            else:
                max_level = "Remembering"

        # Difficulty computation based on cognitive level and distractor similarity
        level_weights = {
            "Remembering": 1.0,
            "Understanding": 2.0,
            "Applying": 3.0,
            "Analyzing": 4.0
        }
        cog_score = level_weights.get(max_level, 1.0)

        # Composite difficulty (1.0 to 10.0 scale)
        composite = (cog_score * 1.5) + (mean_distractor_sim * 4.0)
        if composite < 4.0:
            difficulty_tier = "Easy"
        elif composite < 6.8:
            difficulty_tier = "Medium"
        else:
            difficulty_tier = "Hard"

        return {
            "bloom_level": max_level,
            "difficulty_tier": difficulty_tier,
            "cognitive_score": round(cog_score, 1),
            "estimated_difficulty_index": round(min(10.0, max(1.0, composite)), 2),
            "rationale": f"Classified as '{max_level}' based on linguistic cues; distractor similarity avg = {mean_distractor_sim:.2f}"
        }
