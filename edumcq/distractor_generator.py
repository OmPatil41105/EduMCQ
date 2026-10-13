"""
Semantic Distractor Generation Engine.
Generates plausible, high-quality, and non-obvious incorrect options (distractors) using:
1. WordNet Ontological Hierarchy (Co-hyponym mining under common hypernyms)
2. In-document semantic entity pool mining
3. Dense vector embeddings via SentenceTransformer (all-MiniLM-L6-v2)
4. Cosine similarity thresholding for difficulty calibration (Easy, Medium, Hard)
5. Context contradiction & synonym filtering
"""

import re
import logging
from typing import List, Dict, Any, Set, Optional, Tuple
import numpy as np
from nltk.corpus import wordnet as wn
from .config import DEFAULT_CONFIG

logger = logging.getLogger("edumcq.distractor_generator")


class DistractorGenerator:
    """Generates and ranks plausible distractors for target answers."""

    def __init__(self, embedding_model_name: Optional[str] = None):
        self.embedding_model_name = embedding_model_name or DEFAULT_CONFIG.embedding_model_name
        self.embedder = None
        self._init_embedder()

        # Domain taxonomies for fallback candidate generation
        self.domain_clusters = {
            "cs": [
                "CPU", "RAM", "ROM", "Cache Memory", "Virtual Memory", "Registers",
                "Operating System", "Compiler", "Interpreter", "Kernel", "Device Driver",
                "Process", "Thread", "Deadlock", "Semaphore", "Paging", "Segmentation",
                "Stack", "Queue", "Binary Tree", "Graph", "Hash Table", "Linked List",
                "TCP", "UDP", "IP Address", "DNS", "HTTP", "Router", "Firewall",
                "Supervised Learning", "Unsupervised Learning", "Neural Network",
                "Decision Tree", "Support Vector Machine", "Gradient Descent"
            ],
            "biology": [
                "Mitochondria", "Ribosome", "Nucleus", "Chloroplast", "Golgi Apparatus",
                "Endoplasmic Reticulum", "Lysosome", "Vacuole", "Cell Membrane",
                "Photosynthesis", "Cellular Respiration", "Mitosis", "Meiosis",
                "Enzyme", "Protein", "Lipid", "Carbohydrate", "Nucleic Acid",
                "DNA", "RNA", "Ribose", "ATP", "Glucose"
            ],
            "general_stem": [
                "Kinetic Energy", "Potential Energy", "Thermal Energy", "Electromagnetic Force",
                "Gravity", "Friction", "Acceleration", "Velocity", "Mass", "Density"
            ]
        }

    def _init_embedder(self):
        """Initialize SentenceTransformer embedder."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            logger.warning(f"SentenceTransformer initialization failed: {e}. Falling back to ontological baseline.")
            self.embedder = None

    def get_wordnet_cohyponyms(self, word: str, max_candidates: int = 15) -> List[str]:
        """
        Extract coordinate terms (co-hyponyms) from WordNet.
        e.g., word='Jupiter' -> hypernym='planet' -> hyponyms=['Mars', 'Saturn', 'Venus']
        """
        candidates: Set[str] = set()
        clean_word = word.lower().replace(" ", "_")

        synsets = wn.synsets(clean_word)
        if not synsets:
            tokens = clean_word.split("_")
            if len(tokens) > 1:
                synsets = wn.synsets(tokens[-1])

        for syn in synsets[:3]:
            for hypernym in syn.hypernyms()[:2]:
                for sibling in hypernym.hyponyms():
                    for lemma in sibling.lemmas():
                        candidate = lemma.name().replace("_", " ")
                        if candidate.lower() != word.lower() and word.lower() not in candidate.lower():
                            candidates.add(candidate)
                            if len(candidates) >= max_candidates:
                                return list(candidates)

        return list(candidates)

    def harvest_candidates(self, target_answer: str, doc_concepts: Optional[List[str]] = None) -> List[str]:
        """Harvest candidate distractors across WordNet, document pool, and domain clusters."""
        candidate_pool: Set[str] = set()
        ans_lower = target_answer.lower()

        # 1. WordNet co-hyponyms
        wn_candidates = self.get_wordnet_cohyponyms(target_answer)
        for c in wn_candidates:
            if c.lower() != ans_lower:
                candidate_pool.add(c)

        # 2. In-document concept pool
        if doc_concepts:
            for c in doc_concepts:
                c_clean = c.strip()
                if c_clean.lower() != ans_lower and len(c_clean) > 2:
                    if c_clean.lower() not in ans_lower and ans_lower not in c_clean.lower():
                        candidate_pool.add(c_clean)

        # 3. Domain clusters
        for cluster in self.domain_clusters.values():
            for item in cluster:
                if item.lower() != ans_lower and item.lower() not in ans_lower:
                    candidate_pool.add(item)

        return list(candidate_pool)

    def rank_distractors(
        self,
        target_answer: str,
        candidates: List[str],
        sentence_context: str,
        num_distractors: int = 3,
        difficulty: str = "medium"
    ) -> List[Dict[str, Any]]:
        """
        Rank candidate distractors using dense cosine similarity with SentenceTransformer.
        Calibrates selection against difficulty windows.
        """
        if not candidates:
            return []

        min_sim, max_sim = DEFAULT_CONFIG.similarity_windows.get(
            difficulty.lower(), (0.35, 0.75)
        )

        clean_candidates = []
        ans_lower = target_answer.lower()

        for c in candidates:
            c_str = c.strip()
            c_lower = c_str.lower()
            if len(c_str) < 2 or c_lower == ans_lower:
                continue
            if re.search(rf'\b{re.escape(c_lower)}\b', sentence_context.lower()):
                continue
            clean_candidates.append(c_str)

        if not clean_candidates:
            return []

        scored_candidates: List[Tuple[str, float]] = []

        if self.embedder is not None:
            try:
                ans_emb = self.embedder.encode([target_answer], normalize_embeddings=True)[0]
                cand_embs = self.embedder.encode(clean_candidates, normalize_embeddings=True)
                similarities = np.dot(cand_embs, ans_emb)

                for cand, sim in zip(clean_candidates, similarities):
                    scored_candidates.append((cand, float(sim)))
            except Exception as e:
                logger.warning(f"Embedding calculation fallback: {e}")
                scored_candidates = [(c, 0.5) for c in clean_candidates]
        else:
            scored_candidates = [(c, 0.5) for c in clean_candidates]

        ideal_sim = (min_sim + max_sim) / 2.0

        def rank_score(item: Tuple[str, float]) -> float:
            _, sim = item
            if sim < min_sim:
                return -abs(sim - min_sim) * 2
            elif sim > max_sim:
                return -abs(sim - max_sim) * 3
            else:
                return 1.0 - abs(sim - ideal_sim)

        scored_candidates.sort(key=rank_score, reverse=True)

        selected: List[Dict[str, Any]] = []
        seen_stems: Set[str] = set()
        is_title = target_answer[0].isupper()

        for cand, sim in scored_candidates:
            norm_cand = cand.lower()
            if norm_cand in seen_stems:
                continue
            seen_stems.add(norm_cand)

            formatted_cand = cand.title() if is_title else cand

            selected.append({
                "text": formatted_cand,
                "similarity_score": round(sim, 3),
                "plausibility": "High" if sim >= 0.55 else ("Moderate" if sim >= 0.35 else "Fair")
            })

            if len(selected) >= num_distractors:
                break

        return selected
