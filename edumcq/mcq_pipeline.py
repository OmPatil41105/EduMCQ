"""
Unified MCQ Generation Pipeline.
Orchestrates:
1. Document text parsing and sentence chunking
2. Informative concept and answer extraction
3. Question generation (Syntactic or Transformer Seq2Seq)
4. Semantic distractor harvesting, embedding ranking, and filtering
5. NLI Cross-Encoder contradiction verification & distractor rationale generation
6. Option randomization and correct option labeling (A, B, C, D)
7. Bloom's taxonomy and difficulty assessment
8. Overall quiz quality metrics calculation
"""

import random
from typing import List, Dict, Any, Optional
from .document_parser import parse_file, clean_text, segment_sentences, create_context_windows
from .concept_extractor import ConceptExtractor
from .question_generator import QuestionGenerator
from .distractor_generator import DistractorGenerator
from .nli_validator import NLIDistractorValidator
from .blooms_classifier import BloomsClassifier
from .evaluation import MCQEvaluator


class EduMCQPipeline:
    """End-to-end pipeline for generating educational multiple-choice assessments."""

    def __init__(
        self,
        use_transformer_qg: bool = False,
        use_nli_validation: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        difficulty: str = "medium"
    ):
        self.difficulty = difficulty
        self.concept_extractor = ConceptExtractor()
        self.question_generator = QuestionGenerator(use_transformer=use_transformer_qg)
        self.distractor_generator = DistractorGenerator(embedding_model_name=embedding_model)
        self.nli_validator = NLIDistractorValidator(enabled=use_nli_validation)
        self.blooms_classifier = BloomsClassifier()
        self.evaluator = MCQEvaluator(distractor_generator=self.distractor_generator)

    def generate_from_text(
        self,
        text: str,
        num_questions: int = 5,
        num_options: int = 4,
        difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate MCQs from raw input text string.
        """
        diff = difficulty if difficulty else self.difficulty
        cleaned = clean_text(text)
        sentences = segment_sentences(cleaned)

        if not sentences:
            return {
                "success": False,
                "error": "The provided text does not contain sufficient complete sentences.",
                "mcqs": [],
                "metrics": {}
            }

        # 1. Concept Extraction & Salience Ranking
        ranked_concepts = self.concept_extractor.rank_concepts(cleaned, sentences)
        if not ranked_concepts:
            return {
                "success": False,
                "error": "Could not identify informative answer concepts in the provided text.",
                "mcqs": [],
                "metrics": {}
            }

        # Extract all concept texts for in-document candidate distractor pool
        doc_concept_pool = [c["concept"] for c in ranked_concepts]

        # 2. Process candidates into questions
        generated_mcqs: List[Dict[str, Any]] = []
        letters = ["A", "B", "C", "D", "E"]

        for cand in ranked_concepts:
            if len(generated_mcqs) >= num_questions:
                break

            target_answer = cand["concept"]
            target_sentence = cand["sentence"]
            entity_type = cand["type"]

            # Generate Question Stem
            q_res = self.question_generator.generate_question(
                concept=target_answer,
                sentence=target_sentence,
                entity_type=entity_type
            )
            question_stem = q_res["question"]

            # Generate Plausible Distractors (requesting extra candidates for NLI filtering)
            num_distractors_needed = max(1, num_options - 1)
            distractor_candidates = self.distractor_generator.harvest_candidates(
                target_answer=target_answer,
                doc_concepts=doc_concept_pool
            )

            # Retrieve extra candidates (e.g. needed + 3) to allow NLI filtration
            ranked_distractors = self.distractor_generator.rank_distractors(
                target_answer=target_answer,
                candidates=distractor_candidates,
                sentence_context=target_sentence,
                num_distractors=num_distractors_needed + 3,
                difficulty=diff
            )

            # Apply NLI Cross-Encoder Validation
            validated_distractors = self.nli_validator.filter_and_annotate(
                premise_sentence=target_sentence,
                target_answer=target_answer,
                candidates=ranked_distractors
            )

            # Trim to required count
            final_distractors = validated_distractors[:num_distractors_needed]

            # If fewer distractors remaining after NLI, pad with fallback
            if len(final_distractors) < num_distractors_needed:
                fallback_pool = [
                    "None of the above", "All of the above", "Not defined in the passage",
                    "Alternative mechanism", "Secondary component"
                ]
                for fb in fallback_pool:
                    if fb not in [d["text"] for d in final_distractors]:
                        final_distractors.append({
                            "text": fb,
                            "similarity_score": 0.2,
                            "plausibility": "Fair",
                            "nli_verdict": "Neutral",
                            "rationale": "General negative choice."
                        })
                        if len(final_distractors) >= num_distractors_needed:
                            break

            # Calculate average distractor similarity for Bloom's difficulty weighting
            sim_scores = [d.get("similarity_score", 0.5) for d in final_distractors]
            avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0.5

            # Bloom's Taxonomy Classification
            bloom_info = self.blooms_classifier.classify_question(
                question=question_stem,
                target_answer=target_answer,
                mean_distractor_sim=avg_sim
            )

            # Format Options & Shuffle
            distractor_texts = [d["text"] for d in final_distractors]
            all_opts = [target_answer] + distractor_texts
            random.shuffle(all_opts)

            # Find correct option index and letter
            correct_idx = all_opts.index(target_answer)
            correct_letter = letters[correct_idx] if correct_idx < len(letters) else "A"

            # Build comprehensive option explanations dictionary (Why each option is right or wrong)
            option_explanations = {}
            for opt in all_opts:
                if opt == target_answer:
                    option_explanations[opt] = f"Correct: Stated in source passage: \"{target_sentence}\""
                else:
                    match_dist = next((d for d in final_distractors if d["text"] == opt), None)
                    if match_dist and "rationale" in match_dist:
                        option_explanations[opt] = f"Incorrect: {match_dist['rationale']}"
                    else:
                        option_explanations[opt] = "Incorrect: Plausible distractor not supported as the primary answer."

            mcq_item = {
                "id": len(generated_mcqs) + 1,
                "question": question_stem,
                "options": all_opts,
                "correct_option_letter": correct_letter,
                "correct_option_index": correct_idx,
                "correct_answer": target_answer,
                "distractors": final_distractors,
                "option_explanations": option_explanations,
                "bloom_level": bloom_info["bloom_level"],
                "difficulty_tier": bloom_info["difficulty_tier"],
                "difficulty_index": bloom_info["estimated_difficulty_index"],
                "generation_method": q_res["method"],
                "context_sentence": target_sentence,
                "explanation": f"Based on the text: \"{target_sentence}\""
            }
            generated_mcqs.append(mcq_item)

        # 3. Overall Evaluation Metrics
        quiz_metrics = self.evaluator.evaluate_quiz(generated_mcqs)

        return {
            "success": True,
            "total_generated": len(generated_mcqs),
            "difficulty_setting": diff,
            "mcqs": generated_mcqs,
            "metrics": quiz_metrics
        }

    def generate_from_file(
        self,
        file_path: str,
        num_questions: int = 5,
        num_options: int = 4,
        difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate MCQs from a document file (.txt, .pdf, .docx).
        """
        text = parse_file(file_path)
        return self.generate_from_text(
            text=text,
            num_questions=num_questions,
            num_options=num_options,
            difficulty=difficulty
        )
