"""
Question Generation Module.
Synthesizes natural, direct Multiple-Choice Question (MCQ) stems without fill-in-the-blanks,
without passage citations ("in the passage"), without dangling prepositions, and with answer-leakage protection.

Supported Mechanisms:
1. Linguistic Syntactic Transformation (Grammatical Copula Inversion, Role Framing, Wh-Movement)
2. Seq2Seq Transformer (Flan-T5)
"""

import re
import logging
from typing import Optional, Dict, Any, List, Tuple
import nltk

logger = logging.getLogger("edumcq.question_generator")


class QuestionGenerator:
    """Generates direct, natural conceptual MCQ stems from target concepts and context sentences."""

    def __init__(self, use_transformer: bool = False, model_name: str = "google/flan-t5-small"):
        self.use_transformer = use_transformer
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.model = None

    def _init_transformer(self):
        """Lazy loader for transformer model."""
        if self.pipeline is not None or not self.use_transformer:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            logger.info(f"Loading Transformer QG Model ({self.model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.pipeline = True
            logger.info("Transformer Model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}. Falling back to syntactic engine.")
            self.use_transformer = False
            self.pipeline = None

    def generate_question(self, concept: str, sentence: str, context: Optional[str] = None, entity_type: str = "NP") -> Dict[str, Any]:
        """
        Generate a direct conceptual multiple-choice question stem for the specified concept.
        Returns a dictionary containing the question, method used, and target answer.
        """
        concept_clean = concept.strip()
        sentence_clean = sentence.strip()

        # Try Transformer if enabled
        if self.use_transformer:
            self._init_transformer()
            if self.pipeline:
                transformer_q = self._generate_with_transformer(concept_clean, sentence_clean, context)
                if transformer_q and self._validate_question(transformer_q, concept_clean):
                    return {
                        "question": transformer_q,
                        "method": "transformer_seq2seq",
                        "answer": concept_clean
                    }

        # Syntactic & Linguistic Transformation
        rule_q = self._generate_conceptual_question(concept_clean, sentence_clean, entity_type)
        return {
            "question": self._clean_question(rule_q),
            "method": "syntactic_transformation",
            "answer": concept_clean
        }

    def _generate_with_transformer(self, concept: str, sentence: str, context: Optional[str] = None) -> Optional[str]:
        """Generate question using Flan-T5 seq2seq prompt."""
        try:
            import torch
            ctx = sentence if not context else context
            prompt = (
                f"Generate a direct multiple choice exam question where the correct answer is '{concept}'. "
                f"Do not use fill in the blanks. Do not say 'in the passage'.\n"
                f"Context: {ctx}\n"
                f"Exam Question:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return self._clean_question(generated)
        except Exception:
            return None

    def _generate_conceptual_question(self, concept: str, sentence: str, entity_type: str) -> str:
        """
        Synthesizes direct exam-style conceptual questions:
        - Never uses fill-in-the-blanks ('________')
        - Never quotes the passage ('In the passage, which completes...')
        - Converts statements into active interrogative forms ('Which mechanism...', 'What is defined as...')
        """
        s = sentence.strip().rstrip(".!?")
        c_esc = re.escape(concept)

        # Case A: Target Concept is the SUBJECT at sentence start
        # "The kernel is the core component..." or "Virtual memory is a memory management capability..."
        match_subj_is = re.match(
            rf'^(?:The\s+|A\s+|An\s+)?{c_esc}\s+(?:is|are|was|were|refers to|represents)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z\s\-]+?)\s+(?:that|which|by which|to|used to)\s+(.*)$',
            s,
            re.IGNORECASE
        )
        if match_subj_is:
            category = match_subj_is.group(1).strip()
            rest = match_subj_is.group(2).strip()
            words = category.split()
            if 1 <= len(words) <= 4 and not any(w.lower() in ["one", "it", "this", "something"] for w in words):
                return f"Which {category} {rest}?"

        # Case B: Target Concept is the SUBJECT with general copula definition
        match_copula_subj = re.match(
            rf'^(?:The\s+|A\s+|An\s+)?{c_esc}\s+(is|are|was|were)\s+(.*)$',
            s,
            re.IGNORECASE
        )
        if match_copula_subj:
            copula = match_copula_subj.group(1)
            predicate = match_copula_subj.group(2).strip()
            clean_pred = re.sub(r'^(a|an|the)\s+', '', predicate, flags=re.IGNORECASE)
            return f"Which component or mechanism {copula} defined as {clean_pred}?"

        # Case C: Target Concept is in the PREDICATE after copula:
        # "Subject is a/an [Concept] that/which/by which [Rest]"
        # Example: "Virtual memory is a memory management capability that provides..." (Target: memory management capability)
        match_pred_concept = re.match(
            rf'^(?:The\s+|A\s+|An\s+)?([a-zA-Z\s\-]+?)\s+(?:is|are|was|were)\s+(?:a\s+|an\s+|the\s+)?{c_esc}\s+(?:that|which|by which|used to)\s+(.*)$',
            s,
            re.IGNORECASE
        )
        if match_pred_concept:
            subject = match_pred_concept.group(1).strip()
            rest = match_pred_concept.group(2).strip()
            return f"What is {subject} characterized as that {rest}?"

        # Case D: Target Concept preceded by Preposition (by / in / into / from / through / inside)
        # Example: "...managed independently by an operating system scheduler." (Target: operating system scheduler)
        match_prep = re.search(
            rf'\b(by|in|inside|within|through|from|into|with)\s+(?:a\s+|an\s+|the\s+)?{c_esc}\b',
            s,
            re.IGNORECASE
        )
        if match_prep:
            prep = match_prep.group(1).lower()
            prefix = s[:match_prep.start()].strip()
            # Clean leading conjunctions from prefix
            prefix = re.sub(r'^(in contrast,\s*|furthermore,\s*|moreover,\s*|additionally,\s*)', '', prefix, flags=re.IGNORECASE)
            if prep == "by":
                return f"Which component or entity {self._convert_passive_to_active(prefix)}?"
            elif prep in ["in", "inside", "within"]:
                match_subj = re.match(r'^(?:The\s+|A\s+|An\s+)?([a-zA-Z\s\-]+?)\s+(?:is|are|was|were)\b', prefix, re.IGNORECASE)
                if match_subj:
                    main_s = match_subj.group(1).strip()
                    if any(term in prefix.lower() for term in ["located", "found", "situated", "stored"]):
                        return f"Where is {main_s} located?"
                    else:
                        return f"In which structure or region is {main_s} found?"
                return f"Where {self._invert_subject_verb(prefix)}?"
            elif prep == "into":
                return f"What is produced when {prefix}?"
            elif prep == "through":
                return f"Through which mechanism {self._invert_subject_verb(prefix)}?"

        # Case E: Functional / Active Action Pattern with Target as Subject
        match_action = re.match(
            rf'^(?:The\s+|A\s+|An\s+)?{c_esc}\s+(is used to|serves to|functions to|enables|provides|manages|controls|performs|converts|generates|produces)\s+(.*)$',
            s,
            re.IGNORECASE
        )
        if match_action:
            verb = match_action.group(1)
            rest = match_action.group(2).strip()
            if verb.startswith("is"):
                return f"Which component {verb} {rest}?"
            else:
                return f"Which mechanism or component {verb} {rest}?"

        # Case F: Event / Condition Pattern
        # "A page fault occurs when a program attempts..."
        match_event = re.match(
            rf'^(?:The\s+|A\s+|An\s+)?{c_esc}\s+(occurs|happens|arises|is triggered|takes place)\s+(.*)$',
            s,
            re.IGNORECASE
        )
        if match_event:
            verb = match_event.group(1)
            condition = match_event.group(2).strip()
            return f"What {verb} {condition}?"

        # Case G: Conversion / Output Pattern
        match_convert_into = re.search(
            rf'(converts?|transforms?|produces?|generates?)\s+(.*?)\s+(?:into|to form)\s+(?:a\s+|an\s+|the\s+)?{c_esc}',
            s,
            re.IGNORECASE
        )
        if match_convert_into:
            verb = match_convert_into.group(1)
            source_sub = match_convert_into.group(2).strip()
            return f"What is produced when {source_sub} is {verb}ed?"

        # Case H: Entity-specific queries
        if "PERSON" in entity_type:
            return f"Who is credited with the development or formulation described in: {self._strip_target(s, concept)}?"
        elif "LOCATION" in entity_type or "GPE" in entity_type:
            return f"In which location or environment is {self._strip_target(s, concept)} observed?"

        # Case I: Clean Conceptual Inversion Fallback (No blanks, no citations!)
        sentence_without_target = self._strip_target(s, concept)
        if sentence_without_target:
            sentence_without_target = re.sub(r'^(and|or|but|when|where|while|which)\s+', '', sentence_without_target, flags=re.IGNORECASE)
            match_subj_clause = re.match(r'^(?:The\s+|A\s+|An\s+)?([a-zA-Z\s\-]+?)\s+(?:is|are)\s+.*?\s+to\s+([a-zA-Z]+)\s+(.*)', sentence_without_target, re.IGNORECASE)
            if match_subj_clause:
                topic = match_subj_clause.group(1).strip()
                action = match_subj_clause.group(2).strip()
                rest = match_subj_clause.group(3).strip()
                base_act = action[:-1] if action.endswith("e") else action
                return f"In {topic}, what is {base_act}ed {rest}?"

            return f"Which of the following is characterized by: {sentence_without_target}?"

        return f"Which option defines the concept associated with {s}?"

    def _convert_passive_to_active(self, text: str) -> str:
        """Convert passive clause like 'a thread can be managed independently' into active verb phrasing."""
        text = re.sub(r'^(in contrast,\s*|however,\s*)', '', text, flags=re.IGNORECASE).strip()
        # Case 1: Relative clause: "MainSubj is ... that can be/is [verb] [adverb]"
        match_rel = re.search(
            r'(?:a\s+|an\s+|the\s+)?([a-zA-Z\s\-]+?)\s+(?:is|are|was|were)\s+.*?\s+(?:that|which)\s+(?:can be|is|are|was|were)\s+([a-zA-Z]+ed)\s*(.*)',
            text,
            flags=re.IGNORECASE
        )
        if match_rel:
            main_subj = match_rel.group(1).strip()
            verb = match_rel.group(2).strip()
            base = verb[:-1] if verb.endswith("ed") else verb
            active_verb = base + ("s" if not base.endswith("s") else "")
            rest = match_rel.group(3).strip()
            return f"{active_verb} {main_subj} {rest}".strip()

        # Case 2: Simple passive: "Subj is/can be [verb] [adverb]"
        match_passive = re.search(
            r'^(.*?)\s+(?:is|are|can be|was|were)\s+([a-zA-Z]+ed)\s*(.*)',
            text,
            flags=re.IGNORECASE
        )
        if match_passive:
            subj = match_passive.group(1).strip()
            verb = match_passive.group(2).strip()
            base = verb[:-1] if verb.endswith("ed") else verb
            active_verb = base + ("s" if not base.endswith("s") else "")
            rest = match_passive.group(3).strip()
            return f"{active_verb} {subj} {rest}".strip()

        return f"operates on {text}"

    def _invert_subject_verb(self, text: str) -> str:
        """Convert declarative statement prefix into an interrogative clause."""
        words = text.split()
        if not words:
            return "does this process occur"
        if words[-1].endswith("s"):
            verb = words[-1]
            base_verb = verb[:-1]
            subject = " ".join(words[:-1])
            return f"does {subject} {base_verb}"
        return f"is {text} situated"

    def _strip_target(self, sentence: str, target: str) -> str:
        """Remove target answer smoothly from sentence clause without leaving dangling articles/prepositions."""
        # Strip preceding article + target
        cleaned = re.sub(rf'\b(a|an|the)\s+{re.escape(target)}\b', '', sentence, count=1, flags=re.IGNORECASE)
        # If not matched, strip target alone
        cleaned = re.sub(rf'\b{re.escape(target)}\b', '', cleaned, count=1, flags=re.IGNORECASE)

        # Collapse whitespace
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        cleaned = re.sub(r'\s+([,.;])', r'\1', cleaned).strip()

        # Remove dangling prepositions or articles at the end of the clause
        cleaned = re.sub(r'\s+(by|in|at|from|to|for|with|of|an|a|the)$', '', cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def _validate_question(self, question: str, concept: str) -> bool:
        """Verify that question does not contain the answer and is well-formed."""
        if not question or len(question.split()) < 4:
            return False

        q_lower = question.lower()
        c_lower = concept.lower()

        # Check for answer leakage
        if re.search(rf'\b{re.escape(c_lower)}\b', q_lower):
            return False

        # Reject fill in the blanks
        if "____" in question or "blank" in q_lower or "in the passage" in q_lower:
            return False

        return True

    def _clean_question(self, question: str) -> str:
        """Standardize question formatting and syntax."""
        q = question.strip()
        if not q:
            return "Which option represents the correct concept?"

        # Remove stray quotes and underscores
        q = q.replace('"', '').replace("'", "")
        q = re.sub(r'_+', '', q)

        # Remove dangling articles or prepositions before question mark
        q = re.sub(r'\s+(by|in|at|from|to|for|with|of|an|a|the)\?*$', '', q, flags=re.IGNORECASE)

        # Normalize whitespace
        q = re.sub(r'\s{2,}', ' ', q).strip()

        # Capitalize first letter
        q = q[0].upper() + q[1:]

        # Ensure termination with question mark
        q = re.sub(r'[\s.,;:]+$', '', q)
        if not q.endswith('?'):
            q += '?'

        return q
