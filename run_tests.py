"""
Test Suite for EduMCQ Generator.
Executes unit and integration tests across all pipeline modules including NLI verification.
"""

import sys
import unittest

from edumcq.document_parser import clean_text, segment_sentences, create_context_windows
from edumcq.concept_extractor import ConceptExtractor
from edumcq.question_generator import QuestionGenerator
from edumcq.distractor_generator import DistractorGenerator
from edumcq.nli_validator import NLIDistractorValidator
from edumcq.blooms_classifier import BloomsClassifier
from edumcq.exporter import MCQExporter
from edumcq.mcq_pipeline import EduMCQPipeline


class TestEduMCQ(unittest.TestCase):

    def setUp(self):
        self.sample_passage = (
            "An operating system is system software that manages computer hardware and software resources. "
            "The kernel is the core component of an operating system that maintains complete control over everything in the system. "
            "Virtual memory is a memory management capability that provides an idealized abstraction of storage resources. "
            "A page fault occurs when a program attempts to access a block of memory not stored in RAM."
        )

    def test_document_parser(self):
        cleaned = clean_text("  This is a   test text with \u201cquotes\u201d.  \n\n\nNext line.  ")
        self.assertIn('"quotes"', cleaned)
        sentences = segment_sentences(self.sample_passage)
        self.assertGreaterEqual(len(sentences), 3)
        chunks = create_context_windows(sentences, max_words=60)
        self.assertGreater(len(chunks), 0)

    def test_concept_extractor(self):
        extractor = ConceptExtractor()
        sentences = segment_sentences(self.sample_passage)
        concepts = extractor.rank_concepts(self.sample_passage, sentences)
        self.assertGreater(len(concepts), 0)

        concept_names = [c["concept"].lower() for c in concepts]
        found_key = any("operating system" in c or "kernel" in c or "virtual memory" in c for c in concept_names)
        self.assertTrue(found_key, f"Expected key concepts not found in: {concept_names}")

    def test_question_generator(self):
        qg = QuestionGenerator(use_transformer=False)
        sentence = "Virtual memory is a memory management capability that provides an idealized abstraction of storage resources."
        q_data = qg.generate_question(concept="Virtual memory", sentence=sentence, entity_type="NP")

        question = q_data["question"]
        self.assertTrue(question.endswith("?"))
        self.assertFalse("virtual memory is" in question.lower())
        self.assertNotIn("____", question)
        self.assertNotIn("in the passage", question.lower())
        self.assertNotIn("blank", question.lower())

    def test_distractor_generator(self):
        dg = DistractorGenerator()
        candidates = dg.harvest_candidates("CPU", doc_concepts=["RAM", "Kernel", "Process"])
        self.assertGreater(len(candidates), 0)

        distractors = dg.rank_distractors(
            target_answer="CPU",
            candidates=candidates,
            sentence_context="The CPU executes machine instructions.",
            num_distractors=3,
            difficulty="medium"
        )
        self.assertGreaterEqual(len(distractors), 2)
        for d in distractors:
            self.assertIn("text", d)
            self.assertIn("similarity_score", d)
            self.assertNotEqual(d["text"].lower(), "cpu")

    def test_nli_validator(self):
        validator = NLIDistractorValidator(enabled=True)
        # Test validation on contradiction
        premise = "The kernel is the core component of an operating system."
        res = validator.validate_distractor(
            premise_sentence=premise,
            target_answer="kernel",
            distractor_text="compiler"
        )
        self.assertTrue(res["is_valid"])
        self.assertIn("verdict", res)
        self.assertIn("rationale", res)

    def test_blooms_classifier(self):
        classifier = BloomsClassifier()
        res1 = classifier.classify_question("What is defined as virtual memory?", "virtual memory", 0.6)
        self.assertEqual(res1["bloom_level"], "Remembering")

        res2 = classifier.classify_question("Which statement explains how the kernel manages memory?", "kernel", 0.7)
        self.assertEqual(res2["bloom_level"], "Understanding")

    def test_exporter(self):
        mock_mcq = [{
            "id": 1,
            "question": "What is the core component of an operating system?",
            "options": ["Kernel", "Compiler", "Browser", "Editor"],
            "correct_option_letter": "A",
            "correct_answer": "Kernel",
            "bloom_level": "Remembering",
            "difficulty_tier": "Medium",
            "context_sentence": "The kernel is the core component of an operating system."
        }]

        json_out = MCQExporter.to_json(mock_mcq)
        self.assertIn("Kernel", json_out)

        csv_out = MCQExporter.to_csv(mock_mcq)
        self.assertIn("question_id", csv_out)

        md_out = MCQExporter.to_markdown(mock_mcq)
        self.assertIn("# Automated NLP Assessment Quiz", md_out)

        gift_out = MCQExporter.to_gift(mock_mcq)
        self.assertIn("=Kernel", gift_out)

    def test_end_to_end_pipeline(self):
        pipeline = EduMCQPipeline(use_transformer_qg=False, use_nli_validation=True, difficulty="medium")
        res = pipeline.generate_from_text(self.sample_passage, num_questions=3, num_options=4)

        self.assertTrue(res["success"])
        self.assertEqual(len(res["mcqs"]), 3)
        for q in res["mcqs"]:
            self.assertEqual(len(q["options"]), 4)
            self.assertIn(q["correct_answer"], q["options"])
            self.assertIn("option_explanations", q)
            self.assertIn("metrics", res)
            self.assertGreater(res["metrics"]["quiz_quality_index"], 0)


if __name__ == "__main__":
    unittest.main()
