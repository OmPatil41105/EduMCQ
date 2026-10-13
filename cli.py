"""
Command Line Interface (CLI) for EduMCQ Generator.
Executes batch MCQ generation, multi-format export, and standardized assessment evaluation.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any

from edumcq.mcq_pipeline import EduMCQPipeline
from edumcq.exporter import MCQExporter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def print_header():
    print("-" * 72)
    print("EduMCQ: Context-Aware Question & Distractor Generation Pipeline")
    print("Natural Language Processing (PCCO403) | Department of Computer Engineering")
    print("-" * 72)


def run_interactive_assessment(mcqs: list):
    """Run interactive terminal assessment evaluation."""
    print("\n" + "=" * 60)
    print("INTERACTIVE ASSESSMENT MODE")
    print("=" * 60 + "\n")

    letters = ["A", "B", "C", "D", "E"]
    score = 0

    for idx, item in enumerate(mcqs, 1):
        print(f"\n[Item {idx}/{len(mcqs)}] (Tier: {item.get('difficulty_tier')} | Taxonomy: {item.get('bloom_level')})")
        print(f"{item.get('question')}\n")

        options = item.get('options', [])
        for opt_idx, opt in enumerate(options):
            letter = letters[opt_idx] if opt_idx < len(letters) else str(opt_idx + 1)
            print(f"   [{letter}] {opt}")

        user_choice = ""
        valid_letters = [letters[i] for i in range(len(options))]
        while user_choice not in valid_letters:
            user_choice = input("\nSelect answer (A/B/C/D): ").strip().upper()

        chosen_idx = letters.index(user_choice)
        chosen_text = options[chosen_idx]
        correct_letter = item.get("correct_option_letter")

        if user_choice == correct_letter:
            print("Verdict: CORRECT (+1)")
            score += 1
        else:
            print(f"Verdict: INCORRECT. Correct answer: [{correct_letter}] {item.get('correct_answer')}")
            exp = item.get("option_explanations", {}).get(chosen_text, "Option is not supported.")
            print(f"Distractor Analysis: {exp}")

        print(f"Source Context: \"{item.get('context_sentence')}\"")

    print("\n" + "=" * 60)
    pct = (score / len(mcqs)) * 100 if mcqs else 0
    print(f"Assessment Result: {score}/{len(mcqs)} ({pct:.1f}%)")
    print("=" * 60 + "\n")


def main():
    print_header()

    parser = argparse.ArgumentParser(
        description="EduMCQ: Context-Aware Multiple-Choice Question and Distractor Generator"
    )
    parser.add_argument("-i", "--input", help="Path to input document (.txt, .pdf, .docx)")
    parser.add_argument("-s", "--sample", choices=["os", "bio", "nlp"], default=None, help="Benchmark sample identifier (os, bio, nlp)")
    parser.add_argument("-n", "--num-questions", type=int, default=4, help="Number of questions to synthesize (default: 4)")
    parser.add_argument("-o", "--num-options", type=int, default=4, help="Number of options per question (default: 4)")
    parser.add_argument("-d", "--difficulty", choices=["easy", "medium", "hard"], default="medium", help="Distractor difficulty calibration (default: medium)")
    parser.add_argument("--transformer", action="store_true", help="Enable Seq2Seq Transformer model for QG")
    parser.add_argument("--no-nli", action="store_true", help="Disable NLI Cross-Encoder distractor validation")
    parser.add_argument("--interactive", action="store_true", help="Run interactive terminal assessment mode")
    parser.add_argument("--export-json", help="Path to save output JSON")
    parser.add_argument("--export-csv", help="Path to save output CSV")
    parser.add_argument("--export-md", help="Path to save output Markdown")
    parser.add_argument("--export-gift", help="Path to save output GIFT (Moodle)")

    args = parser.parse_args()

    file_path = args.input
    if args.sample:
        sample_map = {
            "os": "data/samples/operating_systems.txt",
            "bio": "data/samples/cellular_respiration.txt",
            "nlp": "data/samples/natural_language_processing.txt"
        }
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, sample_map[args.sample])

    if not file_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "data/samples/operating_systems.txt")

    if not os.path.exists(file_path):
        print(f"Error: Specified file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    use_nli = not args.no_nli
    print(f"Input Corpus: {file_path}")
    print(f"Configuration: {args.num_questions} items | {args.num_options} options | Difficulty: {args.difficulty.upper()} | NLI: {'Enabled' if use_nli else 'Disabled'}")

    pipeline = EduMCQPipeline(
        use_transformer_qg=args.transformer,
        use_nli_validation=use_nli,
        difficulty=args.difficulty
    )

    result = pipeline.generate_from_file(
        file_path=file_path,
        num_questions=args.num_questions,
        num_options=args.num_options,
        difficulty=args.difficulty
    )

    if not result.get("success"):
        print(f"Pipeline error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)

    mcqs = result.get("mcqs", [])
    metrics = result.get("metrics", {})

    print(f"\nGenerated {len(mcqs)} assessment items.\n")

    print("=" * 72)
    print("ASSESSMENT ITEMS OVERVIEW")
    print("=" * 72)
    letters = ["A", "B", "C", "D", "E"]
    for q in mcqs:
        print(f"\n[Item {q['id']}] {q['question']}")
        print(f"        Taxonomy: {q['bloom_level']} | Difficulty Tier: {q['difficulty_tier']}")
        for i, opt in enumerate(q['options']):
            key_tag = " [KEY]" if opt == q['correct_answer'] else ""
            print(f"        {letters[i]}. {opt}{key_tag}")
        diag = [f"{d['text']} (Cosine: {d.get('similarity_score', 0):.3f}, NLI: {d.get('nli_verdict', 'Contradiction')})" for d in q['distractors']]
        print(f"        Distractor Diagnostics: {diag}")

    print("\n" + "=" * 72)
    print("QUANTITATIVE NLP METRICS")
    print("=" * 72)
    print(f"Mean Distractor Plausibility (DPS): {metrics.get('mean_distractor_plausibility', 0.0):.3f} (Standard benchmark: 0.50 - 0.75)")
    print(f"Distractor Diversity Index (DDI):    {metrics.get('mean_distractor_diversity', 0.0):.3f} (Standard benchmark: > 0.45)")
    print(f"Flesch Reading Ease Index:          {metrics.get('mean_readability_score', 0.0):.1f} / 100")
    print(f"Quiz Quality Index:                 {metrics.get('quiz_quality_index', 0.0):.1f} / 100 [{metrics.get('quality_rating')}]")
    print("=" * 72)

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            f.write(MCQExporter.to_json(mcqs))
        print(f"Exported JSON: {args.export_json}")

    if args.export_csv:
        with open(args.export_csv, "w", encoding="utf-8") as f:
            f.write(MCQExporter.to_csv(mcqs))
        print(f"Exported CSV: {args.export_csv}")

    if args.export_md:
        with open(args.export_md, "w", encoding="utf-8") as f:
            f.write(MCQExporter.to_markdown(mcqs, title=os.path.basename(file_path)))
        print(f"Exported Markdown: {args.export_md}")

    if args.export_gift:
        with open(args.export_gift, "w", encoding="utf-8") as f:
            f.write(MCQExporter.to_gift(mcqs))
        print(f"Exported GIFT (Moodle): {args.export_gift}")

    if args.interactive:
        run_interactive_assessment(mcqs)


if __name__ == "__main__":
    main()
