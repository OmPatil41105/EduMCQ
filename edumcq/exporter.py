"""
Export and Output Formatting Module.
Exports generated MCQs to multiple industry and academic formats:
- JSON (Structured data for APIs / Web apps)
- CSV (Spreadsheet / Excel compatible)
- Markdown (Formatted student quiz & teacher answer key)
- GIFT (Moodle / Canvas LMS Import Standard)
"""

import json
import csv
import io
from typing import List, Dict, Any


class MCQExporter:
    """Handles multi-format quiz exporting."""

    @staticmethod
    def to_json(mcqs: List[Dict[str, Any]], indent: int = 2) -> str:
        """Export to formatted JSON string."""
        return json.dumps(mcqs, indent=indent, ensure_ascii=False)

    @staticmethod
    def to_csv(mcqs: List[Dict[str, Any]]) -> str:
        """Export to CSV format."""
        output = io.StringIO()
        fieldnames = [
            "question_id", "question", "option_a", "option_b", "option_c", "option_d",
            "correct_option", "correct_answer", "bloom_level", "difficulty_tier", "context_sentence"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(mcqs, 1):
            options = item.get("options", [])
            opt_map = {0: "", 1: "", 2: "", 3: ""}
            for i, opt in enumerate(options[:4]):
                opt_map[i] = opt

            writer.writerow({
                "question_id": idx,
                "question": item.get("question", ""),
                "option_a": opt_map[0],
                "option_b": opt_map[1],
                "option_c": opt_map[2],
                "option_d": opt_map[3],
                "correct_option": item.get("correct_option_letter", ""),
                "correct_answer": item.get("correct_answer", ""),
                "bloom_level": item.get("bloom_level", "Remembering"),
                "difficulty_tier": item.get("difficulty_tier", "Medium"),
                "context_sentence": item.get("context_sentence", "")
            })

        return output.getvalue()

    @staticmethod
    def to_markdown(mcqs: List[Dict[str, Any]], title: str = "Automated NLP Assessment Quiz") -> str:
        """Export to clean Markdown student quiz and teacher solution sheet."""
        lines = [f"# {title}\n", f"**Total Questions:** {len(mcqs)}\n", "---\n"]

        # Student Section
        lines.append("## Part 1: Student Assessment Sheet\n")
        letters = ["A", "B", "C", "D"]

        for idx, item in enumerate(mcqs, 1):
            lines.append(f"### Question {idx}")
            lines.append(f"**{item.get('question', '')}**\n")
            for i, opt in enumerate(item.get("options", [])):
                letter = letters[i] if i < len(letters) else str(i + 1)
                lines.append(f"- [{letter}] {opt}")
            lines.append("")

        lines.append("---\n")
        # Teacher Key Section
        lines.append("## Part 2: Instructor Answer Key & NLP Diagnostics\n")
        for idx, item in enumerate(mcqs, 1):
            letter = item.get("correct_option_letter", "")
            ans = item.get("correct_answer", "")
            bloom = item.get("bloom_level", "")
            diff = item.get("difficulty_tier", "")
            ctx = item.get("context_sentence", "")
            method = item.get("generation_method", "")

            lines.append(f"**Q{idx} Answer:** `{letter}. {ans}`")
            lines.append(f"- **Cognitive Level (Bloom):** {bloom} | **Difficulty:** {diff}")
            lines.append(f"- **QG Method:** {method}")
            lines.append(f"- **Source Evidence:** *\"{ctx}\"*")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_gift(mcqs: List[Dict[str, Any]]) -> str:
        """
        Export to GIFT format for direct Moodle / Canvas LMS quiz import.
        Syntax: ::Question Title:: Question text {~WrongOption =CorrectOption ~WrongOption}
        """
        gift_lines = []
        for idx, item in enumerate(mcqs, 1):
            q_text = item.get("question", "").replace("{", "\\{").replace("}", "\\}")
            options = item.get("options", [])
            correct_ans = item.get("correct_answer", "")

            opt_gift = []
            for opt in options:
                clean_opt = opt.replace("{", "\\{").replace("}", "\\}").replace("=", "\\=").replace("~", "\\~")
                if opt.strip().lower() == correct_ans.strip().lower():
                    opt_gift.append(f"={clean_opt}")
                else:
                    opt_gift.append(f"~{clean_opt}")

            gift_block = f"::Q{idx}:: {q_text} {{\n  " + "\n  ".join(opt_gift) + "\n}\n"
            gift_lines.append(gift_block)

        return "\n".join(gift_lines)
