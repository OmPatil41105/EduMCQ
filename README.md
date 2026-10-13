# EduMCQ: Context-Aware Multiple-Choice Question & Distractor Generator

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-blue.svg)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HF-Transformers%20%26%20SentenceTransformers-blue)](https://huggingface.co/)
[![NLI Verification](https://img.shields.io/badge/NLI-Cross--Encoder-blue)](https://huggingface.co/cross-encoder/nli-distilroberta-base)

**EduMCQ** is a Natural Language Processing (NLP) system designed to automatically synthesize pedagogical Multiple-Choice Questions (MCQs) from technical documents, textbooks, and curriculum materials.

The system addresses the core challenge of Automated Distractor Generation (ADG): generating options that are *contextually plausible* yet *unambiguously false*. It integrates **WordNet Ontological Co-hyponym Mining**, **Dense Vector Semantic Embeddings (`all-MiniLM-L6-v2`)**, and **Natural Language Inference (NLI) Cross-Encoder Verification (`cross-encoder/nli-distilroberta-base`)** to eliminate false-positive distractors.

---

## System Architecture

```
                    +-------------------------------+
                    | Raw Input (PDF / DOCX / TXT)  |
                    +---------------+---------------+
                                    |
                         [Document Parser]
                         Paragraph & Sentences Tokenizer
                                    |
                                    v
                    +-------------------------------+
                    | Candidate Concept Extractor   |
                    | - Noun Phrase POS Chunking    |
                    | - Named Entity Recog (NER)    |
                    | - Salience Weighting          |
                    +---------------+---------------+
                                    |
         +--------------------------+--------------------------+
         v                                                     v
+-------------------------------+             +-------------------------------+
|     Question Generator        |             | Semantic Distractor Generator |
| - Syntactic Copula Inversion  |             | - WordNet Co-hyponym Sisters  |
| - Wh-Movement Framing         |             | - In-Document Concept Pool    |
| - T5 Seq2Seq Generation       |             | - Sentence-Transformer Cosine |
| - Answer-Leak Protection      |             | - Plausibility Window Filter  |
+---------------+---------------+             +---------------+---------------+
                |                                             |
                |                             +---------------v---------------+
                |                             |  NLI Cross-Encoder Validator  |
                |                             | - Premise / Hypothesis Test   |
                |                             | - Prunes Entailment (True Ans)|
                |                             | - Synthesizes Error Rationale |
                |                             +---------------+---------------+
                |                                             |
                +----------------------+----------------------+
                                       |
                                       v
                    +-------------------------------+
                    | Bloom's Taxonomy & Difficulty |
                    | - Remembering / Understanding |
                    | - Applying / Analyzing        |
                    | - Difficulty Index (1 - 10)   |
                    +---------------+---------------+
                                    |
                                    v
                    +-------------------------------+
                    |  Evaluation & Multi-Export    |
                    | - Distractor Plausibility(DPS)|
                    | - Distractor Diversity (DDI)  |
                    | - JSON, CSV, Markdown, GIFT   |
                    +-------------------------------+
```

---

## Core Capabilities

1. **Document Ingestion:** Multi-format parser for `.txt`, `.pdf` (`pypdf`), and `.docx` (`python-docx`) with paragraph-aware sentence tokenization.
2. **Concept Salience Engine:** Part-of-Speech regular expression chunking (`NP: {<JJ>* <NN>+}`) and Named Entity Recognition (NER) to isolate technical terms.
3. **Semantic Distractor Synthesis:**
   - **Ontological Coordinates:** WordNet co-hyponym extraction under shared hypernyms.
   - **Dense Embeddings:** `all-MiniLM-L6-v2` cosine similarity scoring.
   - **Plausibility Window:** Discards irrelevant outliers ($\text{Sim} < 0.20$) and ambiguous near-synonyms ($\text{Sim} > 0.85$).
4. **NLI Cross-Encoder Validation:**
   - Uses `cross-encoder/nli-distilroberta-base` to evaluate hypothesis entailment against the source premise.
   - Eliminates false-positive distractors entailed by the text.
   - Synthesizes pedagogical error rationales for incorrect options.
5. **Bloom's Taxonomy Classification:** Automated mapping to cognitive levels (*Remembering, Understanding, Applying, Analyzing*).
6. **Quantitative Evaluation:** Calculates Distractor Plausibility Score (DPS), Distractor Diversity Index (DDI), and Flesch Reading Ease.
7. **Standards-Compliant Export:** Generates Moodle GIFT, CSV, JSON, and Markdown formats.

---

## Installation

### Prerequisites
* Python 3.10 or higher
* Virtual environment (recommended)

```bash
cd /home/shahid/NLP
source venv/bin/activate
pip install -r requirements.txt
```

### Verification
Execute the automated test suite:
```bash
python run_tests.py
```
*(Confirms that all 8 unit and integration tests pass successfully)*

---

## Usage

### 1. Interactive Web Interface (Streamlit)
```bash
streamlit run app.py
```
Access `http://localhost:8501` to use the interactive assessment and diagnostic workbench.

### 2. Command Line Interface (CLI)
```bash
# Generate assessment from benchmark dataset:
python cli.py --sample os --num-questions 4

# Run with custom PDF document and export to Moodle GIFT:
python cli.py --input /path/to/curriculum.pdf --num-questions 5 --export-gift moodle_quiz.gift

# Fast mode without NLI validation:
python cli.py --sample bio --num-questions 3 --no-nli

# Interactive terminal assessment mode:
python cli.py --sample nlp --num-questions 3 --interactive
```

---

## Quantitative Evaluation Metrics

| Metric | Target Range | Description |
|---|---|---|
| **Distractor Plausibility Score (DPS)** | `0.50 - 0.75` | Mean dense embedding cosine similarity to target answer |
| **Distractor Diversity Index (DDI)** | `> 0.45` | Pairwise cosine distance among distractors (prevents redundancy) |
| **NLI Rejection Rate** | `10% - 25%` | Proportion of candidate distractors rejected due to contextual entailment |
| **Reading Ease Index (Flesch)** | `30 - 70` | Syntactic clarity and grammatical accessibility |
| **Quiz Quality Index (QQM)** | `> 65 / 100` | Composite assessment health index |

---

## Repository Structure

```
NLP/
├── data/
│   └── samples/
│       ├── operating_systems.txt
│       ├── cellular_respiration.txt
│       └── natural_language_processing.txt
├── edumcq/
│   ├── __init__.py
│   ├── config.py                # Hyperparameter configurations and thresholds
│   ├── schemas.py               # Dataclass schemas for questions and metrics
│   ├── document_parser.py       # File parsing and sentence segmentation
│   ├── concept_extractor.py     # POS chunking and NER candidate extraction
│   ├── question_generator.py    # Transformer QG and syntactic inversion
│   ├── distractor_generator.py  # WordNet co-hyponyms and SentenceTransformer ranking
│   ├── nli_validator.py         # NLI Cross-Encoder contradiction verification
│   ├── blooms_classifier.py     # Cognitive taxonomy and difficulty estimation
│   ├── evaluation.py            # DPS, DDI, and readability metrics
│   ├── exporter.py              # Multi-format LMS exporters
│   └── mcq_pipeline.py          # Unified execution pipeline
├── app.py                       # Streamlit Workbench Interface
├── cli.py                       # Command Line Interface
├── run_tests.py                 # Automated unit and integration tests
├── requirements.txt             # Dependency specifications
├── README.md                    # Project documentation
└── PBL_PROJECT_REPORT.md        # Academic project report
```

---

## Authors & Academic Context
* **Shahid** – Department of Computer Engineering
* *Course:* Natural Language Processing (PCCO403)
* *Activity:* Project-Based Learning (PBL Part B)
* *Institution:* Sanjivani College of Engineering, Kopargaon
