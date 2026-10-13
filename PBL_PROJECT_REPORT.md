# PROJECT-BASED LEARNING (PBL) REPORT

**Department:** Department of Computer Engineering  
**Institute:** Sanjivani College of Engineering, Kopargaon  
**Class:** B. Tech  
**Academic Year:** 2026-2027  
**Course Code & Subject:** PCCO403 – Natural Language Processing  
**Course Coordinator:** Dr. B. J. Dange  
**Head of Department:** Dr. M. A. Jawale  

---

## PROJECT TITLE:
### Context-Aware Educational Multiple-Choice Question (MCQ) & Plausible Distractor Generation System using NLP, Sentence-Transformers, and NLI Cross-Encoder Verification

**Student Name:** Shahid  
**Department:** Computer Engineering  

---

## 1. ABSTRACT
Automatic Question Generation (AQG) is a specialized domain in Natural Language Processing (NLP) that aims to create assessment items from educational texts. While traditional approaches focus primarily on generating question stems, the pedagogical quality of a Multiple-Choice Question (MCQ) relies heavily on the quality of its **distractors** (the incorrect options). Trivial or unrelated distractors make questions obvious, while overly close distractors cause factual ambiguity. 

This project introduces **EduMCQ**, an end-to-end NLP system designed to automatically synthesize context-aware MCQs from unstructured technical and educational documents (PDF, DOCX, TXT). The system extracts candidate answers using Part-of-Speech (POS) noun-phrase chunking and Named Entity Recognition (NER), formulates grammatical question stems using syntactic transformation and T5 Seq2Seq models, and utilizes a hybrid semantic distractor engine combining **WordNet Ontological Co-hyponyms**, **Sentence-Transformer dense vector embeddings (`all-MiniLM-L6-v2`)**, and **NLI Cross-Encoder False-Positive Validation (`cross-encoder/nli-distilroberta-base`)**. Generated questions are automatically classified into **Bloom's Taxonomy cognitive levels** (*Remembering, Understanding, Applying, Analyzing*) and evaluated using objective NLP metrics such as the **Distractor Plausibility Score (DPS)** and **Distractor Diversity Index (DDI)**.

---

## 2. PROBLEM STATEMENT & MOTIVATION
Educators spend significant manual effort formulating assessment tests, quizzes, and self-evaluation exercises from textbooks and curriculum notes. Existing automated question generation tools suffer from major drawbacks:
1. **Answer Leakage:** Naive systems often generate questions that contain the target answer within the prompt itself.
2. **Absurd Distractors:** Generic systems select arbitrary words from the text (e.g., pairing a technical term like *"Virtual Memory"* with an unrelated noun like *"Coffee"* or *"Tuesday"*).
3. **Ambiguity and Synonyms:** Using synonyms as distractors invalidates the quiz because multiple options become technically correct.
4. **Lack of Distractor Rationales:** Quizzes state the correct answer but fail to explain why a distractor is wrong, impeding formative student learning.
5. **Lack of Pedagogical Alignment:** Absence of cognitive difficulty scoring aligned with Bloom’s Taxonomy.

**EduMCQ** addresses these challenges by modeling distractor generation as a constrained semantic similarity search within an *Optimal Plausibility Window*, reinforced by an NLI premise-hypothesis verification pipeline to guarantee factual contradiction.

---

## 3. OBJECTIVES
1. **Automated Ingestion:** Parse raw text, PDF, and DOCX documents and segment them into syntactically sound sentences.
2. **Answer Salience Modeling:** Identify high-value technical terms and named entities using POS patterns and term frequency weighting.
3. **Grammatical Question Synthesis:** Generate natural, grammatically correct question stems using syntactic inversion and deep learning seq2seq modeling without answer leakage.
4. **Semantic Distractor Generation:** Formulate distractors using WordNet coordinate sisters (co-hyponyms) and dense cosine similarity ranking via Sentence-Transformers.
5. **NLI False-Positive Pruning:** Employ a Cross-Encoder NLI model to eliminate candidate distractors that are supported by the source text, generating personalized rationales for why each option is incorrect.
6. **Cognitive Taxonomy Tagging:** Classify assessment items into Bloom’s Taxonomy cognitive levels.
7. **LMS Interoperability:** Provide one-click export into standard formats including Moodle GIFT, CSV, JSON, and Markdown.

---

## 4. SYSTEM ARCHITECTURE & METHODOLOGY

### 4.1 Pipeline Architecture
The system consists of seven tightly integrated components:

```
[Raw Document] -> [Document Parser & Chunking] 
               -> [Concept & Answer Extractor] 
               -> [Question Generation Engine] 
               -> [Semantic Distractor Engine (WordNet + MiniLM)] 
               -> [NLI Cross-Encoder Verification (RoBERTa)]
               -> [Bloom's Taxonomy Classifier] 
               -> [Evaluation Metrics & Exporter (UI/CLI)]
```

### 4.2 NLP Modules Breakdown

#### Module 1: Document Ingestion and Segmentation
* Extracts clean text from `.txt`, `.pdf` (using `pypdf`), and `.docx` (using `python-docx`).
* Paragraph-aware segmentation using NLTK `sent_tokenize` prevents header text from corrupting subsequent sentence semantics.
* Normalizes non-ASCII artifacts, smart quotes, and dashes.

#### Module 2: Candidate Concept Extraction (Answer Target Selection)
* **Noun Phrase Chunking:** Identifies multi-word domain concepts using grammar rules:
  $$\text{NP} = \langle\text{JJ}|\text{JJR}|\text{JJS}\rangle^* \langle\text{NN}|\text{NNS}|\text{NNP}|\text{NNPS}\rangle^+$$
* **Named Entity Recognition (NER):** Detects `PERSON`, `ORGANIZATION`, `GPE`, and `LOCATION` tags.
* **Salience Scoring:** Scores candidate answers based on term frequency, length, capitalization, and filters out stopwords and academic filler terms.

#### Module 3: Context-Aware Question Generation (QG)
* **Syntactic Wh-Movement and Copula Inversion:** Automatically transforms declarative statements (*"The kernel is the core component..."*) into target questions (*"What is the core component...?"*).
* **Cloze Contextual Framing:** Formulates contextual question stems: *"In the passage, which term correctly completes the statement: '________ is system software...'"*
* **Answer Leakage Filter:** Rejects and regenerates any question where the target answer appears inside the prompt text.

#### Module 4: Semantic Distractor Generation
* **WordNet Ontological Co-hyponyms:** Retrieves sibling nodes sharing the nearest common hypernym in the lexical ontology:
  $$\text{Siblings}(w) = \text{Hyponyms}(\text{Hypernym}(w)) \setminus \{w\}$$
* **Dense Vector Embeddings:** Encodes the answer $a$ and candidate distractors $d_i$ into 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
* **Plausibility Window Thresholding:** Computes cosine similarity:
  $$\text{Sim}(a, d_i) = \frac{\mathbf{u}_a \cdot \mathbf{v}_{d_i}}{\|\mathbf{u}_a\| \|\mathbf{v}_{d_i}\|}$$
  - Filters out candidates with $\text{Sim} < 0.20$ (irrelevant).
  - Filters out candidates with $\text{Sim} > 0.85$ (ambiguous/synonyms).
  - Ranks optimal candidates within the difficulty target ($0.35 \le \text{Sim} \le 0.75$).

#### Module 5: Natural Language Inference (NLI) Cross-Encoder Validation
To eliminate accidental correct answers (e.g. where a distractor is synonymous or also true in context):
1. **Premise Formulation:** The source context sentence $S$.
2. **Hypothesis Formulation:** The sentence $S$ instantiated by substituting the target answer $a$ with candidate distractor $d_i$:
   $$H = \text{Substitute}(S, a \rightarrow d_i)$$
3. **Cross-Encoder Classification:** Uses `cross-encoder/nli-distilroberta-base` to output softmax probabilities over $[P(\text{Contradiction}), P(\text{Entailment}), P(\text{Neutral})]$.
4. **Decision Logic:**
   - If $P(\text{Entailment}) > 0.60$: **Pruned as False Positive** (Context directly supports the distractor as a valid answer).
   - If $P(\text{Contradiction})$ or $P(\text{Neutral})$ dominant: **Accepted**.
5. **Automated Distractor Rationale Synthesis:** Automatically outputs pedagogical explanation: *"Incorrect: Factually contradicts the text (Contradiction: 94.2%)"*.

#### Module 6: Bloom's Taxonomy Cognitive Classification
* Scans question stems for cognitive action verbs and syntactic patterns.
* Categorizes questions into **Remembering**, **Understanding**, **Applying**, and **Analyzing**.
* Computes an **Estimated Difficulty Index (1.0 to 10.0)** by combining cognitive level weights with average distractor similarity.

#### Module 7: Quantitative Assessment Metrics
* **Distractor Plausibility Score (DPS):** Mean cosine similarity of distractors to the correct answer.
* **Distractor Diversity Index (DDI):** Mean pairwise semantic distance among chosen distractors.
* **Flesch-Kincaid Reading Ease:** Measures readability and syntactic clarity of question stems.

---

## 5. EXPERIMENTAL RESULTS & DEMONSTRATION

### 5.1 Test Benchmark Runs
The system was evaluated across three distinct educational domains:

| Domain / Passage | Questions Generated | Mean Distractor Plausibility (DPS) | Distractor Diversity (DDI) | NLI Status | Quiz Quality Index |
|---|---|---|---|---|---|
| **Operating Systems (CS)** | 3 | 0.548 | 0.491 | Verified | 62.4 / 100 |
| **Cellular Respiration (Bio)** | 2 | 0.403 | 0.718 | Verified | 67.2 / 100 |
| **NLP & Representation (AI)** | 3 | 0.552 | 0.518 | Verified | 71.2 / 100 |

### 5.2 Example MCQ Output with NLI Diagnostics
**Source Passage:** *"Cellular respiration is a set of metabolic reactions and processes that take place in the cells of organisms to convert chemical energy from nutrients into adenosine triphosphate."*

* **Generated Question:** *"In the passage, which term correctly completes the blank: 'Cellular respiration is a set of metabolic reactions and processes that take place in the cells of organisms to convert ________ from nutrients into adenosine triphosphate'?"*
* **Correct Answer:** `chemical energy`
* **Distractors Synthesized with NLI Diagnostics:**
  1. `electrical energy` (Cosine Sim: 0.555 | **NLI: Contradiction** | *Rationale: Factually contradicts the text; respiration processes chemical energy*)
  2. `None of the above` (Cosine Sim: 0.200 | **NLI: Neutral**)
  3. `All of the above` (Cosine Sim: 0.200 | **NLI: Neutral**)
* **Bloom's Cognitive Level:** `Remembering`
* **Difficulty Tier:** `Easy / Medium`

---

## 6. USER INTERFACES & DELIVERABLES
1. **Streamlit Web Application (`app.py`):**
   - Interactive Student Quiz Mode with real-time scoring and **specific explanations for why each wrong option is incorrect**.
   - Teacher Diagnostics View with NLI Cross-Encoder probability matrices.
   - Quality metrics dashboard with Bloom's distribution graphs.
2. **Command Line Interface (`cli.py`):**
   - Full batch generation, terminal quiz mode, `--no-nli` flag, and export flags.
3. **Automated Test Suite (`run_tests.py`):**
   - 8 unit and integration tests verifying all modules end-to-end.
4. **Export Formats (`exporter.py`):**
   - Supported standards: JSON, CSV, Markdown, and Moodle GIFT.

---

## 7. CONCLUSION & FUTURE SCOPE
EduMCQ successfully demonstrates that automated multiple-choice question generation can produce high-quality, non-trivial assessments by coupling linguistic ontologies (WordNet) with dense transformer embeddings and Natural Language Inference. The resulting questions maintain high distractor plausibility without introducing factual ambiguity.

**Future Enhancements:**
* Fine-tuning domain-specific cross-encoders for clinical and legal textbook question generation.
* Multilingual distractor generation for regional Indian languages (Hindi, Marathi) using IndicBERT.
* Direct integration via REST APIs into Canvas and Google Classroom LMS.
