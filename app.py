"""
EduMCQ Assessment Workbench.
Academic Streamlit Interface for Context-Aware Question and Distractor Generation
with NLI Cross-Encoder Verification and Pedagogical Diagnostics.
"""

import os
import streamlit as st
import pandas as pd
from typing import Dict, Any

from edumcq.mcq_pipeline import EduMCQPipeline
from edumcq.exporter import MCQExporter
from edumcq.document_parser import extract_text_from_pdf, extract_text_from_docx, clean_text

# Page Configuration - Clean, Institutional Styling
st.set_page_config(
    page_title="EduMCQ - Question and Distractor Generation Workbench",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic Theme
st.markdown("""
<style>
    .reportview-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .institution-tag {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .badge-correct {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-incorrect {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-neutral {
        background-color: #f8fafc;
        color: #334155;
        border: 1px solid #cbd5e1;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache pipeline loader
@st.cache_resource
def load_pipeline(use_transformer: bool = False, use_nli: bool = True, difficulty: str = "medium"):
    return EduMCQPipeline(
        use_transformer_qg=use_transformer,
        use_nli_validation=use_nli,
        difficulty=difficulty
    )

SAMPLES = {
    "Operating Systems (Computer Engineering)": "data/samples/operating_systems.txt",
    "Cellular Respiration (Life Sciences)": "data/samples/cellular_respiration.txt",
    "Natural Language Processing (AI and Computation)": "data/samples/natural_language_processing.txt"
}

# Sidebar: System Configuration
st.sidebar.markdown("### System Configuration")

input_method = st.sidebar.radio(
    "Corpus Source",
    ["Benchmark Dataset", "Raw Text Input", "Document File (.txt, .pdf, .docx)"]
)

raw_text = ""
if input_method == "Benchmark Dataset":
    sample_choice = st.sidebar.selectbox("Select Academic Corpus", list(SAMPLES.keys()))
    sample_path = SAMPLES[sample_choice]
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
elif input_method == "Raw Text Input":
    raw_text = st.sidebar.text_area(
        "Corpus Input Text",
        height=220,
        placeholder="Enter technical or curriculum passage for question synthesis..."
    )
else:
    uploaded_file = st.sidebar.file_uploader("Upload File", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext == ".txt":
            raw_text = clean_text(uploaded_file.read().decode("utf-8", errors="ignore"))
        elif file_ext == ".pdf":
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            raw_text = extract_text_from_pdf(tmp_path)
            os.unlink(tmp_path)
        elif file_ext == ".docx":
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            raw_text = extract_text_from_docx(tmp_path)
            os.unlink(tmp_path)

st.sidebar.markdown("---")
st.sidebar.markdown("### Generation Parameters")

num_questions = st.sidebar.slider("Target Question Count", min_value=2, max_value=8, value=4)
difficulty = st.sidebar.select_slider(
    "Distractor Difficulty Calibration",
    options=["easy", "medium", "hard"],
    value="medium",
    help="Modulates dense cosine similarity boundaries between target answer and candidate distractors."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### NLP Inference Engines")

enable_nli = st.sidebar.checkbox(
    "NLI Cross-Encoder Verification",
    value=True,
    help="Executes cross-encoder/nli-distilroberta-base to eliminate false-positive distractors entailed by context."
)

qg_engine = st.sidebar.selectbox(
    "Question Generation Mechanism",
    ["Syntactic Transformation Engine", "Flan-T5 Seq2Seq Transformer"]
)
use_transformer = (qg_engine == "Flan-T5 Seq2Seq Transformer")

generate_btn = st.sidebar.button("Run Assessment Pipeline", type="primary", use_container_width=True)

# Main Header
st.markdown('<div class="institution-tag">Department of Computer Engineering | Natural Language Processing (PCCO403)</div>', unsafe_allow_html=True)
st.markdown('<div class="header-title">EduMCQ: Context-Aware Assessment Generation Workbench</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-subtitle">'
    'Automated generation of Multiple-Choice Questions with WordNet ontological coordinate mining, '
    'Sentence-Transformer semantic calibration, and Natural Language Inference (NLI) contradiction verification.'
    '</div>',
    unsafe_allow_html=True
)

if "mcq_data" not in st.session_state:
    st.session_state.mcq_data = None
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if generate_btn:
    if not raw_text or len(raw_text.strip()) < 50:
        st.error("Invalid input: Provide sufficient passage length (minimum 3 complete sentences).")
    else:
        with st.spinner("Executing pipeline: concept extraction, semantic ranking, and NLI verification..."):
            pipeline = load_pipeline(use_transformer=use_transformer, use_nli=enable_nli, difficulty=difficulty)
            result = pipeline.generate_from_text(
                text=raw_text,
                num_questions=num_questions,
                difficulty=difficulty
            )

            if result.get("success"):
                st.session_state.mcq_data = result
                st.session_state.user_answers = {}
                st.session_state.quiz_submitted = False
                st.success(f"Pipeline executed successfully. Generated {result['total_generated']} assessment items.")
            else:
                st.error(f"Pipeline execution failure: {result.get('error')}")

# Output Section
if st.session_state.mcq_data:
    data = st.session_state.mcq_data
    mcqs = data.get("mcqs", [])
    metrics = data.get("metrics", {})

    # Top metrics bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Items", len(mcqs))
    with col2:
        st.metric("Mean Distractor Plausibility (DPS)", f"{metrics.get('mean_distractor_plausibility', 0):.3f}")
    with col3:
        st.metric("Distractor Diversity Index (DDI)", f"{metrics.get('mean_distractor_diversity', 0):.3f}")
    with col4:
        st.metric("Quality Rating", metrics.get("quality_rating", "Acceptable"))

    tab_assessment, tab_diagnostics, tab_metrics, tab_exports = st.tabs([
        "Assessment Execution",
        "Pedagogical Diagnostics",
        "Evaluation Metrics",
        "Export and Interoperability"
    ])

    # Tab 1: Assessment Execution
    with tab_assessment:
        st.markdown("#### Student Assessment Interface")
        st.caption("Standardized test delivery mode with immediate feedback and distractor error rationales.")

        letters = ["A", "B", "C", "D", "E"]
        with st.form("assessment_form"):
            for item in mcqs:
                qid = item["id"]
                st.markdown(f"**Item {qid}.** {item['question']}")
                st.caption(f"Taxonomy: {item['bloom_level']} | Tier: {item['difficulty_tier']}")

                opt_labels = [f"[{letters[i]}] {opt}" for i, opt in enumerate(item["options"])]
                selected = st.radio(
                    f"Selection for Item {qid}",
                    opt_labels,
                    key=f"q_{qid}",
                    label_visibility="collapsed"
                )
                st.session_state.user_answers[qid] = selected
                st.markdown("---")

            submit_assessment = st.form_submit_button("Submit Assessment", type="primary")

        if submit_assessment:
            st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:
            correct_count = 0
            st.markdown("#### Performance Evaluation & Item Review")

            for item in mcqs:
                qid = item["id"]
                user_choice = st.session_state.user_answers.get(qid, "")
                correct_letter = item["correct_option_letter"]
                correct_ans = item["correct_answer"]

                user_letter = user_choice.split("]")[0].replace("[", "").strip() if "]" in user_choice else ""
                user_text = user_choice.split("] ")[1].strip() if "] " in user_choice else user_choice
                is_correct = (user_letter == correct_letter)

                if is_correct:
                    correct_count += 1
                    status = '<span class="badge-correct">CORRECT</span>'
                else:
                    status = '<span class="badge-incorrect">INCORRECT</span>'

                st.markdown(f"**Item {qid}:** {item['question']} &nbsp; {status}", unsafe_allow_html=True)
                st.write(f"- Selected: `{user_choice}`")
                st.write(f"- Key: `[{correct_letter}] {correct_ans}`")

                if not is_correct:
                    explanation = item.get("option_explanations", {}).get(user_text, "Option is not asserted by context.")
                    st.warning(f"Distractor Analysis for '{user_text}': {explanation}")

                st.info(f"Source Context: \"{item['context_sentence']}\"")
                st.markdown("---")

            total_items = len(mcqs)
            percentage = (correct_count / total_items) * 100 if total_items else 0
            st.markdown(f"### Final Score: **{correct_count} / {total_items}** ({percentage:.1f}%)")

    # Tab 2: Pedagogical Diagnostics
    with tab_diagnostics:
        st.markdown("#### Item Diagnostics & Semantic Alignment")
        st.caption("Quantitative audit of target concepts, distractor candidates, and NLI verification scores.")

        for item in mcqs:
            with st.expander(f"Item {item['id']}: {item['question']} (Target: {item['correct_answer']})", expanded=True):
                st.markdown(f"**Bloom Taxonomy Level:** `{item['bloom_level']}` | **Difficulty Tier:** `{item['difficulty_tier']}` (Index: {item['difficulty_index']})")
                st.markdown(f"**Synthesis Method:** `{item['generation_method']}`")
                st.markdown(f"**Corpus Evidence:** *\"{item['context_sentence']}\"*")

                dist_table = [{
                    "Option Classification": "Ground Truth Target",
                    "Surface Form": item["correct_answer"],
                    "Cosine Similarity": 1.000,
                    "NLI Verdict": "Entailment (Affirmed)",
                    "Pedagogical Justification": "Explicitly affirmed in context sentence."
                }]

                for d in item.get("distractors", []):
                    dist_table.append({
                        "Option Classification": "Synthesized Distractor",
                        "Surface Form": d["text"],
                        "Cosine Similarity": d.get("similarity_score", 0.500),
                        "NLI Verdict": d.get("nli_verdict", "Contradiction"),
                        "Pedagogical Justification": d.get("rationale", "Non-entailed candidate.")
                    })

                st.dataframe(pd.DataFrame(dist_table), use_container_width=True)

    # Tab 3: Evaluation Metrics
    with tab_metrics:
        st.markdown("#### System Quantitative Evaluation")

        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown("##### Bloom's Taxonomy Cognitive Distribution")
            bloom_distribution = pd.Series([q["bloom_level"] for q in mcqs]).value_counts()
            st.bar_chart(bloom_distribution)

        with mcol2:
            st.markdown("##### Quality Metrics Summary")
            st.markdown(f"""
            - **Distractor Plausibility Score (DPS):** `{metrics.get('mean_distractor_plausibility', 0):.3f}`  
              *Mean cosine similarity of synthesized distractors relative to ground truth target vector.*
            - **Distractor Diversity Index (DDI):** `{metrics.get('mean_distractor_diversity', 0):.3f}`  
              *Pairwise semantic distance among wrong options to prevent redundant clusters.*
            - **NLI Cross-Encoder Verification:** `{'Active' if enable_nli else 'Disabled'}`  
              *Model: `cross-encoder/nli-distilroberta-base`. Threshold $\\tau = 0.60$.*
            - **Reading Ease (Flesch Index):** `{metrics.get('mean_readability_score', 0):.1f} / 100`  
              *Syntactic fluency and structural readability.*
            """)

    # Tab 4: Export and Interoperability
    with tab_exports:
        st.markdown("#### Standardized Export Formats")
        st.caption("Interoperable outputs for Learning Management Systems (LMS), research databases, and tabular analysis.")

        ecol1, ecol2, ecol3, ecol4 = st.columns(4)

        with ecol1:
            st.download_button(
                "Export JSON",
                data=MCQExporter.to_json(mcqs),
                file_name="assessment_schema.json",
                mime="application/json",
                use_container_width=True
            )

        with ecol2:
            st.download_button(
                "Export CSV",
                data=MCQExporter.to_csv(mcqs),
                file_name="assessment_items.csv",
                mime="text/csv",
                use_container_width=True
            )

        with ecol3:
            st.download_button(
                "Export Markdown",
                data=MCQExporter.to_markdown(mcqs),
                file_name="assessment_sheet.md",
                mime="text/markdown",
                use_container_width=True
            )

        with ecol4:
            st.download_button(
                "Export GIFT (Moodle LMS)",
                data=MCQExporter.to_gift(mcqs),
                file_name="moodle_assessment.gift",
                mime="text/plain",
                use_container_width=True
            )

        st.markdown("##### GIFT Format Representation")
        st.code(MCQExporter.to_gift(mcqs), language="markdown")

else:
    st.info("Select an academic corpus from the sidebar or provide custom text, then select 'Run Assessment Pipeline'.")

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("""
        **1. Target Extraction**  
        Extracts key domain concepts through Part-of-Speech noun phrase chunking, Named Entity Recognition, and frequency salience.
        """)
    with colB:
        st.markdown("""
        **2. NLI Distractor Synthesis**  
        Extracts coordinate terms via WordNet ontologies and dense vectors, filtered via Natural Language Inference cross-encoders.
        """)
    with colC:
        st.markdown("""
        **3. Pedagogical Taxonomy**  
        Maps questions against Bloom's cognitive taxonomy with multi-format LMS export (Moodle GIFT, JSON, CSV).
        """)
