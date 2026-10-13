"""
Document Parser and Text Preprocessing Module.
Handles document ingestion (.txt, .pdf, .docx), cleaning, paragraph-aware sentence segmentation,
and coherent context window chunking.
"""

import os
import re
from typing import List, Dict, Any, Optional
import nltk


def clean_text(text: str) -> str:
    """Normalize whitespace, quotes, dashes, and remove non-printable artifacts."""
    if not text:
        return ""
    # Standardize curly quotes and apostrophes
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u201C\u201D]', '"', text)
    # Standardize dashes
    text = re.sub(r'[\u2013\u2014]', '-', text)
    # Remove control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    # Collapse spaces within lines
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
    cleaned = '\n'.join(lines)
    # Remove repeated blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    extracted_pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_pages.append(page_text)
    return clean_text("\n\n".join(extracted_pages))


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from Word document using python-docx."""
    import docx
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n\n".join(paragraphs))


def parse_file(file_path: str) -> str:
    """
    Parse a document from file path based on its extension.
    Supports .txt, .pdf, .docx.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return clean_text(f.read())
    elif ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return clean_text(f.read())


def segment_sentences(text: str) -> List[str]:
    """
    Segment input text into clean, grammatically complete sentences.
    Splits across paragraph boundaries first to prevent header bleed-over.
    Filters out headers, fragmented lines, and incomplete clauses.
    """
    text = clean_text(text)
    if not text:
        return []

    # Split into lines/paragraphs first
    raw_blocks = [b.strip() for b in text.split('\n') if b.strip()]

    valid_sentences = []
    for block in raw_blocks:
        try:
            block_sents = nltk.sent_tokenize(block)
        except Exception:
            block_sents = re.split(r'(?<=[.!?])\s+', block)

        for s in block_sents:
            s = s.strip()
            words = s.split()
            # Must be at least 25 characters and at least 5 words
            if len(s) >= 25 and len(words) >= 5:
                # Must contain alphabetic characters and look like a full sentence
                if any(w.isalpha() for w in words):
                    # Filter out pure headers lacking terminal punctuation if too short
                    if not s.endswith(('.', '!', '?')) and len(words) < 8:
                        continue
                    valid_sentences.append(s)

    return valid_sentences


def create_context_windows(sentences: List[str], max_words: int = 160, overlap_sentences: int = 1) -> List[Dict[str, Any]]:
    """
    Create overlapping context chunks from sentences to feed into Question Generators.
    """
    if not sentences:
        return []

    chunks = []
    i = 0
    while i < len(sentences):
        current_chunk_sentences = []
        current_word_count = 0
        j = i

        while j < len(sentences):
            sent = sentences[j]
            words = sent.split()
            if current_word_count + len(words) > max_words and len(current_chunk_sentences) > 0:
                break
            current_chunk_sentences.append(sent)
            current_word_count += len(words)
            j += 1

        context_text = " ".join(current_chunk_sentences)
        chunks.append({
            "chunk_index": len(chunks),
            "context": context_text,
            "sentences": current_chunk_sentences,
            "start_idx": i,
            "end_idx": j
        })

        if j >= len(sentences):
            break
        step = max(1, len(current_chunk_sentences) - overlap_sentences)
        i += step

    return chunks
