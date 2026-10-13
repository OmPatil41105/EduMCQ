"""
Concept and Candidate Answer Extraction Module.
Identifies high-value educational target answers using:
- Named Entity Recognition (NER)
- Part-Of-Speech (POS) Noun Phrase Chunking
- TF-IDF and Frequency Salience Scoring
- Stopword, length, and triviality filters
"""

import re
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
import nltk
from nltk.corpus import stopwords


class ConceptExtractor:
    """Extracts informative answer candidates from educational text."""

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()

        self.academic_blacklist = {
            "example", "examples", "case", "cases", "study", "studies", "fact",
            "time", "way", "ways", "thing", "things", "result", "results",
            "number", "numbers", "information", "detail", "details", "process",
            "system", "type", "types", "method", "methods", "section", "part",
            "point", "points", "level", "levels", "step", "steps",
            "year", "years", "day", "days", "problem", "problems", "approach"
        }

    def _clean_candidate(self, phrase: str) -> str:
        """Strip surrounding punctuation and extraneous articles."""
        phrase = phrase.strip().strip(".,;:\"'()[]{}")
        # Strip leading and trailing articles
        phrase = re.sub(r'^(the|a|an)\s+', '', phrase, flags=re.IGNORECASE)
        phrase = re.sub(r'\s+(the|a|an)$', '', phrase, flags=re.IGNORECASE)
        return phrase.strip()

    def _is_valid_candidate(self, phrase: str) -> bool:
        """Check if candidate phrase is non-trivial and grammatically valid."""
        phrase = self._clean_candidate(phrase)
        if not phrase:
            return False

        words = phrase.split()
        if len(words) == 0 or len(words) > 4:
            return False

        # Reject single character words or pure numbers
        if len(phrase) < 3 or phrase.isdigit():
            return False

        # Reject all stopwords
        lower_phrase = phrase.lower()
        if lower_phrase in self.stop_words or lower_phrase in self.academic_blacklist:
            return False

        # If all individual words are stopwords
        if all(w.lower() in self.stop_words for w in words):
            return False

        # Reject phrases with strange punctuation
        if re.search(r'[^a-zA-Z0-9\s\-]', phrase):
            return False

        return True

    def extract_named_entities(self, sentence: str) -> List[Tuple[str, str]]:
        """Extract named entities (PERSON, ORGANIZATION, GPE, etc.) using NLTK."""
        entities = []
        try:
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            tree = nltk.ne_chunk(tagged)

            for subtree in tree:
                if hasattr(subtree, 'label'):
                    entity_name = " ".join([token for token, tag in subtree.leaves()])
                    entity_type = subtree.label()
                    clean_name = self._clean_candidate(entity_name)
                    if self._is_valid_candidate(clean_name):
                        entities.append((clean_name, entity_type))
        except Exception:
            pass
        return entities

    def extract_noun_phrases(self, sentence: str) -> List[Tuple[str, str]]:
        """Extract multi-word and single-word noun phrases via POS regex chunking."""
        noun_phrases = []
        try:
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)

            # Grammar for extracting Noun Phrases:
            grammar = r"""
                NP: {<JJ|JJR|JJS>* <NN|NNS|NNP|NNPS>+}
            """
            chunk_parser = nltk.RegexpParser(grammar)
            tree = chunk_parser.parse(tagged)

            for subtree in tree:
                if hasattr(subtree, 'label') and subtree.label() == 'NP':
                    phrase = " ".join([token for token, tag in subtree.leaves()])
                    clean_phrase = self._clean_candidate(phrase)
                    pos_pattern = "-".join([tag for token, tag in subtree.leaves()])
                    if self._is_valid_candidate(clean_phrase):
                        noun_phrases.append((clean_phrase, f"NP:{pos_pattern}"))
        except Exception:
            pass
        return noun_phrases

    def rank_concepts(self, doc_text: str, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Extract and rank answer concepts across the document.
        Combines frequency, capitalized prominence, entity tags, and position.
        """
        all_candidates: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, int]] = set()

        # Build document-level frequency for salience weighting
        tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', doc_text)]
        word_freq = Counter(tokens)

        for sent_idx, sentence in enumerate(sentences):
            # Extract Named Entities
            ne_list = self.extract_named_entities(sentence)
            for entity_text, entity_type in ne_list:
                key = (entity_text.lower(), sent_idx)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    freq = sum(word_freq[w.lower()] for w in entity_text.split())
                    score = freq * 2.0 + 3.0
                    all_candidates.append({
                        "concept": entity_text,
                        "type": entity_type,
                        "sentence_idx": sent_idx,
                        "sentence": sentence,
                        "score": score
                    })

            # Extract Noun Phrases
            np_list = self.extract_noun_phrases(sentence)
            for np_text, np_type in np_list:
                key = (np_text.lower(), sent_idx)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    words = np_text.split()
                    freq = sum(word_freq[w.lower()] for w in words)

                    length_bonus = 1.5 if len(words) >= 2 else 0.8
                    is_capitalized = 1.2 if np_text[0].isupper() else 1.0

                    score = (freq * 1.2 + len(words) * 0.5) * length_bonus * is_capitalized
                    all_candidates.append({
                        "concept": np_text,
                        "type": np_type,
                        "sentence_idx": sent_idx,
                        "sentence": sentence,
                        "score": score
                    })

        # Sort candidates by salience score descending
        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        # De-duplicate near-identical concepts
        deduped = []
        covered_concepts: Set[str] = set()
        for cand in all_candidates:
            c_lower = cand["concept"].lower()
            if c_lower in covered_concepts:
                continue
            covered_concepts.add(c_lower)
            deduped.append(cand)

        return deduped
