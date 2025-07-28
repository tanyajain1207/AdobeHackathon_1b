#!/usr/bin/env python3
"""
Advanced Subsection Extractor for Round 1B: Persona-Driven Document Intelligence
"""

from __future__ import annotations
import logging
import re
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .persona_analyzer import PersonaAnalyzer, PersonaProfile
    from . import _package_logger as _LOGGER
except ImportError:
    # Standalone fallback
    import logging as _logging
    _LOGGER = _logging.getLogger("subsection_extractor")
    _LOGGER.setLevel(_logging.INFO)
    from persona_analyzer import PersonaAnalyzer, PersonaProfile  # type: ignore

class SubsectionExtractor:
    """
    Splits top-ranked sections into semantic chunks and selects the most
    relevant chunk per section using the persona+job embedding.
    """

    def __init__(self, analyzer: PersonaAnalyzer, 
                 max_chunk_size: int = 200):
        """
        Args:
            analyzer: Shared PersonaAnalyzer instance (loads model once)
            max_chunk_size: Approximate max characters per chunk
        """
        self.analyzer = analyzer
        self.max_chunk_size = max_chunk_size

    def extract_refined_subsections(
        self,
        top_sections: List[Dict[str, Any]],
        persona: str,
        job_to_be_done: str
    ) -> List[Dict[str, Any]]:
        """
        For each top section, split its content into chunks, score them,
        and choose the highest-scoring chunk.

        Args:
            top_sections: List of section dicts containing keys:
                'document', 'page', 'section_title', 'content'
            persona, job_to_be_done: Strings defining the user context

        Returns:
            List of dicts each with:
              - 'document': str
              - 'page_number': int
              - 'parent_section': str
              - 'refined_text': str
        """
        profile: PersonaProfile = self.analyzer.build_persona_profile(
            persona, job_to_be_done
        )
        query_emb = profile.query_embedding.reshape(1, -1)  # shape (1,384)

        refined: List[Dict[str, Any]] = []

        for sec in top_sections:
            content = sec.get('content', '').strip()
            if not content or len(content) < 20:
                _LOGGER.debug(
                    "Skipping section '%s' (too short)", sec.get('section_title')
                )
                continue

            chunks = self._split_into_chunks(content)
            if not chunks:
                continue

            # Batch‐encode all chunks
            chunk_embs = PersonaAnalyzer._EMBED_MODEL.encode(chunks)
            sims = cosine_similarity(query_emb, chunk_embs)[0]

            # Apply concept-based boost
            boosted_scores = [
                self.analyzer.boost_score_for_concepts(ch, profile, base)
                for ch, base in zip(chunks, sims)
            ]

            # Select best chunk
            best_idx = int(np.argmax(boosted_scores))
            best_chunk = chunks[best_idx].strip()

            refined.append({
                'document': sec.get('document'),
                'page_number': sec.get('page'),
                'parent_section': sec.get('section_title'),
                'refined_text': best_chunk
            })
            _LOGGER.debug(
                "Selected chunk for section '%s': %s...",
                sec.get('section_title'), best_chunk[:50]
            )

        _LOGGER.info("Extracted %d refined subsections", len(refined))
        return refined

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks by sentences, grouping until approx max_chunk_size.

        Args:
            text: Full section content

        Returns:
            List of text chunks (each ≤ max_chunk_size chars, except last)
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[str] = []
        current = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            if current:
                # Would overflow?
                if len(current) + len(sent) + 1 > self.max_chunk_size:
                    chunks.append(current.strip())
                    current = sent
                else:
                    current += " " + sent
            else:
                current = sent

        if current:
            chunks.append(current.strip())

        return chunks
