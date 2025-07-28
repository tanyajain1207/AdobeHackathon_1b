#!/usr/bin/env python3
"""
Advanced Relevance Ranker for Round 1B: Persona-Driven Document Intelligence
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .persona_analyzer import PersonaAnalyzer, PersonaProfile
    from . import _package_logger as _LOGGER
except ImportError:
    # Standalone fallback
    import logging as _logging
    _LOGGER = _logging.getLogger("relevance_ranker")
    _LOGGER.setLevel(_logging.INFO)
    from persona_analyzer import PersonaAnalyzer, PersonaProfile  # type: ignore

class RelevanceRanker:
    """
    Scores and ranks document sections based on semantic similarity to a
    PersonaProfile.query_embedding, applying concept-based boost.
    """

    def __init__(self, analyzer: PersonaAnalyzer):
        """
        Args:
            analyzer: Shared PersonaAnalyzer instance (loads model once)
        """
        self.analyzer = analyzer

    def rank_sections(
        self,
        sections: List[Dict[str, Any]],
        persona: str,
        job_to_be_done: str
    ) -> List[Dict[str, Any]]:
        """
        Compute relevance_score for each section and sort descending.

        Args:
            sections: List of dicts each with keys 'section_title' & 'content'
            persona, job_to_be_done: Strings defining the user context

        Returns:
            List of dicts augmented with:
              - 'relevance_score': float
              - 'importance_rank': int   (1 = most relevant)
        """
        # Build or reuse profile
        profile: PersonaProfile = self.analyzer.build_persona_profile(persona, job_to_be_done)
        query_emb = profile.query_embedding.reshape(1, -1)  # shape (1, 384)

        # Prepare texts for embedding
        texts = [
            f"{sec['section_title']} {sec['content']}".strip()
            for sec in sections
        ]
        if not texts:
            _LOGGER.warning("No sections to rank")
            return []

        # Batch‐encode all sections
        _LOGGER.debug(f"Encoding {len(texts)} section embeddings")
        section_embs = PersonaAnalyzer._EMBED_MODEL.encode(texts)  # shape (N, 384)

        # Cosine similarity
        sims = cosine_similarity(query_emb, section_embs)[0]  # shape (N,)

        # Apply heuristic boost for key concepts
        boosted = []
        for sec, base_score in zip(sections, sims):
            boosted_score = self.analyzer.boost_score_for_concepts(
                sec['section_title'] + " " + sec['content'],
                profile,
                base_score
            )
            sec_copy = sec.copy()
            sec_copy['relevance_score'] = float(boosted_score)
            boosted.append(sec_copy)

        # Sort by boosted score descending
        boosted.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Assign importance ranks
        for idx, sec in enumerate(boosted, start=1):
            sec['importance_rank'] = idx

        _LOGGER.info("Ranked %d sections", len(boosted))
        return boosted

    def select_top_sections(
        self,
        ranked_sections: List[Dict[str, Any]],
        max_sections: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Return the top‐N most relevant sections.

        Args:
            ranked_sections: Output of rank_sections()
            max_sections: Max number of sections to return

        Returns:
            Sliced list of dicts
        """
        return ranked_sections[:max_sections]
