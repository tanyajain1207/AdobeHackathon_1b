#!/usr/bin/env python3
"""
Advanced Persona Analyzer for Round 1B: Persona-Driven Document Intelligence
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "⚠️  sentence-transformers package not found. "
        "Add it to requirements.txt → `sentence-transformers>=2.2.2`"
    ) from exc

# Fetch the intelligent logging utilities from package root if available
try:
    from . import _package_logger as _LOGGER, intelligent_error_handler
except ImportError:  # Stand-alone fallback
    _LOGGER = logging.getLogger("persona_analyzer")
    _LOGGER.setLevel(logging.INFO)

    def intelligent_error_handler(func):  # type: ignore
        return func


# ════════════════════════════════════════════════════════════════════════
# ENUMS & DATACLASSES
# ════════════════════════════════════════════════════════════════════════
class PersonaDomain(Enum):
    """High-level expertise domains used for adaptive ranking."""
    RESEARCH = "research"
    BUSINESS = "business"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    LEGAL = "legal"
    GENERAL = "general"


class CognitiveStyle(Enum):
    """Very lightweight cognitive/reading-preference model."""
    DETAIL_ORIENTED = "detail_oriented"
    SUMMARY_ORIENTED = "summary_oriented"
    VISUAL = "visual"
    DATA_DRIVEN = "data_driven"
    UNKNOWN = "unknown"


@dataclass
class PersonaProfile:
    """Structured representation of a persona after analysis."""
    raw_persona: str
    raw_job: str
    domain: PersonaDomain
    cognitive_style: CognitiveStyle
    key_concepts: List[str]
    query_embedding: np.ndarray
    cache_key: str = field(init=False)

    def __post_init__(self) -> None:
        # Deterministic cache key for quick re-use
        digest = hashlib.sha256(
            (self.raw_persona + "|" + self.raw_job).encode("utf-8")
        ).hexdigest()[:16]
        self.cache_key = digest


# ════════════════════════════════════════════════════════════════════════
# MAIN ANALYZER
# ════════════════════════════════════════════════════════════════════════
class PersonaAnalyzer:
    """
    Converts “persona + job-to-be-done” strings into an *actionable* profile.

    Typical use:
        analyzer = PersonaAnalyzer()
        profile  = analyzer.build_persona_profile(persona, job)
        embedding = profile.query_embedding        # 1 × 384 ndarray
        concepts  = profile.key_concepts           # up to 10 keywords
    """

    _MODEL_LOCK = threading.Lock()
    _EMBED_MODEL: Optional[SentenceTransformer] = None

    # --------------------------- INITIALIZATION ------------------------
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_concepts: int = 10,
    ) -> None:
        self.model_name = model_name
        self.max_concepts = max_concepts
        if PersonaAnalyzer._EMBED_MODEL is None:
            # Thread-safe singleton load
            with PersonaAnalyzer._MODEL_LOCK:
                if PersonaAnalyzer._EMBED_MODEL is None:
                    _LOGGER.info("🔌 Loading embedding model: %s", model_name)
                    PersonaAnalyzer._EMBED_MODEL = SentenceTransformer(
                        model_name, device="cpu"
                    )

    # --------------------------- PUBLIC API ----------------------------
    @intelligent_error_handler
    def build_persona_profile(
        self, persona: str, job_to_be_done: str
    ) -> PersonaProfile:
        """Main entry: returns fully populated PersonaProfile dataclass."""
        persona = persona.strip()
        job_to_be_done = job_to_be_done.strip()

        domain = self._infer_domain(persona)
        cognitive_style = self._infer_cognitive_style(job_to_be_done)
        key_concepts = self._extract_key_concepts(persona, job_to_be_done)
        query_embedding = self._create_query_embedding(persona, job_to_be_done)

        return PersonaProfile(
            raw_persona=persona,
            raw_job=job_to_be_done,
            domain=domain,
            cognitive_style=cognitive_style,
            key_concepts=key_concepts,
            query_embedding=query_embedding,
        )

    def boost_score_for_concepts(
        self,
        text: str,
        profile: PersonaProfile,
        base_score: float,
        factor: float = 0.05,
        max_boost: float = 0.20,
    ) -> float:
        """
        Heuristically boost a similarity score when a text chunk contains
        persona-specific key concepts. Designed to be *idempotent*—do not call
        twice on the same score.
        """
        text_lower = text.lower()
        hits = sum(1 for kw in profile.key_concepts if kw in text_lower)
        boost = min(max_boost, hits * factor)
        return float(base_score) + boost

    # ----------------------- EMBEDDING UTILITIES -----------------------
    def _create_query_embedding(self, persona: str, job: str) -> np.ndarray:
        """Compose persona + job into a single 384-dim embedding."""
        model = PersonaAnalyzer._EMBED_MODEL  # type: ignore
        composite = f"{persona.strip()}: {job.strip()}"
        return model.encode([composite])[0]  # shape → (384,)

    # ------------------ KEY CONCEPT EXTRACTION -------------------------
    def _extract_key_concepts(self, persona: str, job: str) -> List[str]:
        """
        GPU-free concept mining:
        1. TF-IDF across the two sentences
        2. Fallback: regex token counts
        """
        corpus = [persona, job]
        try:
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                min_df=1,
                max_features=50,
            )
            tfidf.fit(corpus)
            scores = tfidf.idf_
            terms = tfidf.get_feature_names_out()
            ranked = sorted(
                zip(terms, scores), key=lambda t: t[1], reverse=True
            )
            keywords = [t for t, _ in ranked[: self.max_concepts]]
            return keywords
        except Exception as exc:  # pragma: no cover
            _LOGGER.warning("⚠️  TF-IDF failed (%s). Using fallback.", exc)
            return self._fallback_keyword_extract(corpus)

    # ------------------- DOMAIN / STYLE INFERENCE ----------------------
    _DOMAIN_HINTS = {
        PersonaDomain.RESEARCH: (
            "research",
            "phd",
            "professor",
            "scientist",
            "academic",
            "literature review",
        ),
        PersonaDomain.BUSINESS: (
            "analyst",
            "market",
            "manager",
            "executive",
            "investment",
        ),
        PersonaDomain.TECHNICAL: ("engineer", "developer", "architect", "technical"),
        PersonaDomain.MEDICAL: ("doctor", "clinical", "medicine", "physician"),
        PersonaDomain.LEGAL: ("lawyer", "attorney", "legal", "counsel"),
    }

    def _infer_domain(self, persona: str) -> PersonaDomain:
        lower = persona.lower()
        for domain, hints in self._DOMAIN_HINTS.items():
            if any(h in lower for h in hints):
                return domain
        return PersonaDomain.GENERAL

    _STYLE_HINTS = {
        CognitiveStyle.DETAIL_ORIENTED: ("in-depth", "comprehensive", "detailed"),
        CognitiveStyle.SUMMARY_ORIENTED: ("executive summary", "overview", "brief"),
        CognitiveStyle.DATA_DRIVEN: ("metrics", "benchmarks", "kpi", "data"),
        CognitiveStyle.VISUAL: ("diagram", "chart", "visual", "figure"),
    }

    def _infer_cognitive_style(self, job: str) -> CognitiveStyle:
        lower = job.lower()
        for style, hints in self._STYLE_HINTS.items():
            if any(h in lower for h in hints):
                return style
        return CognitiveStyle.UNKNOWN

    # ------------------- INTERNAL HELPERS ------------------------------
    def _fallback_keyword_extract(self, corpus: Sequence[str]) -> List[str]:
        """Regex-based fallback if TF-IDF unavailable."""
        tokens = re.findall(r"\b[a-zA-Z]{4,}\b", " ".join(corpus).lower())
        common = Counter(tokens)
        return [w for w, _ in common.most_common(self.max_concepts)]

    # ------------------ SERIALISATION HELPERS --------------------------
    @staticmethod
    def save_profile(profile: PersonaProfile, path: Path) -> None:
        """Persist profile to JSON (embedding is base64-encoded)."""
        data = {
            "persona": profile.raw_persona,
            "job_to_be_done": profile.raw_job,
            "domain": profile.domain.value,
            "cognitive_style": profile.cognitive_style.value,
            "key_concepts": profile.key_concepts,
            "embedding": profile.query_embedding.tolist(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def load_profile(path: Path) -> PersonaProfile:
        """Load profile JSON previously saved by `save_profile`."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return PersonaProfile(
            raw_persona=data["persona"],
            raw_job=data["job_to_be_done"],
            domain=PersonaDomain(data["domain"]),
            cognitive_style=CognitiveStyle(data["cognitive_style"]),
            key_concepts=data["key_concepts"],
            query_embedding=np.asarray(data["embedding"], dtype=np.float32),
        )


# ════════════════════════════════════════════════════════════════════════
# PUBLIC EXPORTS
# ════════════════════════════════════════════════════════════════════════
__all__ = [
    "PersonaAnalyzer",
    "PersonaProfile",
    "PersonaDomain",
    "CognitiveStyle",
]
