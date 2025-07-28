#!/usr/bin/env python3
"""
Advanced Document Processor for Round 1B: Persona-Driven Document Intelligence
"""

import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Attempt to import package-level utilities; fallback if standalone
try:
    from . import _package_logger, intelligent_error_handler, _performance_monitor
except ImportError:
    import logging
    _package_logger = logging.getLogger("document_processor")
    _package_logger.setLevel(logging.INFO)

    def intelligent_error_handler(func):
        return func

# Enums to classify document complexity and section types
class DocumentComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"

class SectionType(Enum):
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    UNKNOWN = "unknown"

@dataclass
class DocumentMetadata:
    file_path: Path
    file_size_mb: float
    page_count: int
    total_characters: int
    estimated_complexity: DocumentComplexity
    title: Optional[str] = None
    author: Optional[str] = None
    has_images: bool = False
    has_tables: bool = False
    confidence_score: float = 0.8

@dataclass
class ExtractedSection:
    document: str
    section_title: str
    content: str
    page: int
    level: str
    section_type: SectionType = SectionType.UNKNOWN
    confidence_score: float = 0.0
    word_count: int = 0
    character_count: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)

class IntelligentStructureAnalyzer:
    """Analyzes document to find and classify sections intelligently."""

    _SECTION_KEYWORDS = {
        SectionType.TITLE: ["title", "heading", "main", "primary"],
        SectionType.ABSTRACT: ["abstract", "summary", "overview", "executive summary"],
        SectionType.INTRODUCTION: ["introduction", "intro", "background", "motivation"],
        SectionType.METHODOLOGY: ["methodology", "methods", "approach", "materials", "experimental", "procedure"],
        SectionType.RESULTS: ["results", "findings", "outcomes", "data analysis"],
        SectionType.DISCUSSION: ["discussion", "analysis", "interpretation", "implications"],
        SectionType.CONCLUSION: ["conclusion", "conclusions", "summary", "final thoughts"],
        SectionType.REFERENCES: ["references", "bibliography", "citations", "works cited"],
    }

    def analyze_document_structure(self, doc: fitz.Document, metadata: DocumentMetadata) -> List[ExtractedSection]:
        _package_logger.info(f"Starting structure analysis of {metadata.file_path.name}")

        sections: List[Dict[str, Any]] = []
       
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict").get("blocks", [])

            for block in blocks:
                if block.get("type") != 0:
                    continue  # Only process text blocks

                for line in block.get("lines", []):
                    line_text = " ".join(span.get("text", "") for span in line.get("spans", [])).strip()
                    if not line_text:
                        continue

                    avg_font_size = np.mean([span.get("size", 12) for span in line.get("spans", [])])

                    if self._is_probable_heading(line_text, avg_font_size):
                        s_type = self._classify_section(line_text)
                        confidence = self._calc_confidence(line_text, avg_font_size)
                        bbox = line.get("bbox", None)

                        sections.append({
                            "text": line_text,
                            "page": page_num + 1,
                            "font_size": avg_font_size,
                            "section_type": s_type,
                            "confidence": confidence,
                            "bbox": tuple(bbox) if bbox else None,
                        })

        unique_sections = self._deduplicate_sections(sections)
        scored_sections = self._score_sections(unique_sections)
        enhanced_sections = self._enhance_sections(scored_sections)

        _package_logger.info(f"Extracted {len(enhanced_sections)} sections")
        return enhanced_sections

    def _is_probable_heading(self, text: str, font_size: float) -> bool:
        # Heuristic checks on length and font size
        if len(text) < 3 or len(text) > 200:
            return False
        if font_size < 10:
            return False
        # Check for capitalization or numeric prefix
        if re.match(r"^\d+\.?\s+[A-Z]", text) or text.isupper() or text.istitle():
            return True
        # Contains any section keyword
        text_l = text.lower()
        for keywords in self._SECTION_KEYWORDS.values():
            for kw in keywords:
                if kw in text_l:
                    return True
        return False

    def _classify_section(self, text: str) -> SectionType:
        text_l = text.lower()
        scores = {}
        for stype, keywords in self._SECTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_l)
            if score > 0:
                scores[stype] = score
        return max(scores, key=scores.get) if scores else SectionType.UNKNOWN

    def _calc_confidence(self, text: str, font_size: float) -> float:
        confidence = 0.5
        if text.isupper():
            confidence += 0.1
        if text.istitle():
            confidence += 0.1
        if re.match(r"^\d+\.?\s+", text):
            confidence += 0.2
        if font_size > 14:
            confidence += 0.1
        return min(confidence, 1.0)

    def _deduplicate_sections(self, sections: List[Dict]) -> List[Dict]:
        unique = []
        seen_texts = set()
        for sec in sections:
            txt_norm = sec["text"].strip().lower()
            if txt_norm not in seen_texts:
                unique.append(sec)
                seen_texts.add(txt_norm)
        return unique

    def _score_sections(self, sections: List[Dict]) -> List[Dict]:
        # Add a rough quality score using confidence and font size
        for sec in sections:
            score = sec.get("confidence", 0.0)
            font_size = sec.get("font_size", 12)
            if font_size > 12:
                score += 0.2
            sec["quality_score"] = min(score, 1.0)
        return sorted(sections, key=lambda s: s["quality_score"], reverse=True)

    def _enhance_sections(self, sections: List[Dict]) -> List[ExtractedSection]:
        enhanced = []
        for s in sections:
            level = self._estimate_heading_level(s["font_size"])
            esec = ExtractedSection(
                document="",  # Setup later
                section_title=s["text"],
                content="",
                page=s["page"],
                level=level,
                section_type=s["section_type"],
                confidence_score=s.get("confidence", 0.0),
                bbox=s.get("bbox"),
                quality_metrics={"quality_score": s.get("quality_score", 0.0)},
            )
            enhanced.append(esec)
        return enhanced

    def _estimate_heading_level(self, font_size: float) -> str:
        if font_size >= 18:
            return "H1"
        if font_size >= 16:
            return "H2"
        if font_size >= 14:
            return "H3"
        return "H4"

class IntelligentContentExtractor:
    """Extracts and cleans content text for document sections."""

    def __init__(self):
        pass

    @intelligent_error_handler
    def extract_section_content(self, doc: fitz.Document, sections: List[ExtractedSection], metadata: DocumentMetadata) -> List[ExtractedSection]:
        _package_logger.info(f"Extracting content for {len(sections)} sections")
        enhanced_sections = []
        for sect in sections:
            try:
                page = doc[sect.page - 1]  # Pages are 0-based internally
                content = self._extract_content_around_title(page, sect.section_title)
                processed = self._clean_text(content)
                sect.content = processed
                sect.word_count = len(processed.split())
                sect.character_count = len(processed)
                enhanced_sections.append(sect)
            except Exception as e:
                _package_logger.warning(f"Failed to extract content for section '{sect.section_title}': {e}")
                sect.content = ""
                enhanced_sections.append(sect)
        return enhanced_sections

    def _extract_content_around_title(self, page: fitz.Page, title: str) -> str:
        text = page.get_text()
        lowered = text.lower()
        title_l = title.lower()
        start_idx = lowered.find(title_l)
        if start_idx < 0:
            # Title not found, fallback: return first 500 chars
            return text[:500]
        # Extract text after title up to various heuristics (next heading or ~2000 chars)
        remainder = text[start_idx + len(title):]
        lines = remainder.splitlines()
        content_lines = []
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue
            # Heuristic stop conditions for new headings or sections
            if len(line_strip) < 100:
                if line_strip.isupper():
                    break
                if re.match(r"^\d+\.?\s+[A-Z]", line_strip):
                    break
            content_lines.append(line_strip)
            if sum(len(l) for l in content_lines) > 2000:
                break
        return "\n".join(content_lines)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix broken words from PDFs
        text = re.sub(r'\s+([.,:;!?])', r'\1', text)
        text = text.strip()
        return text

class DocumentProcessor:
    """Main orchestrator for document loading, structure and content extraction."""

    def __init__(self):
        self.structure_analyzer = IntelligentStructureAnalyzer()
        self.content_extractor = IntelligentContentExtractor()

    @intelligent_error_handler
    def _extract_intelligent_structure(self, pdf_path: Path) -> Dict[str, Any]:
        monitor_ctx = _performance_monitor.start_monitoring("structure_extraction")
        try:
            with fitz.open(str(pdf_path)) as doc:
                metadata = self._gather_metadata(pdf_path, doc)
                sections = self.structure_analyzer.analyze_document_structure(doc, metadata)
                for s in sections:
                    s.document = pdf_path.name  # Assign document name to sections

                outline = [{
                    "level": s.level,
                    "text": s.section_title,
                    "page": s.page,
                    "section_type": s.section_type.value,
                    "confidence": s.confidence_score,
                } for s in sections]

                _performance_monitor.end_monitoring(monitor_ctx)
                return {
                    "title": metadata.title or pdf_path.stem,
                    "outline": outline,
                    "metadata": {
                        "processing_method": "intelligent_structure",
                        "complexity": metadata.estimated_complexity.value,
                        "sections_found": len(sections),
                        "confidence": metadata.confidence_score,
                    }
                }
        except Exception as e:
            _package_logger.error(f"Failed intelligent structure extraction on {pdf_path}: {e}")
            _performance_monitor.end_monitoring(monitor_ctx)
            return {"title": pdf_path.stem, "outline": []}

    @intelligent_error_handler
    def extract_sections_with_content(self, pdf_path: Path, outline: List[Dict]) -> List[Dict]:
        monitor_ctx = _performance_monitor.start_monitoring("content_extraction")
        try:
            with fitz.open(str(pdf_path)) as doc:
                metadata = self._gather_metadata(pdf_path, doc)
                sections = []
                for item in outline:
                    s = ExtractedSection(
                        document=pdf_path.name,
                        section_title=item.get("text", ""),
                        content="",
                        page=item.get("page", 1),
                        level=item.get("level", "H3"),
                        section_type=SectionType(item.get("section_type", "unknown")),
                        confidence_score=item.get("confidence", 0.0),
                    )
                    sections.append(s)

                enhanced = self.content_extractor.extract_section_content(doc, sections, metadata)

                # Convert to dicts for compatibility
                result = []
                for s in enhanced:
                    result.append({
                        "document": s.document,
                        "section_title": s.section_title,
                        "content": s.content,
                        "page": s.page,
                        "level": s.level,
                        "section_type": s.section_type.value,
                        "confidence": s.confidence_score,
                        "word_count": s.word_count,
                        "quality_metrics": s.quality_metrics,
                    })

                _performance_monitor.end_monitoring(monitor_ctx)
                return result
        except Exception as e:
            _package_logger.error(f"Failed content extraction on {pdf_path}: {e}")
            _performance_monitor.end_monitoring(monitor_ctx)
            return []

    @intelligent_error_handler
    def extract_all_sections(self, pdf_path: Path) -> List[Dict]:
        """Combines structure extraction and content extraction in one step."""
        outline_result = self._extract_intelligent_structure(pdf_path)
        outline = outline_result.get("outline", [])
        return self.extract_sections_with_content(pdf_path, outline)

    def _gather_metadata(self, pdf_path: Path, doc: fitz.Document) -> DocumentMetadata:
        file_size = pdf_path.stat().st_size / (1024 * 1024)
        page_count = len(doc)
        total_chars = sum(len(page.get_text()) for page in doc)
        title = doc.metadata.get("title")
        author = doc.metadata.get("author")
        has_images = any(len(page.get_images()) > 0 for page in doc)
        has_tables = any(('\t' in page.get_text() or page.get_text().count('|') > 10) for page in doc)
        complexity = self._estimate_complexity(doc)
        return DocumentMetadata(
            file_path=pdf_path,
            file_size_mb=file_size,
            page_count=page_count,
            total_characters=total_chars,
            estimated_complexity=complexity,
            title=title,
            author=author,
            has_images=has_images,
            has_tables=has_tables,
            confidence_score=0.8,
        )

    def _estimate_complexity(self, doc: fitz.Document) -> DocumentComplexity:
        page_count = len(doc)
        image_count = sum(len(page.get_images()) for page in doc)
        text_volume = sum(len(page.get_text()) for page in doc)

        score = 0
        if page_count > 50:
            score += 2
        elif page_count > 20:
            score += 1

        if image_count > page_count * 2:
            score += 2
        elif image_count > 0:
            score += 1

        if text_volume / max(page_count,1) > 3000:
            score += 1

        if score >= 5:
            return DocumentComplexity.HIGHLY_COMPLEX
        elif score >= 3:
            return DocumentComplexity.COMPLEX
        elif score >= 1:
            return DocumentComplexity.MODERATE
        return DocumentComplexity.SIMPLE

# Module logger initialization
_package_logger.info("DocumentProcessor module loaded.")
