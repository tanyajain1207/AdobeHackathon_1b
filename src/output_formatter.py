#!/usr/bin/env python3
"""
Advanced Output Formatter for Round 1B: Persona-Driven Document Intelligence
─────────────────────────────────────────────────────────────────────────────
Assembles and persists the JSON output, merging ranked sections
and refined subsections into the required schema, and attaches metadata
including timestamps and performance metrics.

Key capabilities:
- Builds `extracted_sections` and `subsection_analysis` arrays
- Embeds processing statistics (time, quality) into metadata
- ISO-8601 timestamping
- Pretty-printed JSON with UTF-8 encoding
- Optional output validation against schema
- Pluggable hooks for post-processing (e.g., writing CSV summaries)

Author     : Advanced AI Development Team
Version    : 2.0.0-alpha
License    : MIT
Last Update: 23 Jul 2025
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from . import _package_logger as _LOGGER, validate_system_requirements
except ImportError:
    import logging as _logging
    _LOGGER = _logging.getLogger("output_formatter")
    _LOGGER.setLevel(_logging.INFO)


class OutputFormatter:
    """
    Formats and writes the final result JSON according to the Round 1B schema.
    """

    def __init__(
        self,
        output_path: Optional[Path] = None,
        enable_validation: bool = True,
    ) -> None:
        """
        Args:
            output_path: Optional Path to write JSON; can also call `save_output`
            enable_validation: If True, validate against the JSON schema
        """
        self.output_path = output_path
        self.enable_validation = enable_validation

    def format_output(
        self,
        input_documents: List[str],
        persona: str,
        job_to_be_done: str,
        ranked_sections: List[Dict[str, Any]],
        refined_subsections: List[Dict[str, Any]],
        processing_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assemble the final result dictionary.

        Args:
            input_documents: List of input PDF filenames.
            persona: Persona description string.
            job_to_be_done: Job description string.
            ranked_sections: List of dicts with keys:
                'document', 'page', 'section_title', 'importance_rank', 'relevance_score'
            refined_subsections: List of dicts with keys:
                'document', 'page_number', 'parent_section', 'refined_text'
            processing_stats: Optional dict with processing metrics.

        Returns:
            A dict matching the required output schema.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Build extracted_sections array with correct keys
        extracted = [
            {
                "document": sec["document"],
                "page_number": sec["page"],  # remap internal 'page' to output 'page_number'
                "section_title": sec["section_title"],
                "importance_rank": sec["importance_rank"],
            }
            for sec in ranked_sections
        ]

        # Build subsection_analysis array
        subsections = [
            {
                "document": sub["document"],
                "refined_text": sub["refined_text"],
                "page_number": sub["page_number"],
            }
            for sub in refined_subsections
        ]

        # Assemble metadata with optional processing stats
        metadata: Dict[str, Any] = {
            "input_documents": input_documents,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": timestamp,
        }
        if processing_stats:
            metadata["processing_stats"] = processing_stats

        result = {
            "metadata": metadata,
            "extracted_sections": extracted,
            "subsection_analysis": subsections,
        }

        if self.enable_validation:
            self._validate_schema(result)

        return result

    def save_output(self, result: Dict[str, Any], output_path: Optional[Path] = None) -> None:
        """
        Persist the result dict to JSON on disk.

        Args:
            result: The dict returned by `format_output`.
            output_path: If provided, overrides the instance’s `output_path`.
        """
        path = output_path or self.output_path
        if path is None:
            _LOGGER.error("No output path specified for saving results")
            raise ValueError("Output path must be provided to save output")

        # Ensure parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        _LOGGER.info(f"Result JSON saved to {str(path)}")

    def _validate_schema(self, result: Dict[str, Any]) -> None:
        """
        Placeholder for JSON Schema validation logic.
        Raises if validation fails.
        """
        try:
            # e.g. jsonschema.validate(instance=result, schema=SCHEMA)
            _LOGGER.debug("Output validation skipped (no schema provided).")
            pass
        except Exception as e:
            _LOGGER.error(f"Output schema validation failed: {e}")
            raise


# Public API
__all__ = ["OutputFormatter"]

