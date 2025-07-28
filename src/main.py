#!/usr/bin/env python3
import sys
import argparse
import logging
from pathlib import Path

from src import (
    get_component,
    get_orchestrator,
    get_performance_monitor,
    validate_system_requirements,
    intelligent_error_handler,
    PERFORMANCE_CONSTRAINTS,
)

# Configure logger
LOGGER = logging.getLogger("round1b_main")
LOGGER.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s | MAIN | %(levelname)s | %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
LOGGER.addHandler(handler)


@intelligent_error_handler
def main(args):
    # 1. Validate environment and dependencies
    env = validate_system_requirements()
    if not env.get("system_ready", False):
        LOGGER.warning("Environment not ready. Missing or failing dependencies: %s", env.get("failed"))

    # 2. Resolve input directory
    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        LOGGER.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # 3. Determine output directory (optional)
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = Path.cwd() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Load config.json
    cfg_file = input_dir / "config.json"
    if not cfg_file.exists():
        LOGGER.error("Missing config.json in input directory: %s", input_dir)
        sys.exit(1)
    import json
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    persona = cfg.get("persona", "")
    job_to_be_done = cfg.get("job_to_be_done", "")
    LOGGER.info("Persona: %s", persona)
    LOGGER.info("Job-to-be-done: %s", job_to_be_done)

    # 5. Gather PDFs
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        LOGGER.error("No PDF files found in input directory: %s", input_dir)
        sys.exit(1)
    docs = [p.name for p in pdf_files]
    LOGGER.info("Found %d PDF documents", len(pdf_files))

    # 6. Orchestrator & workflow
    orchestrator = get_orchestrator()
    context = orchestrator.analyze_processing_context({"documents": docs})
    workflow_cfg = orchestrator.optimize_workflow(context)
    LOGGER.info("Workflow strategy selected: %s", workflow_cfg["strategy"])

    # 7. Initialize components
    dp = get_component("document_processor")
    rr = get_component("relevance_ranker")
    se = get_component("subsection_extractor")
    of = get_component("output_formatter")  # instance

    # 8. Configure OutputFormatter output path
    output_json_path = output_dir / "result.json"
    of.output_path = output_json_path

    # 9. Start performance monitoring
    perf_mon = get_performance_monitor()
    monitor_ctx = perf_mon.start_monitoring("full_pipeline")

    # 10. Extract all sections and content
    all_sections = []
    for pdf_path in pdf_files:
        sections = dp.extract_all_sections(pdf_path)
        all_sections.extend(sections)
    LOGGER.info("Extracted %d sections", len(all_sections))

    # 11. Rank & select top sections
    ranked = rr.rank_sections(all_sections, persona, job_to_be_done)
    top_sections = rr.select_top_sections(ranked, max_sections=workflow_cfg.get("batch_size", 10))
    LOGGER.info("Selected top %d sections", len(top_sections))

    # 12. Refine subsections
    refined = se.extract_refined_subsections(top_sections, persona, job_to_be_done)
    LOGGER.info("Extracted %d refined subsections", len(refined))

    # 13. Format and save output
    result = of.format_output(docs, persona, job_to_be_done, top_sections, refined, {"workflow": workflow_cfg})
    of.save_output(result)
    LOGGER.info("Output saved to %s", output_json_path)

    # 14. End performance monitoring
    perf = perf_mon.end_monitoring(monitor_ctx)
    LOGGER.info("Pipeline completed in %.2f seconds", perf["processing_time"])
    if perf["processing_time"] > PERFORMANCE_CONSTRAINTS["max_processing_time_seconds"]:
        LOGGER.warning("Processing time exceeded limit of %d seconds", PERFORMANCE_CONSTRAINTS["max_processing_time_seconds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round 1B: Persona-Driven Document Intelligence Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Path to input directory containing config.json and PDFs")
    parser.add_argument(
        "--output", "-o",
        required=False,
        help="Optional path to output directory; defaults to ./output"
    )
    args = parser.parse_args()
    main(args)
