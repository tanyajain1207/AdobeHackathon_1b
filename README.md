# Adobe India Hackathon 2025
## Persona-Driven Document Intelligence
Theme: “Connect What Matters — For the User Who Matters”

Inside this GitHub Repository, there is the source code for the Intelligent, Persona-Driven Document (PDF) Parser and Dockerfile and Docker Setup for the Round 1(b) of the Adobe Hackathon. This source code contains a program, when ran, it processes PDF's in the "input" folder, parses the PDF, refines the documents and produces a structured JSON output.

### Contents of the repository
project-1b/
│
├── Dockerfile                 # Docker image build instructions
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview and instructions
├── src/                       # Source code folder
│   ├── __init__.py
│   ├── main.py                # Main entry point for pipeline
│   ├── document_processor.py
│   ├── subsection_extractor.py
│   ├── persona_analyzer.py
│   ├── relevence_ranker.py
│   ├── output_formatter.py
│   ├── config.json            # (optional) config file inside src
│
├── input/                     # Input PDFs and config.json
│   ├── (your input files)
│
├── output/                    # Results output folder
│   ├── (output files)

### Overall Approach for the Intelligent, Persona-Driven Document Intelligence Pipeline

1. Pipeline Orchestration (main-1.py)
Input and Configuration Parsing

Reads the input folder containing PDFs plus a required config.json that specifies the persona and the job-to-be-done context.

Validates environment, input presence, and config correctness.

Adaptive Workflow Selection

Analyzes the input context (document names, persona, job) to choose an optimized processing strategy dynamically.

Component Initialization

Lazily loads and initializes core pipeline components:

DocumentProcessor (for document parsing and content extraction)

PersonaAnalyzer (builds persona+job embedding profile)

RelevanceRanker (ranks extracted sections by persona relevance)

SubsectionExtractor (further refines top-ranked sections into concise subsections)

OutputFormatter (formats and saves final structured JSON output)

Processing Steps

Loads Round 1A outputs if available to bootstrap processing; otherwise, runs intelligent structure extraction.

Extracts semantic sections from PDFs.

Ranks and selects the top relevant sections according to persona-context embedding similarity.

Refines these top sections by splitting into manageable chunks and selecting the best snippet aligned with persona+job inputs.

Formats the extracted and refined content into a well-structured JSON document.

Performance Monitoring

Tracks processing time and quality metrics.

Issues warnings if processing time exceeds limits.

2. Semantic Document Processing (document_processor.py)
Intelligent Document Structure Analysis

Combines multiple extraction strategies:

Font-based analysis detecting headings by font size/style.

Layout-based spatial analysis examining visual structure and text positioning.

Content-based semantic classification using keywords and heuristics.

Deduplicates and fuses overlapping sections extracted through different strategies.

Classifies sections semantically (e.g., title, abstract, introduction, methodology).

Computes confidence and quality scores for extracted structure segments.

Content Extraction and Enhancement

Extracts meaningful content for each section based on bounding boxes, titles, or page position.

Cleans and normalizes text (fixing common PDF extraction errors, formatting paragraphs).

Computes content quality metrics (readability, completeness, information density).

Adaptive Processing Strategy

Able to handle documents of varying complexity from simple reports to highly complex research papers.

Supports multi-strategy, quality-versus-speed trade-offs depending on runtime considerations.

3. Persona-Driven Semantic Ranking and Refinement
Persona Analysis (persona_analyzer.py)

Converts raw persona and job descriptions into actionable embeddings using lightweight Sentence-Transformers (MiniLM).

Extracts key concepts for boosting relevance scoring.

Infers domain and cognitive style to adapt ranking behaviors.

Relevance Ranking (relevance_ranker.py)

Scores all extracted sections based on cosine similarity between section+title embeddings and persona+job embedding.

Applies heuristic boosts if the section contains persona-specific key concepts.

Sorts and ranks sections by relevance.

Subsection Extraction (subsection_extractor.py)

For the top-ranked sections, splits contents into smaller, manageable chunks (by sentences).

Encodes and scores chunks to select the most relevant fragment per section (persona and job aware).

Optimizes for CPU execution, caching, and minimal overhead.

Output Formatting (output_formatter.py)

Builds final JSON output with metadata, ranked sections, refined subsections, and timestamps.

Optionally validates output against schema.

Supports pretty printing and post-processing hooks.

4. Runtime and Resource Management
Uses caching and singleton patterns to load heavy models (sentence transformers) only once per run.

Designed for CPU-only execution with lightweight models like MiniLM (~80MB size), meeting competition constraints.

Ensures no internet access required at runtime (models are pre-downloaded in Docker build).

Monitors performance metrics and adapts workflow accordingly.

### Models and Libraries Used
sentence-transformers>=2.2.2
PyMuPDF>=1.23.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8
pandas>=1.5.0

### Installation & Setup

1) Prerequisites
    Docker Desktop installed and running (for Windows/Mac/Linux)
    Free disk space (recommend at least 5GB+)
    docker logged in

2) Clone the Git Repo:
   git clone https://github.com/tanyajain1207/AdobeHackathon_1b.git
   cd AdobeHackathon_1b

3) Pull the docker image:
   docker pull tanyajain1207/adobe1b:latest

4) Run the Image:
  docker run --rm \
  -v "$PWD/input":/app/input \
  -v "$PWD/output":/app/output \
  tanyajain1207/adobe1b:latest



