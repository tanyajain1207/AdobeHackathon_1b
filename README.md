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



