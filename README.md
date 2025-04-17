# MyJobSpy (Enhanced with GenAI Matching)

This project scrapes job postings (original functionality assumed) and provides tools to analyze job descriptions against a candidate's resume using Generative AI ( Ollama with models like Llama 3, Mistral, Phi-3) to generate suitability scores and insights.

## Features

*   Parses resumes from `.docx` and `.pdf` files.
*   Loads job descriptions from a JSON file (list of job objects).
*   Utilizes a local GenAI model via Ollama for analysis:
    *   Extracts structured data (skills, experience, education) from the resume.
    *   Compares resume data against each job description.
    *   Generates a suitability score (0-100%).
    *   Provides a textual justification for the score.
    *   Identifies potential skill/experience/qualification matches.
    *   Assesses potential salary and benefit alignment (basic).
*   Filters jobs based on:
    *   Salary range (annual)
    *   Location (including "Remote")
    *   Work model (Remote, Hybrid, On-site)
    *   Job type (Full-time, Contract, etc.)
*   Outputs results in JSON format, combining original job data with GenAI analysis, sorted by suitability score.

## Prerequisites

*   **Python 3.8+**
*   **Git**
*   **Ollama:** You MUST have Ollama installed and running.
    *   Download from [https://ollama.com/](https://ollama.com/)
    *   Ensure the Ollama server is running (`ollama serve` or via its application).
*   **Ollama Model:** Pull a suitable instruction-following model. Recommended starting points:
    *   `ollama pull llama3:instruct` (Good general purpose, ~4.7GB)
    *   `ollama pull mistral:instruct` (Solid alternative, ~4.1GB)
    *   `ollama pull phi3:instruct` (Smaller, faster, might be less accurate, ~2.3GB)
    *   *The default model used by the script is `llama3:instruct`. You can change this in `config.py` or via the `OLLAMA_MODEL` environment variable.*

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd MyJobSpy
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Ollama (Optional):**
    *   If Ollama is running on a different host/port, set the `OLLAMA_BASE_URL` environment variable or modify `config.py`.
    *   To use a different model, set the `OLLAMA_MODEL` environment variable or modify `config.py`.

## Input Data Format

### Resume
*   A single `.docx` or `.pdf` file. Text-based PDFs work best. Complex layouts or image-based PDFs might yield poor results.

### Job Mandates JSON (`--jobs` argument)
*   A JSON file containing a *list* of job objects. Each object should be a dictionary.
*   **Required fields for analysis:** The script works best if jobs have a `description` field.
*   **Fields used for filtering (case-insensitive):**
    *   `salary_min`, `salary_max`: (Optional) Numerical minimum/maximum annual salary.
    *   `salary_text` or `salary`: (Optional) String containing salary info if numerical fields aren't present (e.g., "$90k - $110k per year"). The parser is basic.
    *   `location`: (Optional) String describing the job location(s).
    *   `work_model` or `remote`: (Optional) String like "Remote", "Hybrid", "On-site". The script also checks `location` for "remote".
    *   `job_type` or `employment_type`: (Optional) String like "Full-time", "Part-time", "Contract".
*   **Example `jobs.json`:**
    ```json
    [
      {
        "title": "Senior Python Developer",
        "company": "TechCorp",
        "location": "New York, NY (Hybrid)",
        "description": "Looking for an experienced Python developer with 5+ years experience in Django, DRF, and cloud platforms (AWS preferred). Strong understanding of databases (PostgreSQL) and CI/CD pipelines required. Experience with microservices is a plus.",
        "required_skills": ["Python", "Django", "DRF", "AWS", "PostgreSQL", "CI/CD"],
        "salary_text": "$120,000 - $150,000 annually",
        "work_model": "Hybrid",
        "job_type": "Full-time",
        "url": "http://example.com/job1"
      },
      {
        "title": "Frontend Engineer (React)",
        "company": "Innovate Solutions",
        "location": "Remote",
        "description": "Seeking a skilled Frontend Engineer proficient in React, TypeScript, and modern CSS frameworks. Build and maintain user interfaces for our web applications. Collaborate with backend teams.",
        "salary_min": 90000,
        "salary_max": 110000,
        "work_model": "Remote",
        "employment_type": "Full-time",
        "benefits_text": "Includes health, dental, vision insurance, and 401k matching.",
        "url": "http://example.com/job2"
      }
    ]
    ```

## Usage

Run the main analysis script from the project's root directory:

```bash
python main_matcher.py --resume /path/to/your/resume.pdf --jobs /path/to/your/jobs.json --output output/results.json