# MyJobSpy Analyst: Scrape, Analyze, and Filter Jobs with GenAI

This project enhances job searching by combining the scraping power of **[JobSpy](https://github.com/speedyapply/JobSpy)** with local Generative AI analysis via **Ollama** and advanced filtering, including location awareness.

**Core Workflow:**

1.  **Configure:** Set up your preferences, API keys (if needed), and Ollama model in `config.yaml`. Prepare LinkedIn cookies in `config.toml` (optional but recommended).
2.  **Scrape:** Uses `JobSpy` to find jobs based on your search criteria and location across multiple job boards (LinkedIn, Indeed, etc.).
3.  **Analyze (Async):** Parses your resume (DOCX/PDF) using `PyPDF2`/`python-docx`. It then uses a local Ollama model (e.g., Llama 3, Mistral) concurrently via `asyncio` to:
    *   Extract structured data from your resume (results are cached for speed).
    *   Compare your structured resume profile against each scraped job description.
    *   Generate a suitability score (0-100%), detailed justification, and skill/experience match assessment based on enhanced comparison logic.
4.  **Filter & Rank:** Filters the analyzed jobs based on a wide range of criteria including salary, job type, work model, company names (include/exclude), title keywords, date posted, location (remote country or proximity), and minimum suitability score. Ranks the final list by suitability score.
5.  **Output:** Saves the detailed analysis results (including original job data) to a JSON file and prints a summary table of top matches to the console using `rich`. Geocoding results are cached to disk.

## Features

*   **Multi-Site Scraping:** Leverages `JobSpy` to scrape from sites like LinkedIn, Indeed, ZipRecruiter, Glassdoor, etc. (Check `JobSpy` docs for current support).
*   **Asynchronous Analysis:** Significantly speeds up analysis by processing multiple jobs concurrently with the Ollama LLM using `asyncio` and `httpx`.
*   **Resume Parsing:** Handles `.docx` and `.pdf` resume files.
*   **Local GenAI Analysis:** Uses Ollama (`llama3:instruct`, `mistral:instruct`, etc.) for privacy, control, and cost savings.
    *   Configurable model, timeout, and retries via `config.yaml`.
    *   Structured resume data extraction with emphasis on quantifiable achievements.
    *   Detailed job suitability scoring based on recruiter-like evaluation criteria (essentials, relevance, impact).
    *   Evidence-based justification for scores.
*   **Advanced Filtering:**
    *   Salary range (min/max).
    *   Job Type(s) (Full-time, Contract, etc.).
    *   Work Model(s) (Remote, Hybrid, On-site).
    *   Company Name inclusion or exclusion lists.
    *   Job Title keyword matching (any keyword).
    *   Date Posted range (after/before YYYY-MM-DD).
    *   Remote jobs within a specific country (uses Geopy).
    *   Hybrid/On-site jobs within a specific mileage range of a location (uses Geopy).
    *   Minimum Suitability Score (0-100).
*   **Caching:**
    *   **Resume Analysis:** Caches structured resume data based on file hash to speed up subsequent runs with the same resume (`output/.resume_cache/`). Use `--force-resume-reparse` to override.
    *   **Geocoding:** Caches geocoding results to disk (`output/.geocode_cache.json`) to minimize API calls to Nominatim and improve speed on repeated runs or similar locations.
*   **Robustness:**
    *   More specific error handling for scraping, analysis, file I/O, and network issues.
    *   Retry logic for Ollama API calls.
    *   Graceful handling of `Ctrl+C` interruptions.
    *   Automatic input truncation for LLM prompts to mitigate context window errors/timeouts.
    *   Handles jobs with missing descriptions scraped by `JobSpy` by skipping analysis for them.
*   **Configuration:** Centralized settings via `config.yaml` with environment variable overrides for key parameters.
*   **Rich Output:** Provides detailed JSON output and a configurable summary table in the console.

**❗ Disclaimer:** Web scraping is inherently fragile. Job boards change frequently, implement anti-scraping measures (CAPTCHAs, blocks), and may forbid automated scraping in their Terms of Service. Scraping success (especially for sites like Glassdoor or non-cookied LinkedIn) is **NOT guaranteed.** This tool relies heavily on the `JobSpy` library; its effectiveness depends on `JobSpy`'s maintenance state and the current behavior of target websites. Geocoding relies on the Nominatim service, which has usage limits. Use responsibly and ethically.

## Prerequisites

*   **Python 3.9+** (Recommended for enhanced `asyncio` support and type hinting)
*   **Git**
*   **Ollama:** Install and run Ollama locally.
    *   Download: [https://ollama.com/](https://ollama.com/)
    *   Ensure the server is running (`ollama serve` or via App).
*   **Ollama Model:** Pull a suitable instruction-following model. Models are configured in `config.yaml` (default: `llama3:instruct`). Run in terminal:
    *   `ollama pull llama3:instruct` (Recommended Default)
    *   `ollama pull mistral:instruct`
    *   *(Or any other compatible model)*
*   **Playwright Browsers:** `JobSpy` uses Playwright for some scrapers. Install required browsers (might take time/disk space):
    ```bash
    playwright install
    ```
*   **Geocoding User Agent:** Nominatim (used by Geopy) requires a unique user agent. Set the `GEOPY_USER_AGENT` environment variable OR edit the `geocoding.user_agent` value in `config.yaml`. **Using the default placeholder will likely result in geocoding errors.** Example format: `MyAppName/1.0 (myemail@example.com)`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kasnycdev/MyJobSpy.git
    cd MyJobSpy
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Playwright browsers:**
    ```bash
    playwright install
    ```

5.  **Configure `config.yaml`:**
    *   Review and edit `config.yaml` in the project root.
    *   **Crucially, set `geocoding.user_agent`** to a unique value representing your application/email.
    *   Adjust Ollama model, default scraping sites, timeouts, cache paths, etc., as needed.

6.  **Configure LinkedIn Cookies (Optional but Recommended):**
    *   Create a `config.toml` file in the project root (this file is ignored by git).
    *   Find your `li_at` and `JSESSIONID` cookies from your browser after logging into LinkedIn (use Developer Tools -> Application/Storage -> Cookies).
    *   Add them to `config.toml`:
        ```toml
        # config.toml
        [linkedin]
        li_at = "YOUR_LI_AT_COOKIE_VALUE"
        JSESSIONID = "YOUR_JSESSIONID_COOKIE_VALUE_WITH_QUOTES" # Include quotes if present
        ```
    *   `JobSpy` will automatically detect and use this file for authenticated LinkedIn scraping.

## Usage

Run the main pipeline script from the project's root directory:

```bash
python run_pipeline.py --search "Your Job Search" --resume /path/to/your/resume.pdf [LOCATION_FLAG] [OPTIONS]