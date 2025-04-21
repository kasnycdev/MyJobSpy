# config.py
import os
import logging

# --- General Settings ---
# Get the absolute path of the directory where this config file is located
# This assumes config.py is in the project root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_SCRAPED_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "scraped_jobs.json")
DEFAULT_ANALYSIS_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "analyzed_jobs.json")

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Default to llama3:instruct, as it seemed more reliable in earlier tests than llama3:latest or others for JSON structure
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct")
# Increased default timeout - adjust if still needed
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", 450)) # 7.5 minutes
OLLAMA_MAX_RETRIES = 3
OLLAMA_RETRY_DELAY = 5 # Base delay in seconds for exponential backoff

# --- Analysis Prompts ---
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "prompts")
RESUME_PROMPT_FILE = "resume_extraction.prompt"
SUITABILITY_PROMPT_FILE = "suitability_analysis.prompt"
# Context window approximation - Keep relatively large, but adjust if needed
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", 24000))

# --- JobSpy Scraping Defaults ---
# Sites supported by jobspy (check jobspy documentation for the latest list)
DEFAULT_SCRAPE_SITES = ["linkedin", "indeed", "zip_recruiter", "glassdoor"]
DEFAULT_RESULTS_LIMIT = 20 # Default number of results per site for scraping
DEFAULT_HOURS_OLD = 72 # Default max age of jobs in hours (3 days)
DEFAULT_COUNTRY_INDEED = "usa" # Default country for Indeed searches

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO # Default level (can be overridden by -v flag)
LOG_FORMAT = "%(message)s" # Let Rich handler manage detailed formatting
LOG_DATE_FORMAT = "[%X]"

# --- Geocoding ---
# IMPORTANT: Nominatim requires a unique user agent. Replace with your app name/email.
# It's better to set this as an environment variable (GEOPY_USER_AGENT)
# than hardcoding it here if sharing the code.
GEOPY_USER_AGENT = os.getenv("GEOPY_USER_AGENT", "MyJobSpyAnalysisApp/1.0 (anonymous_user)")


# --- Function to create output dir ---
def ensure_output_dir():
    """Creates the default output directory if it doesn't exist."""
    try:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    except OSError as e:
        # Log an error if directory creation fails, but don't crash immediately
        # The error will likely occur later when trying to write files.
        logging.error(f"Could not create output directory '{DEFAULT_OUTPUT_DIR}': {e}")

# Ensure the default output directory exists when this module is loaded
ensure_output_dir()

# You can add other project-wide configurations below if needed.