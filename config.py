import os
import logging

# --- General Settings ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_SCRAPED_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "scraped_jobs.json")
DEFAULT_ANALYSIS_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "analyzed_jobs.json")

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct") # Example default
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", 450)) # Increased further
OLLAMA_MAX_RETRIES = 2 # Reduced retries if timeout is long
OLLAMA_RETRY_DELAY = 5

# --- Analysis Prompts ---
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "prompts")
RESUME_PROMPT_FILE = "resume_extraction.prompt"
SUITABILITY_PROMPT_FILE = "suitability_analysis.prompt"
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", 24000))

# --- JobSpy Scraping Defaults ---
DEFAULT_SCRAPE_SITES = ["linkedin", "indeed"] # Reduced default, Glassdoor often problematic
DEFAULT_RESULTS_LIMIT = 25 # Slightly increased default
DEFAULT_HOURS_OLD = 72
DEFAULT_COUNTRY_INDEED = "usa"

# --- Geocoding --- ADDED/MODIFIED ---
# IMPORTANT: Replace placeholder with your actual app name/email or use ENV var
GEOPY_USER_AGENT = os.getenv("GEOPY_USER_AGENT", "MyJobSpyAnalysisApp/1.0 (your_email@example.com)")
# --- END ADDED/MODIFIED ---

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(message)s"
LOG_DATE_FORMAT = "[%X]"

# --- Function to create output dir ---
def ensure_output_dir():
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Ensure the default output directory exists when this module is loaded
ensure_output_dir()