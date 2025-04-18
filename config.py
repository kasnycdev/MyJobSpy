import os
import logging

# --- General Settings ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_SCRAPED_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "scraped_jobs.json")
DEFAULT_ANALYSIS_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "analyzed_jobs.json")

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1") # Or "mistral:instruct", "phi3:instruct", "llama3:instruct", "openthinker", "deepseek-r1:14b", "llama3.1"
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", 600)) # Increased timeout
OLLAMA_MAX_RETRIES = 3
OLLAMA_RETRY_DELAY = 5 # Base delay in seconds for exponential backoff

# --- Analysis Prompts ---
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "prompts")
RESUME_PROMPT_FILE = "resume_extraction.prompt"
SUITABILITY_PROMPT_FILE = "suitability_analysis.prompt"
# Context window approximation (adjust based on model, very rough estimate)
# This is complex to get right without tokenizing - start large and tune if needed
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", 24000)) # ~8k tokens for Llama3 base prompt + data

# --- JobSpy Scraping Defaults ---
# Sites supported by jobspy (check jobspy documentation for the latest list)
DEFAULT_SCRAPE_SITES = ["linkedin", "indeed", "zip_recruiter", "glassdoor"]
DEFAULT_RESULTS_LIMIT = 100
DEFAULT_HOURS_OLD = 72 # 3 days
DEFAULT_COUNTRY_INDEED = "United States"

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO # Default level
LOG_FORMAT = "%(message)s" # Let Rich handle detailed formatting
LOG_DATE_FORMAT = "[%X]"

# --- Function to create output dir ---
def ensure_output_dir():
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Ensure the default output directory exists when this module is loaded
ensure_output_dir()