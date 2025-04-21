# config.py
import os
import logging
import yaml # Import YAML
from pathlib import Path

# Setup logger *early* in case loading fails
# Configure basic logging first, it will be reconfigured later if Rich is available
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
log = logging.getLogger(__name__) # Get logger instance using __name__

# --- Configuration Loading ---
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

DEFAULT_CONFIG = {
    # Provide sensible defaults in case YAML loading fails or keys are missing
    "ollama": { "base_url": "http://localhost:11434", "model": "llama3:instruct",
                "request_timeout": 450, "max_retries": 2, "retry_delay": 5, "max_prompt_chars": 24000 },
    "prompts": { "directory": "analysis/prompts", "resume_extraction_file": "resume_extraction.prompt",
                 "suitability_analysis_file": "suitability_analysis.prompt" }, # Make sure this filename is correct!
    "scraping": { "default_sites": ["linkedin", "indeed"], "default_results_limit": 25,
                  "default_hours_old": 72, "default_country_indeed": "usa" },
    "geocoding": { "user_agent": "MyJobSpyAnalysisBot/1.0 (plz_set_in_config_or_env@example.com)",
                   "cache_file": "output/.geocode_cache.json" },
    "caching": { "resume_cache_dir": "output/.resume_cache" },
    "output": { "directory": "output", "scraped_json_filename": "scraped_jobs.json",
                "analysis_json_filename": "analyzed_jobs.json" },
    "logging": { "level": "INFO" }
}

def load_config():
    config_data = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        log.debug(f"Loading configuration from {CONFIG_FILE}")
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Simple merge (overwrites entire top-level keys if present in yaml)
                # For deeper merging, a utility function would be needed
                config_data.update(yaml_config)
                log.info(f"Successfully loaded configuration from {CONFIG_FILE}")
        except yaml.YAMLError as e:
            log.error(f"Error parsing YAML configuration file {CONFIG_FILE}: {e}. Using default settings.")
        except Exception as e:
            log.error(f"Error loading configuration from {CONFIG_FILE}: {e}. Using default settings.")
    else:
        log.warning(f"Configuration file {CONFIG_FILE} not found. Using default settings.")

    # --- Environment Variable Overrides (Examples) ---
    # Ollama
    config_data["ollama"]["base_url"] = os.getenv("OLLAMA_BASE_URL", config_data["ollama"]["base_url"])
    config_data["ollama"]["model"] = os.getenv("OLLAMA_MODEL", config_data["ollama"]["model"])
    if os.getenv("OLLAMA_REQUEST_TIMEOUT"):
        try: config_data["ollama"]["request_timeout"] = int(os.getenv("OLLAMA_REQUEST_TIMEOUT"))
        except ValueError: log.warning("Invalid OLLAMA_REQUEST_TIMEOUT env var value.")
    # Geocoding
    config_data["geocoding"]["user_agent"] = os.getenv("GEOPY_USER_AGENT", config_data["geocoding"]["user_agent"])
    # Logging
    config_data["logging"]["level"] = os.getenv("LOG_LEVEL", config_data["logging"]["level"]).upper()
    # --- Add more overrides as needed ---

    # Basic validation for critical settings
    if not config_data.get("ollama") or not config_data.get("prompts"):
         log.critical("Core 'ollama' or 'prompts' configuration sections missing. Exiting.")
         raise ValueError("Missing critical configuration sections in config.py/config.yaml")

    return config_data

# Load config globally for other modules to import
try:
    CFG = load_config()
except Exception as e:
     log.critical(f"Failed to load configuration during startup: {e}", exc_info=True)
     # Exit here if config loading fails critically
     import sys
     sys.exit("Configuration loading failed.")


# --- Derived Paths using Pathlib and CFG ---
# Ensure keys exist before accessing using .get with fallback to avoid KeyErrors if loading failed partially
OUTPUT_DIR = PROJECT_ROOT / CFG.get('output', {}).get('directory', 'output')
DEFAULT_SCRAPED_JSON = OUTPUT_DIR / CFG.get('output', {}).get('scraped_json_filename', 'scraped_jobs.json')
DEFAULT_ANALYSIS_JSON = OUTPUT_DIR / CFG.get('output', {}).get('analysis_json_filename', 'analyzed_jobs.json')
PROMPTS_DIR = PROJECT_ROOT / CFG.get('prompts', {}).get('directory', 'analysis/prompts')
RESUME_PROMPT_FILE = CFG.get('prompts', {}).get('resume_extraction_file', 'resume_extraction.prompt')
SUITABILITY_PROMPT_FILE = CFG.get('prompts', {}).get('suitability_analysis_file', 'suitability_analysis.prompt')
RESUME_CACHE_DIR = PROJECT_ROOT / CFG.get('caching', {}).get('resume_cache_dir', 'output/.resume_cache')
GEOCODE_CACHE_FILE = PROJECT_ROOT / CFG.get('geocoding', {}).get('cache_file', 'output/.geocode_cache.json')

# --- Function to create output/cache dirs ---
def ensure_required_dirs():
    """Ensures necessary output and cache directories exist."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RESUME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        geocode_cache_parent = GEOCODE_CACHE_FILE.parent
        if geocode_cache_parent != PROJECT_ROOT: # Avoid trying to create project root
            geocode_cache_parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error(f"Could not create necessary directories: {e}", exc_info=True)
        # Optionally raise the error or exit if directories are critical
        # raise

# Ensure directories exist when module is loaded
ensure_required_dirs()

# --- Make LOG_LEVEL accessible directly for basicConfig in run_pipeline ---
LOG_LEVEL = CFG.get('logging', {}).get('level', 'INFO')