# config.py
import os
import logging
import yaml # Import YAML
from pathlib import Path

# Setup basic logging early for loading process itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration Loading ---
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

DEFAULT_CONFIG = {
    # Sensible defaults matching the structure in config.yaml
    "ollama": { "base_url": "http://localhost:11434", "model": "llama3:instruct",
                "request_timeout": 450, "max_retries": 2, "retry_delay": 5,
                "max_prompt_chars": 24000, "max_resume_extract_chars": 15000,
                "max_job_desc_chars_in_prompt": 5000 },
    "prompts": { "directory": "analysis/prompts", "resume_extraction_file": "resume_extraction.prompt",
                 "suitability_analysis_file": "suitability_analysis.prompt" },
    "scraping": { "default_sites": ["linkedin", "indeed"], "default_results_limit": 25,
                  "default_hours_old": 72, "default_country_indeed": "usa" },
    "geocoding": { "user_agent": "MyJobSpyAnalysisBot/1.0 (plz_change@example.com)",
                   "cache_file": "output/.geocode_cache.json" },
    "caching": { "resume_cache_dir": "output/.resume_cache" },
    "output": { "directory": "output", "scraped_json_filename": "scraped_jobs.json",
                "analysis_json_filename": "analyzed_jobs.json" },
    "logging": { "level": "INFO" }
}

def _deep_update(source, overrides):
    """Recursively update a dict with values from another dict."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = _deep_update(source[key], value)
        else:
            source[key] = value
    return source

def load_config():
    config_data = DEFAULT_CONFIG.copy() # Start with defaults
    if CONFIG_FILE.exists():
        log.debug(f"Loading configuration from {CONFIG_FILE}")
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Use deep update for nested dictionaries
                config_data = _deep_update(config_data, yaml_config)
                log.info(f"Successfully loaded and merged configuration from {CONFIG_FILE}")
        except yaml.YAMLError as e: log.error(f"Error parsing YAML {CONFIG_FILE}: {e}. Using defaults.")
        except Exception as e: log.error(f"Error loading config {CONFIG_FILE}: {e}. Using defaults.")
    else: log.warning(f"Config file {CONFIG_FILE} not found. Using default settings.")

    # --- Environment Variable Overrides ---
    config_data["ollama"]["base_url"] = os.getenv("OLLAMA_BASE_URL", config_data["ollama"]["base_url"])
    config_data["ollama"]["model"] = os.getenv("OLLAMA_MODEL", config_data["ollama"]["model"])
    config_data["geocoding"]["user_agent"] = os.getenv("GEOPY_USER_AGENT", config_data["geocoding"]["user_agent"])
    config_data["logging"]["level"] = os.getenv("LOG_LEVEL", config_data["logging"]["level"]).upper()
    # Add integer overrides with error handling
    for key, env_var in [("request_timeout", "OLLAMA_REQUEST_TIMEOUT"),
                         ("max_retries", "OLLAMA_MAX_RETRIES"),
                         ("retry_delay", "OLLAMA_RETRY_DELAY"),
                         ("max_prompt_chars", "MAX_PROMPT_CHARS"),
                         ("max_resume_extract_chars", "MAX_RESUME_EXTRACT_CHARS"),
                         ("max_job_desc_chars_in_prompt", "MAX_JOB_DESC_CHARS_IN_PROMPT")]:
        env_val = os.getenv(env_var)
        if env_val:
            try: config_data["ollama"][key] = int(env_val)
            except (ValueError, TypeError): log.warning(f"Invalid integer value for env var {env_var}: '{env_val}'")

    return config_data

# Load config globally
try:
    CFG = load_config()
except Exception as e:
     log.critical(f"Failed critical configuration loading: {e}", exc_info=True)
     sys.exit("Configuration loading failed.")

# --- Derived Paths using Pathlib and CFG ---
# Use .get with defaults to prevent KeyErrors if config loading failed partially
OUTPUT_DIR = PROJECT_ROOT / CFG.get('output', {}).get('directory', 'output')
DEFAULT_SCRAPED_JSON = OUTPUT_DIR / CFG.get('output', {}).get('scraped_json_filename', 'scraped_jobs.json')
DEFAULT_ANALYSIS_JSON = OUTPUT_DIR / CFG.get('output', {}).get('analysis_json_filename', 'analyzed_jobs.json')
PROMPTS_DIR = PROJECT_ROOT / CFG.get('prompts', {}).get('directory', 'analysis/prompts')
RESUME_PROMPT_FILE = CFG.get('prompts', {}).get('resume_extraction_file', 'resume_extraction.prompt')
SUITABILITY_PROMPT_FILE = CFG.get('prompts', {}).get('suitability_analysis_file', 'suitability_analysis.prompt')
RESUME_CACHE_DIR = PROJECT_ROOT / CFG.get('caching', {}).get('resume_cache_dir', 'output/.resume_cache')
GEOCODE_CACHE_FILE = PROJECT_ROOT / CFG.get('geocoding', {}).get('cache_file', 'output/.geocode_cache.json')
LOG_LEVEL = CFG.get('logging', {}).get('level', 'INFO').upper() # For use before full logging setup in run_pipeline

# --- Function to create output/cache dirs ---
def ensure_required_dirs():
    """Ensures necessary output and cache directories exist."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RESUME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        GEOCODE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e: log.error(f"Could not create directories: {e}", exc_info=True)

# Ensure directories exist when module is loaded
ensure_required_dirs()