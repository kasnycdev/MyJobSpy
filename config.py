
---

**4. Update `config.py` (Now Loads from YAML)**

```python
import os
import logging
import yaml # Import YAML
from pathlib import Path

log = logging.getLogger(__name__)

# --- Configuration Loading ---
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

DEFAULT_CONFIG = {
    # Provide sensible defaults in case YAML loading fails or keys are missing
    "ollama": { "base_url": "http://localhost:11434", "model": "llama3:instruct",
                "request_timeout": 450, "max_retries": 2, "retry_delay": 5, "max_prompt_chars": 24000 },
    "prompts": { "directory": "analysis/prompts", "resume_extraction_file": "resume_extraction.prompt",
                 "suitability_analysis_file": "suitability_analysis.prompt" }, # Default name
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
            with open(CONFIG_FILE, 'r') as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Deep merge - simple version, might need more robust merging for nested dicts
                for key, value in yaml_config.items():
                    if key in config_data and isinstance(config_data[key], dict) and isinstance(value, dict):
                         config_data[key].update(value)
                    else:
                         config_data[key] = value
        except Exception as e:
            log.error(f"Error loading configuration from {CONFIG_FILE}: {e}. Using default settings.")
    else:
        log.warning(f"Configuration file {CONFIG_FILE} not found. Using default settings.")

    # Override with Environment Variables (Example for Ollama)
    config_data["ollama"]["base_url"] = os.getenv("OLLAMA_BASE_URL", config_data["ollama"]["base_url"])
    config_data["ollama"]["model"] = os.getenv("OLLAMA_MODEL", config_data["ollama"]["model"])
    config_data["geocoding"]["user_agent"] = os.getenv("GEOPY_USER_AGENT", config_data["geocoding"]["user_agent"])

    return config_data

# Load config globally for other modules to import
CFG = load_config()

# --- Access specific config values like this: CFG['ollama']['model'] ---

# --- Derived Paths ---
OUTPUT_DIR = PROJECT_ROOT / CFG['output']['directory']
DEFAULT_SCRAPED_JSON = OUTPUT_DIR / CFG['output']['scraped_json_filename']
DEFAULT_ANALYSIS_JSON = OUTPUT_DIR / CFG['output']['analysis_json_filename']
PROMPTS_DIR = PROJECT_ROOT / CFG['prompts']['directory']
RESUME_PROMPT_FILE = CFG['prompts']['resume_extraction_file']
SUITABILITY_PROMPT_FILE = CFG['prompts']['suitability_analysis_file'] # Use the name from config
RESUME_CACHE_DIR = PROJECT_ROOT / CFG['caching']['resume_cache_dir']
GEOCODE_CACHE_FILE = PROJECT_ROOT / CFG['geocoding']['cache_file']

# --- Function to create output/cache dirs ---
def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESUME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure directory for geocode cache exists if specified within output
    geocode_cache_parent = GEOCODE_CACHE_FILE.parent
    if geocode_cache_parent != PROJECT_ROOT: # Avoid trying to create project root
        geocode_cache_parent.mkdir(parents=True, exist_ok=True)

# Ensure directories exist when this module is loaded
ensure_directories()