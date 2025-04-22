# config.py
import yaml # Requires PyYAML installation
import os
import logging
from pathlib import Path
from typing import Any # Import Any for type hinting

log = logging.getLogger(__name__)

# --- Constants defining paths ---
PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" # Default relative path

# --- Default Settings (Fallback if YAML is missing fields) ---
DEFAULT_SETTINGS = {
    "output_dir": "output", # Keep as relative path here
    "scraped_jobs_filename": "scraped_jobs.json",
    "analysis_filename": "analyzed_jobs.json",
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama3:instruct",
        "request_timeout": 450,
        "max_retries": 2,
        "retry_delay": 5,
    },
    "analysis": {
        "prompts_dir": "analysis/prompts",
        "resume_prompt_file": "resume_extraction.prompt",
        "suitability_prompt_file": "suitability_analysis.prompt",
        "max_prompt_chars": 24000,
    },
    "scraping": {
        "default_sites": ["linkedin", "indeed"],
        "default_results_limit": 25,
        "default_hours_old": 72,
        "default_country_indeed": "usa",
    },
    "geocoding": {
        "geopy_user_agent": "MyJobSpyAnalysisBot/1.2 (PLEASE_UPDATE_EMAIL@example.com)"
    },
    "logging": {
        "level": "INFO",
        "format": "%(message)s",
        "date_format": "[%X]",
    }
}

def _deep_update(source, overrides):
    """Recursively update a dict with values from another dict."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = _deep_update(source[key], value)
        else:
            source[key] = value
    return source

def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Loads configuration from YAML file, merging deeply with defaults."""
    settings = DEFAULT_SETTINGS.copy() # Start with defaults

    if config_path.exists():
        log.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)

            if user_config and isinstance(user_config, dict):
                # Use deep merge
                settings = _deep_update(settings, user_config)
                log.info("Successfully loaded and merged configuration.")
            elif user_config is None:
                 log.warning(f"Configuration file {config_path} is empty or invalid. Using default settings.")
            else:
                 log.warning(f"Configuration file {config_path} has invalid format (not a dictionary). Using default settings.")
        except yaml.YAMLError as e:
            log.error(f"Error parsing YAML configuration file {config_path}: {e}. Using default settings.")
        except Exception as e:
            log.error(f"Error loading configuration file {config_path}: {e}. Using default settings.", exc_info=True)
    else:
        log.warning(f"Configuration file not found at {config_path}. Using default settings.")

    # --- Make paths absolute based on PROJECT_ROOT ---
    settings["output_dir"] = str(PROJECT_ROOT / settings.get("output_dir", "output"))
    if "analysis" in settings and "prompts_dir" in settings["analysis"]:
        settings["analysis"]["prompts_dir"] = str(PROJECT_ROOT / settings["analysis"]["prompts_dir"])

    # --- Ensure output dir exists ---
    try:
        os.makedirs(settings["output_dir"], exist_ok=True)
    except OSError as e:
        log.error(f"Could not create output directory '{settings['output_dir']}': {e}")

    # --- Add derived absolute paths for convenience ---
    settings["scraped_jobs_path"] = os.path.join(settings["output_dir"], settings.get("scraped_jobs_filename", "scraped_jobs.json"))
    settings["analysis_output_path"] = os.path.join(settings["output_dir"], settings.get("analysis_filename", "analyzed_jobs.json"))
    if "analysis" in settings:
        settings["analysis"]["resume_prompt_path"] = os.path.join(settings["analysis"]["prompts_dir"], settings["analysis"]["resume_prompt_file"])
        settings["analysis"]["suitability_prompt_path"] = os.path.join(settings["analysis"]["prompts_dir"], settings["analysis"]["suitability_prompt_file"])

    return settings

# --- Load settings ONCE when module is imported ---
settings = load_config()

# --- Optional: Functions to access settings easily ---
def get_setting(key_path: str, default: Any = None) -> Any:
    """Access nested settings using dot notation, e.g., 'ollama.model'."""
    keys = key_path.split('.')
    value = settings
    try:
        for key in keys:
            if isinstance(value, dict):
                 value = value[key]
            else: # Prevent TypeError if trying to index a non-dict
                 log.warning(f"Cannot access key '{key}' in non-dictionary element while looking for '{key_path}'.")
                 return default
        return value
    except KeyError:
        log.debug(f"Setting '{key_path}' not found, returning default '{default}'.")
        return default