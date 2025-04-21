# analysis/analyzer.py
import ollama # Keep sync client for list/pull at startup
import json
import logging
import time
import os
import asyncio # Import asyncio
import httpx   # Import httpx for async client
from typing import Dict, Optional, Any

from analysis.models import ResumeData, JobAnalysisResult # AnalyzedJob not needed here
# Use CFG for configuration values loaded from YAML
from config import CFG, PROMPTS_DIR, RESUME_PROMPT_FILE, SUITABILITY_PROMPT_FILE

log = logging.getLogger(__name__)

# --- Load Prompts ---
def load_prompt(filename: str) -> str:
    """Loads a prompt template from the configured prompts directory."""
    path = PROMPTS_DIR / filename # config.py now uses Path objects
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError:
        log.error(f"Prompt file not found: {path}")
        raise # Re-raise critical error
    except Exception as e:
        log.error(f"Error reading prompt file {path}: {e}")
        raise # Re-raise critical error

# --- Main Analyzer Class ---
class ResumeAnalyzer:
    """Handles interaction with Ollama for resume and job analysis (now async)."""

    def __init__(self):
        """Initializes synchronous and asynchronous clients and loads prompts."""
        # Use httpx.AsyncClient for async requests
        # Set limits based on potential server capacity (adjust as needed)
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        try:
            self.async_client = httpx.AsyncClient(
                base_url=CFG['ollama']['base_url'],
                timeout=CFG['ollama']['request_timeout'],
                limits=limits
            )
        except KeyError as e:
             log.critical(f"Missing critical Ollama configuration in config.yaml: {e}")
             raise ValueError(f"Missing critical Ollama configuration: {e}") from e
        except Exception as e:
             log.critical(f"Failed to initialize async httpx client: {e}", exc_info=True)
             raise RuntimeError("Failed to setup async HTTP client") from e

        # Keep sync client ONLY for non-performance-critical tasks like list/pull at startup
        try:
            self.sync_client = ollama.Client(host=CFG['ollama']['base_url'])
        except Exception as e:
            log.warning(f"Could not initialize synchronous Ollama client (used for setup): {e}")
            self.sync_client = None

        # Load prompts
        try:
            self.resume_prompt_template = load_prompt(RESUME_PROMPT_FILE)
            self.suitability_prompt_template = load_prompt(SUITABILITY_PROMPT_FILE)
            log.info("Prompts loaded successfully.")
        except Exception as e:
            log.critical(f"Failed to load critical prompt files from {PROMPTS_DIR}: {e}", exc_info=True)
            raise # Re-raise critical error

        self._check_connection_and_model() # Sync check at init

    # --- _check_connection_and_model remains sync ---
    def _check_connection_and_model(self):
        """Checks Ollama connection and ensures configured model is available using sync client."""
        if not self.sync_client:
            log.error("Sync Ollama client not initialized. Skipping connection/model check.")
            return # Cannot perform check without sync client

        try:
            target_model = CFG['ollama']['model']
            log.info(f"Checking Ollama connection at {CFG['ollama']['base_url']} for model '{target_model}'...")
            self.sync_client.ps() # Basic check
            log.info("Ollama connection successful (sync check).")

            log.info("Fetching list of local Ollama models...")
            ollama_list_response = self.sync_client.list()
            log.debug(f"Raw Ollama list() response content: {ollama_list_response}")

            models_data = ollama_list_response.get('models', [])
            if not isinstance(models_data, list):
                log.error(f"Ollama list response 'models' key not a list: {type(models_data)}")
                models_data = []

            local_models = []
            for m in models_data:
                if hasattr(m, 'model') and isinstance(m.model, str) and m.model:
                    local_models.append(m.model)
                elif hasattr(m, 'name') and isinstance(m.name, str) and m.name: # Fallback check for 'name' just in case
                     log.warning(f"Found model object without 'model' attribute, using 'name': {m}")
                     local_models.append(m.name)
                else:
                    log.warning(f"Could not extract model name from item in Ollama models list: {m} (Type: {type(m)})")

            log.info(f"Parsed local models: {local_models}")

            if target_model not in local_models:
                log.warning(f"Model '{target_model}' not found locally. Attempting pull (this may take time)...")
                try:
                    self._pull_model_with_progress(target_model)
                    # Re-verify after pulling
                    log.info("Re-fetching model list after pull attempt...")
                    time.sleep(2) # Give Ollama a moment to register
                    updated_list = self.sync_client.list().get('models', [])
                    updated_names = [m.model for m in updated_list if hasattr(m, 'model')]
                    log.debug(f"Model list after pull: {updated_names}")
                    if target_model not in updated_names:
                         # Try final check with slight delay
                         time.sleep(3)
                         final_list = self.sync_client.list().get('models', [])
                         final_names = [m.model for m in final_list if hasattr(m, 'model')]
                         if target_model not in final_names:
                             log.error(f"Model '{target_model}' still not found after pull/delay.")
                             raise ConnectionError(f"Ollama model pull seemed complete but model '{target_model}' not listed.")
                         else:
                              log.info("Model found after pull and delay.")
                    else:
                         log.info("Model found after pull.")

                except Exception as pull_err:
                    log.error(f"Failed to pull or verify Ollama model '{target_model}': {pull_err}", exc_info=True)
                    raise ConnectionError(f"Required Ollama model '{target_model}' unavailable and pull failed.") from pull_err
            else:
                log.info(f"Using configured Ollama model: {target_model}")

        except (ollama.ResponseError, ConnectionError, TimeoutError) as conn_e:
             log.error(f"Failed to connect or communicate with Ollama at {CFG['ollama']['base_url']}. Is Ollama running? Error: {conn_e}", exc_info=True)
             # Don't raise here, let async calls handle connection errors later
             # raise ConnectionError(f"Ollama sync check failed: {conn_e}") from conn_e
        except Exception as e:
            log.error(f"An unexpected error occurred during Ollama sync check: {e}", exc_info=True)
            # Don't raise here

    # --- _pull_model_with_progress remains sync ---
    def _pull_model_with_progress(self, model_name: str):
        """Pulls an Ollama model, showing progress using sync client."""
        if not self.sync_client: log.error("Sync client needed for pull, but not available."); return
        current_digest, status = "", ""
        try:
            for progress in self.sync_client.pull(model_name, stream=True):
                digest = progress.get("digest", "")
                if digest != current_digest and current_digest != "": print()
                if digest:
                    current_digest = digest
                    status = progress.get('status', '')
                    if status: print(f"Pulling {model_name}: {status}", end='\r')
                else:
                    status = progress.get('status', '')
                    if status and 'pulling' not in status.lower(): print(f"Pulling {model_name}: {status}")
                if progress.get('error'): raise Exception(f"Pull error: {progress['error']}")
                if 'status' in progress and 'success' in progress['status'].lower():
                    print(); log.info(f"Successfully pulled model {model_name}"); break
        except Exception as e: print(); log.error(f"Error during model pull: {e}"); raise
        finally: print() # Ensure final newline

    # --- _call_ollama becomes ASYNC ---
    async def _call_ollama(self, prompt: str, request_context: str = "request") -> Optional[Dict[str, Any]]:
        """ASYNC Calls Ollama API via httpx with retry logic, expects JSON."""
        # --- Ensure MAX_CHARS is int ---
        try: max_prompt_chars = int(CFG.get('ollama', {}).get('max_prompt_chars', 24000))
        except (ValueError, TypeError): max_prompt_chars = 24000

        log.debug(f"Sending ASYNC {request_context} to Ollama model {CFG['ollama']['model']}. Prompt length: {len(prompt)} chars.")
        if len(prompt) > max_prompt_chars:
            log.warning(f"Prompt length ({len(prompt)}) exceeds threshold ({max_prompt_chars}). Context issues may occur for {request_context}.")

        last_exception = None
        request_data = { "model": CFG['ollama']['model'], "messages": [{"role": "user", "content": prompt}],
                         "format": "json", "stream": False, "options": {"temperature": 0.1} }
        request_url = "/api/chat"
        max_retries = CFG.get('ollama', {}).get('max_retries', 2)
        retry_delay = CFG.get('ollama', {}).get('retry_delay', 5)

        for attempt in range(max_retries + 1):
            try:
                response = await self.async_client.post(request_url, json=request_data)
                response.raise_for_status()
                try:
                    response_json = response.json()
                    content = response_json.get('message', {}).get('content', '')
                    if not content: raise ValueError("Empty message content received")
                    log.debug(f"Ollama raw response for {request_context} (Attempt {attempt+1}, first 500): {content[:500]}...")
                    try: result = json.loads(content.strip()); log.debug("Parsed JSON directly."); return result
                    except json.JSONDecodeError as json_err:
                        log.warning(f"Initial JSON decode failed ({request_context}, Attempt {attempt+1}): {json_err}. Cleaning...")
                        content_strip = content.strip()
                        if content_strip.startswith("```json"): content_strip = content_strip[7:]
                        if content_strip.endswith("```"): content_strip = content_strip[:-3]
                        content_strip = content_strip.strip()
                        # Add more repair attempts here if needed
                        try: result = json.loads(content_strip); log.warning("Parsed JSON after cleanup."); return result
                        except json.JSONDecodeError as final_err: log.error(f"JSON decode failed after cleanup ({request_context}, Attempt {attempt+1}): {final_err}"); last_exception = final_err; log.debug(f"Content: {content}")
                except json.JSONDecodeError as resp_err: log.error(f"Error decoding Ollama response wrapper ({request_context}, Attempt {attempt+1}): {resp_err}"); last_exception = resp_err; log.debug(f"Raw text: {response.text[:500]}")
            except httpx.TimeoutException as err: log.warning(f"Ollama timeout ({request_context}, Attempt {attempt+1}): {err}"); last_exception = err
            except httpx.RequestError as err: log.warning(f"Ollama request error ({request_context}, Attempt {attempt+1}): {err}"); last_exception = err
            except httpx.HTTPStatusError as err: log.warning(f"Ollama HTTP error {err.response.status_code} ({request_context}, Attempt {attempt+1}): {err.response.text[:200]}"); last_exception = err
            except ValueError as err: log.warning(f"Data error ({request_context}, Attempt {attempt+1}): {err}"); last_exception = err
            except Exception as e: log.error(f"Unexpected error in _call_ollama ({request_context}, Attempt {attempt+1}): {e}", exc_info=True); last_exception = e

            if attempt < max_retries: # Check if more retries left
                delay = retry_delay * (2 ** attempt); log.info(f"Retrying Ollama call ({request_context}) in {delay:.1f}s..."); await asyncio.sleep(delay)
            else: log.error(f"Ollama call failed after {max_retries+1} attempts for {request_context}. Last error: {last_exception}")
        return None

    # --- extract_resume_data becomes ASYNC, adds TRUNCATION ---
    async def extract_resume_data(self, resume_text: str) -> Optional[ResumeData]:
        """ASYNC Extracts structured data from resume text using the LLM."""
        try: max_chars_config = CFG.get('ollama', {}).get('max_prompt_chars', 24000); MAX_CHARS = int(max_chars_config)
        except (ValueError, TypeError) as e: log.error(f"Invalid max_prompt_chars: '{max_chars_config}'. Using 24000. Error: {e}"); MAX_CHARS = 24000

        if not resume_text or not resume_text.strip(): log.warning("Resume text empty."); return None
        if len(resume_text) > MAX_CHARS:
            log.warning(f"Resume text ({len(resume_text)}) > limit ({MAX_CHARS}). Truncating."); resume_text_for_prompt = resume_text[:MAX_CHARS]
        else: resume_text_for_prompt = resume_text

        prompt = self.resume_prompt_template.format(resume_text=resume_text_for_prompt)
        log.info("Requesting resume data extraction from LLM...")
        extracted_json = await self._call_ollama(prompt, request_context="resume extraction")

        if extracted_json:
            try:
                if isinstance(extracted_json, dict):
                     resume_data = ResumeData(**extracted_json); log.info("Parsed extracted resume data."); return resume_data
                else: log.error(f"LLM response for resume not dict: {type(extracted_json)}"); return None
            except Exception as e: log.error(f"Failed validation: {e}", exc_info=True); log.error(f"Invalid JSON: {extracted_json}"); return None
        else: log.error("Failed extraction: No valid JSON from LLM."); return None

    # --- analyze_suitability becomes ASYNC, adds TRUNCATION ---
    async def analyze_suitability(self, resume_data: ResumeData, job_data: Dict[str, Any]) -> Optional[JobAnalysisResult]:
        """ASYNC Analyzes job suitability against resume data using the LLM."""
        job_title = job_data.get('title', 'N/A')
        if not resume_data: log.warning(f"Missing resume data for {job_title}."); return None
        job_desc = job_data.get("description", "")
        if not job_data or not job_desc: log.warning(f"Missing job data/desc for {job_title}. Skipping."); return None

        try:
            # Prepare JSON strings and check lengths
            resume_data_json_str = resume_data.model_dump_json(indent=2)
            temp_job_data = job_data.copy() # Work with copy for potential truncation
            job_data_json_str = json.dumps(temp_job_data, indent=2, default=str)

            # --- Truncation Logic ---
            try:
                 max_suitability_prompt_chars = int(CFG.get('ollama', {}).get('max_prompt_chars', 24000))
                 # max_job_desc_chars = int(CFG.get('ollama', {}).get('max_job_desc_chars_in_prompt', 5000)) # Example specific limit for desc
            except (ValueError, TypeError): max_suitability_prompt_chars = 24000; #max_job_desc_chars = 5000

            PROMPT_TEMPLATE_OVERHEAD = 2500 # Increased estimate for safer margin
            resume_len = len(resume_data_json_str)
            # Estimate base job JSON length excluding description
            job_len_no_desc = len(json.dumps({k:v for k,v in temp_job_data.items() if k != 'description'}, default=str))
            desc_len = len(job_desc)
            estimated_total_len = PROMPT_TEMPLATE_OVERHEAD + resume_len + job_len_no_desc + desc_len

            if estimated_total_len > max_suitability_prompt_chars:
                chars_to_remove = estimated_total_len - max_suitability_prompt_chars
                # Calculate chars to keep from description, leaving some buffer
                keep_desc_len = max(100, desc_len - chars_to_remove - 100) # Keep at least 100, remove excess + buffer
                if keep_desc_len < desc_len:
                     log.warning(f"Truncating description for '{job_title}' from {desc_len} to {keep_desc_len} chars due to estimated prompt length ({estimated_total_len} > {max_suitability_prompt_chars}).")
                     temp_job_data["description"] = job_desc[:keep_desc_len] + "..."
                     job_data_json_str = json.dumps(temp_job_data, indent=2, default=str) # Re-serialize truncated
                else: log.warning(f"Est. prompt length {estimated_total_len} > {max_suitability_prompt_chars} for '{job_title}', but desc already short. Check resume/prompt complexity.")

            prompt = self.suitability_prompt_template.format(
                resume_data_json=resume_data_json_str, job_data_json=job_data_json_str )
        except Exception as e: log.error(f"Error preparing data for prompt '{job_title}': {e}", exc_info=True); return None

        log.info(f"Requesting suitability analysis from LLM for job: {job_title}")
        combined_json_response = await self._call_ollama(prompt, request_context=f"suitability '{job_title}'")

        if not combined_json_response or not isinstance(combined_json_response, dict):
            log.error(f"Failed JSON dict response from LLM for '{job_title}'."); log.debug(f"Raw resp: {combined_json_response}"); return None
        analysis_data = combined_json_response.get("analysis")
        if not analysis_data or not isinstance(analysis_data, dict):
            log.error(f"LLM response missing 'analysis' dict for '{job_title}'."); log.debug(f"Full resp: {combined_json_response}"); return None
        try:
            analysis_result = JobAnalysisResult(**analysis_data); log.info(f"Score for '{job_title}': {analysis_result.suitability_score}%"); return analysis_result
        except Exception as e: log.error(f"Failed validation '{job_title}': {e}", exc_info=True); log.error(f"Invalid 'analysis' JSON: {analysis_data}"); return None

    # --- Add method to close async client ---
    async def close(self):
        """Closes the async HTTP client."""
        if hasattr(self, 'async_client') and self.async_client:
             log.debug("Closing async Ollama client.")
             try: await self.async_client.aclose()
             except Exception as e: log.error(f"Error closing httpx client: {e}")