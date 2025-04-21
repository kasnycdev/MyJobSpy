# analysis/analyzer.py
import ollama # Keep sync client for list/pull
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

# --- Load Prompts (Unchanged conceptually, uses config vars) ---
def load_prompt(filename: str) -> str:
    """Loads a prompt template from the configured prompts directory."""
    path = PROMPTS_DIR / filename # config.py now uses Path objects
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError: log.error(f"Prompt file not found: {path}"); raise
    except Exception as e: log.error(f"Error reading prompt file {path}: {e}"); raise

# --- Main Analyzer Class ---
class ResumeAnalyzer:
    """Handles interaction with Ollama for resume and job analysis (now async)."""

    def __init__(self):
        """Initializes synchronous and asynchronous clients and loads prompts."""
        # Use httpx.AsyncClient for async requests
        # Set limits based on potential server capacity (adjust as needed)
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        self.async_client = httpx.AsyncClient(
            base_url=CFG['ollama']['base_url'],
            timeout=CFG['ollama']['request_timeout'],
            limits=limits
        )
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
        except Exception as e:
            log.critical(f"Failed to load critical prompt files: {e}", exc_info=True)
            raise # Re-raise critical error

        self._check_connection_and_model() # Sync check at init

    # --- _check_connection_and_model remains sync ---
    def _check_connection_and_model(self):
        """Checks Ollama connection and ensures configured model is available using sync client."""
        if not self.sync_client:
            log.error("Sync Ollama client not initialized. Skipping connection/model check.")
            # Consider adding a basic async check here if sync client fails completely
            return
        try:
            log.info(f"Checking Ollama connection at {CFG['ollama']['base_url']}...")
            self.sync_client.ps()
            log.info("Ollama connection successful (sync check).")
            log.info("Fetching list of local Ollama models...")
            ollama_list_response = self.sync_client.list()
            log.debug(f"Raw Ollama list response: {ollama_list_response}")
            models_data = ollama_list_response.get('models', [])
            local_models = []
            # ... (rest of the model parsing and pulling logic using self.sync_client - unchanged) ...
            for m in models_data:
                if hasattr(m, 'model') and isinstance(m.model, str) and m.model: local_models.append(m.model)
                else: log.warning(f"Could not extract model name from item: {m}")
            log.info(f"Parsed local models: {local_models}")
            if CFG['ollama']['model'] not in local_models:
                log.warning(f"Model '{CFG['ollama']['model']}' not found locally. Attempting pull...")
                try:
                    self._pull_model_with_progress(CFG['ollama']['model'])
                    # Re-verify
                    time.sleep(1) # Give Ollama a moment
                    updated_list = self.sync_client.list().get('models', [])
                    updated_names = [m.model for m in updated_list if hasattr(m, 'model')]
                    if CFG['ollama']['model'] not in updated_names:
                         raise ConnectionError(f"Pull seemed complete but model '{CFG['ollama']['model']}' not listed.")
                except Exception as pull_err:
                     log.error(f"Failed to pull model '{CFG['ollama']['model']}': {pull_err}", exc_info=True)
                     raise ConnectionError(f"Model '{CFG['ollama']['model']}' unavailable/pull failed.") from pull_err
            log.info(f"Using configured Ollama model: {CFG['ollama']['model']}")
        except Exception as e:
             log.error(f"Error during sync Ollama connection/model check: {e}", exc_info=True)
             # Don't raise here maybe, let async calls fail later if connection truly down
             # raise ConnectionError(f"Ollama sync check failed: {e}") from e

    # --- _pull_model_with_progress remains sync ---
    def _pull_model_with_progress(self, model_name: str):
         # ... (Keep the existing sync logic using self.sync_client.pull) ...
         pass # Placeholder - assume this exists and works

    # --- _call_ollama becomes ASYNC ---
    async def _call_ollama(self, prompt: str, request_context: str = "request") -> Optional[Dict[str, Any]]:
        """ASYNC Calls Ollama API via httpx with retry logic, expects JSON."""
        log.debug(f"Sending ASYNC {request_context} to Ollama model {CFG['ollama']['model']}. Prompt length: {len(prompt)} chars.")
        if len(prompt) > CFG['ollama']['max_prompt_chars']:
            log.warning(f"Prompt length ({len(prompt)}) exceeds threshold ({CFG['ollama']['max_prompt_chars']}). Context issues may occur for {request_context}.")

        last_exception = None
        request_data = {
            "model": CFG['ollama']['model'], "messages": [{"role": "user", "content": prompt}],
            "format": "json", "stream": False, "options": {"temperature": 0.1} }
        request_url = "/api/chat"

        for attempt in range(CFG['ollama']['max_retries'] + 1): # Corrected range for retries
            try:
                response = await self.async_client.post(request_url, json=request_data)
                response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx

                # Attempt to parse JSON directly from response
                try:
                    response_json = response.json()
                    content = response_json.get('message', {}).get('content', '')
                    if not content:
                         log.warning(f"Ollama response message content is empty for {request_context} (Attempt {attempt + 1}). Response: {response_json}")
                         # Treat as potentially recoverable error for retry
                         raise ValueError("Empty message content received")

                    log.debug(f"Ollama raw response content for {request_context} (first 500 chars): {content[:500]}...")

                    # Basic JSON Repair Attempt (Only if initial parse fails)
                    try:
                        result = json.loads(content.strip()) # Try parsing the content string
                        log.debug(f"Successfully parsed JSON response from Ollama for {request_context}.")
                        return result
                    except json.JSONDecodeError as json_err:
                        log.warning(f"Initial JSON decode failed (Attempt {attempt + 1}) for {request_context}: {json_err}. Trying basic cleanup...")
                        content_strip = content.strip()
                        if content_strip.startswith("```json"): content_strip = content_strip[7:]
                        if content_strip.endswith("```"): content_strip = content_strip[:-3]
                        content_strip = content_strip.strip()
                        # Add more robust repair logic here if needed (e.g., regex for trailing commas)
                        try:
                            result = json.loads(content_strip)
                            log.warning(f"Parsed JSON successfully after basic cleanup for {request_context}.")
                            return result
                        except json.JSONDecodeError as final_json_err:
                             log.error(f"JSON decode failed even after cleanup (Attempt {attempt + 1}) for {request_context}: {final_json_err}")
                             log.debug(f"Problematic Ollama response content for {request_context}: {content}")
                             last_exception = final_json_err # Keep the final error

                except json.JSONDecodeError as resp_json_err:
                     # Error decoding the *response wrapper* itself
                     log.error(f"Error decoding Ollama response wrapper JSON (Attempt {attempt + 1}) for {request_context}: {resp_json_err}")
                     log.debug(f"Raw response text: {response.text[:500]}")
                     last_exception = resp_json_err

            except httpx.TimeoutException as timeout_err: log.warning(f"Ollama request timed out (Attempt {attempt + 1}) for {request_context}: {timeout_err}"); last_exception = timeout_err
            except httpx.RequestError as req_err: log.warning(f"Ollama request error (Attempt {attempt + 1}) for {request_context}: {req_err}"); last_exception = req_err
            except httpx.HTTPStatusError as status_err: log.warning(f"Ollama HTTP error {status_err.response.status_code} (Attempt {attempt + 1}) for {request_context}: {status_err.response.text[:200]}"); last_exception = status_err
            except ValueError as val_err: # Catch specific errors like empty content
                 log.warning(f"Data processing error (Attempt {attempt+1}) for {request_context}: {val_err}"); last_exception = val_err
            except Exception as e: log.error(f"Unexpected error calling Ollama (Attempt {attempt + 1}) for {request_context}: {e}", exc_info=True); last_exception = e

            # --- Retry Logic ---
            if attempt < CFG['ollama']['max_retries'] -1: # Check retries left
                delay = CFG['ollama']['retry_delay'] * (2 ** attempt)
                log.info(f"Retrying Ollama call for {request_context} in {delay:.1f} seconds...")
                await asyncio.sleep(delay) # Use asyncio.sleep
            else:
                 log.error(f"Ollama call failed after {CFG['ollama']['max_retries']} attempts for {request_context}.")
                 if last_exception: log.error(f"Last error encountered: {last_exception}")
        return None # Failed after all retries

    # --- extract_resume_data becomes ASYNC, adds TRUNCATION ---
    async def extract_resume_data(self, resume_text: str) -> Optional[ResumeData]:
        """ASYNC Extracts structured data from resume text using the LLM."""
        MAX_CHARS = CFG['ollama']['max_prompt_chars'] # Use general limit for simplicity or define specific resume limit
        if not resume_text or not resume_text.strip(): log.warning("Resume text empty."); return None

        # --- Truncation ---
        if len(resume_text) > MAX_CHARS:
            log.warning(f"Resume text ({len(resume_text)} chars) exceeds general limit ({MAX_CHARS}). Truncating for extraction.")
            resume_text_for_prompt = resume_text[:MAX_CHARS]
        else: resume_text_for_prompt = resume_text

        prompt = self.resume_prompt_template.format(resume_text=resume_text_for_prompt)
        log.info("Requesting resume data extraction from LLM...")
        extracted_json = await self._call_ollama(prompt, request_context="resume extraction")

        if extracted_json:
            try:
                if isinstance(extracted_json, dict):
                     resume_data = ResumeData(**extracted_json)
                     log.info("Successfully parsed extracted resume data.")
                     log.debug(f"Extracted skills: T:{len(resume_data.technical_skills)} M:{len(resume_data.management_skills)}")
                     log.debug(f"Extracted experience years: {resume_data.total_years_experience}")
                     return resume_data
                else: log.error(f"LLM response for resume not dict: {type(extracted_json)}"); return None
            except Exception as e: # Catch Pydantic validation errors too
                log.error(f"Failed to validate extracted resume data: {e}", exc_info=True); log.error(f"Invalid JSON received for resume: {extracted_json}"); return None
        else: log.error("Failed to get valid JSON from LLM for resume extraction."); return None

    # --- analyze_suitability becomes ASYNC, adds TRUNCATION ---
    async def analyze_suitability(self, resume_data: ResumeData, job_data: Dict[str, Any]) -> Optional[JobAnalysisResult]:
        """ASYNC Analyzes job suitability against resume data using the LLM."""
        job_title = job_data.get('title', 'N/A') # Get title early for logging
        if not resume_data: log.warning(f"Missing structured resume data for job {job_title}."); return None
        job_desc = job_data.get("description", "")
        if not job_data or not job_desc: log.warning(f"Missing job data or description for job: {job_title}. Skipping."); return None

        try:
            # Prepare JSON strings
            resume_data_json_str = resume_data.model_dump_json(indent=2)
            # Serialize job data once, create truncated version if needed
            temp_job_data = job_data.copy()
            job_data_json_str = json.dumps(temp_job_data, indent=2, default=str)

            # --- Estimate length and Truncate Job Description if needed ---
            # Very rough estimate, assumes prompt template is ~1500 chars
            PROMPT_TEMPLATE_OVERHEAD = 2000
            resume_len = len(resume_data_json_str)
            job_len_no_desc = len(json.dumps({k:v for k,v in temp_job_data.items() if k != 'description'}, default=str))
            desc_len = len(job_desc)
            estimated_total_len = PROMPT_TEMPLATE_OVERHEAD + resume_len + job_len_no_desc + desc_len

            if estimated_total_len > CFG['ollama']['max_prompt_chars']:
                chars_over = estimated_total_len - CFG['ollama']['max_prompt_chars']
                keep_desc_len = max(100, desc_len - chars_over) # Keep at least 100 chars
                if keep_desc_len < desc_len:
                    log.warning(f"Truncating description for '{job_title}' from {desc_len} to {keep_desc_len} chars due to estimated prompt length ({estimated_total_len} > {CFG['ollama']['max_prompt_chars']}).")
                    temp_job_data["description"] = job_desc[:keep_desc_len] + "..."
                    # Re-serialize only if truncated
                    job_data_json_str = json.dumps(temp_job_data, indent=2, default=str)
                else:
                    # This means resume was likely the main contributor to length
                    log.warning(f"Estimated prompt length ({estimated_total_len}) exceeds limit but description already short. Check resume size/prompt complexity for '{job_title}'.")

            prompt = self.suitability_prompt_template.format(
                resume_data_json=resume_data_json_str,
                job_data_json=job_data_json_str # Use potentially truncated job data JSON string
            )
        except Exception as e:
            log.error(f"Error preparing data for suitability prompt for '{job_title}': {e}", exc_info=True)
            return None

        log.info(f"Requesting suitability analysis from LLM for job: {job_title}")
        combined_json_response = await self._call_ollama(prompt, request_context=f"suitability analysis for '{job_title}'")

        if not combined_json_response or not isinstance(combined_json_response, dict):
            log.error(f"Failed to get valid JSON dict response from LLM for suitability analysis for '{job_title}'.")
            # Log raw response ONLY in debug for potentially large/sensitive data
            log.debug(f"Raw response received from Ollama for '{job_title}': {combined_json_response}")
            return None

        analysis_data = combined_json_response.get("analysis")
        if not analysis_data or not isinstance(analysis_data, dict):
            log.error(f"LLM response JSON did not contain a valid 'analysis' dict for '{job_title}'.")
            log.debug(f"Full LLM response received for '{job_title}': {combined_json_response}")
            return None

        try:
            analysis_result = JobAnalysisResult(**analysis_data)
            log.info(f"Suitability score for '{job_title}': {analysis_result.suitability_score}%")
            return analysis_result
        except Exception as e: # Catch Pydantic validation errors
            log.error(f"Failed to validate LLM analysis result for '{job_title}': {e}", exc_info=True)
            log.error(f"Invalid 'analysis' JSON structure received for '{job_title}': {analysis_data}")
            return None

    # --- Add method to close async client ---
    async def close(self):
        """Closes the async HTTP client."""
        if hasattr(self, 'async_client') and self.async_client:
             log.debug("Closing async Ollama client.")
             await self.async_client.aclose()