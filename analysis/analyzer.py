import ollama
import json
import logging
import time
import os
import asyncio # Import asyncio
from typing import Dict, Optional, Any

# Models are imported correctly
from analysis.models import ResumeData, JobAnalysisResult, AnalyzedJob
import config

# Setup logger
log = logging.getLogger(__name__)

def load_prompt(filename: str) -> str:
    """Loads a prompt template from the configured prompts directory."""
    path = os.path.join(config.PROMPTS_DIR, filename)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        log.error(f"Prompt file not found: {path}")
        raise
    except Exception as e:
        log.error(f"Error reading prompt file {path}: {e}")
        raise

class ResumeAnalyzer:
    """Handles interaction with Ollama for resume and job analysis (Sync & Async)."""

    def __init__(self):
        # Initialize both sync and async clients
        log.info("Initializing Ollama clients...")
        try:
            self.sync_client = ollama.Client(host=config.OLLAMA_BASE_URL, timeout=config.OLLAMA_REQUEST_TIMEOUT)
            self.async_client = ollama.AsyncClient(host=config.OLLAMA_BASE_URL, timeout=config.OLLAMA_REQUEST_TIMEOUT)
            log.info("Ollama clients initialized.")
        except Exception as e:
             log.critical(f"Failed to initialize Ollama clients: {e}", exc_info=True)
             raise RuntimeError(f"Ollama client initialization failed: {e}") from e

        # Load prompts (sync)
        self.resume_prompt_template = load_prompt(config.RESUME_PROMPT_FILE)
        self.suitability_prompt_template = load_prompt(config.SUITABILITY_PROMPT_FILE)
        # Run initial checks synchronously
        self._check_connection_and_model() # Keep initial check sync

    # --- _check_connection_and_model remains mostly synchronous ---
    def _check_connection_and_model(self):
        """Checks Ollama connection and ensures the configured model is available."""
        try:
            log.info(f"Checking Ollama connection at {config.OLLAMA_BASE_URL}...")
            # Use the sync client for initial checks
            self.sync_client.ps()
            log.info("Ollama connection successful (basic check passed).")
            log.info("Fetching list of local Ollama models...")
            ollama_list_response = self.sync_client.list()
            log.debug(f"Raw Ollama list() response content: {ollama_list_response}")
            models_data = ollama_list_response.get('models', [])
            if not isinstance(models_data, list): models_data = []

            local_models = []
            for m in models_data:
                if hasattr(m, 'model') and isinstance(m.model, str) and m.model: local_models.append(m.model)
                elif isinstance(m, dict) and m.get('name'): log.warning(f"Found dict item: {m}"); local_models.append(m.get('name'))
                else: log.warning(f"Could not extract model name from item: {m}")

            log.info(f"Successfully parsed local models: {local_models}")

            if config.OLLAMA_MODEL not in local_models:
                log.warning(f"Model '{config.OLLAMA_MODEL}' not found locally.")
                log.info(f"Attempting to pull model '{config.OLLAMA_MODEL}'. This may take time...")
                try:
                    # Pulling remains synchronous for startup simplicity
                    self._pull_model_with_progress(config.OLLAMA_MODEL)
                    # Re-verify after pulling
                    log.info("Re-fetching model list after pull...")
                    updated_list_response = self.sync_client.list()
                    updated_models_data = updated_list_response.get('models', [])
                    updated_names = [m_upd.model for m_upd in updated_models_data if hasattr(m_upd, 'model')]
                    log.debug(f"Model list after pull: {updated_names}")
                    if config.OLLAMA_MODEL not in updated_names:
                         log.error(f"Model '{config.OLLAMA_MODEL}' still not found after attempting pull.")
                         raise ConnectionError(f"Ollama model pull seemed complete but model '{config.OLLAMA_MODEL}' not listed.")
                except Exception as pull_err:
                    log.error(f"Failed to pull or verify Ollama model '{config.OLLAMA_MODEL}': {pull_err}", exc_info=True)
                    raise ConnectionError(f"Required Ollama model '{config.OLLAMA_MODEL}' unavailable/pull failed.") from pull_err
            else:
                log.info(f"Using configured Ollama model: {config.OLLAMA_MODEL}")

        except Exception as e:
            log.critical(f"Ollama connection/setup failed: {e}", exc_info=True)
            raise ConnectionError(f"Ollama connection/setup failed: {e}") from e

    # --- _pull_model_with_progress remains synchronous ---
    def _pull_model_with_progress(self, model_name: str):
        # (Content of this function remains the same as previous versions)
        # It uses self.sync_client implicitly via ollama package logic if needed
        current_digest = ""; status = ""
        try:
            # Use ollama.pull which might use the default sync client implicitly
            # Or if issues arise, explicitly use self.sync_client.pull if available
            for progress in ollama.pull(model_name, stream=True): # Assuming ollama.pull uses default sync client
                 digest = progress.get("digest", "");
                 if digest != current_digest and current_digest != "": print()
                 if digest: current_digest = digest; status = progress.get('status', ''); print(f"Pulling {model_name}: {status}", end='\r')
                 else: status = progress.get('status', ''); print(f"Pulling {model_name}: {status}")
                 if progress.get('error'): raise Exception(f"Pull error: {progress['error']}")
                 if 'status' in progress and 'success' in progress['status'].lower(): print(); log.info(f"Successfully pulled model {model_name}"); break
        except Exception as e: print(); log.error(f"Error during model pull: {e}"); raise
        finally: print()


    # --- ASYNC Ollama Call ---
    async def _call_ollama_async(self, prompt: str) -> Optional[Dict[str, Any]]:
        """ASYNC: Calls the Ollama API with retry logic and expects JSON output."""
        log.debug(f"ASYNC: Sending request to Ollama model {config.OLLAMA_MODEL}. Prompt length: {len(prompt)} chars.")
        if len(prompt) > config.MAX_PROMPT_CHARS:
             log.warning(f"Prompt length ({len(prompt)} chars) exceeds threshold ({config.MAX_PROMPT_CHARS}). May risk context window issues.")

        last_exception = None
        for attempt in range(config.OLLAMA_MAX_RETRIES):
            try:
                # Use the async client and await the call
                response = await self.async_client.chat(
                    model=config.OLLAMA_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    format='json',
                    options={'temperature': 0.1}
                )
                content = response['message']['content']
                log.debug(f"ASYNC: Ollama raw response (first 500 chars): {content[:500]}...")

                try:
                    content_strip = content.strip()
                    if content_strip.startswith("```json"): content_strip = content_strip[7:]; content_strip = content_strip[:-3] if content_strip.endswith("```") else content_strip
                    elif content_strip.startswith("```"): content_strip = content_strip[3:]; content_strip = content_strip[:-3] if content_strip.endswith("```") else content_strip
                    result = json.loads(content_strip.strip())
                    log.debug("ASYNC: Successfully parsed JSON response from Ollama.")
                    return result
                except json.JSONDecodeError as json_err:
                    log.warning(f"ASYNC: Failed to decode JSON response (Attempt {attempt + 1}): {json_err}")
                    log.debug(f"ASYNC: Problematic Ollama response content: {content}")
                    last_exception = json_err

            # Catch async compatible errors if client library defines them, else broad network errors
            except (ollama.ResponseError, asyncio.TimeoutError, ConnectionError, TimeoutError) as conn_err: # Added asyncio.TimeoutError
                log.warning(f"ASYNC: Ollama API communication error (Attempt {attempt + 1}): {conn_err}")
                last_exception = conn_err
            except Exception as e:
                log.error(f"ASYNC: Unexpected error calling Ollama API (Attempt {attempt + 1}): {e}", exc_info=True)
                last_exception = e

            if attempt < config.OLLAMA_MAX_RETRIES - 1:
                delay = config.OLLAMA_RETRY_DELAY * (2 ** attempt)
                log.info(f"ASYNC: Retrying Ollama call in {delay:.1f} seconds...")
                await asyncio.sleep(delay) # Use asyncio.sleep
            else:
                 log.error(f"ASYNC: Ollama call failed after {config.OLLAMA_MAX_RETRIES} attempts.")
                 if last_exception: log.error(f"ASYNC: Last error encountered: {last_exception}")
        return None


    # --- ASYNC Resume Extraction ---
    async def extract_resume_data_async(self, resume_text: str) -> Optional[ResumeData]:
        """ASYNC: Extracts structured data from resume text using the LLM."""
        MAX_RESUME_CHARS_FOR_LLM = 15000 # Example limit
        if not resume_text or not resume_text.strip():
            log.warning("Resume text is empty, cannot extract data.")
            return None

        if len(resume_text) > MAX_RESUME_CHARS_FOR_LLM:
            log.warning(f"Resume text length ({len(resume_text)}) exceeds limit ({MAX_RESUME_CHARS_FOR_LLM}). Truncating.")
            resume_text_for_prompt = resume_text[:MAX_RESUME_CHARS_FOR_LLM]
        else:
            resume_text_for_prompt = resume_text

        prompt = self.resume_prompt_template.format(resume_text=resume_text_for_prompt)
        log.info("ASYNC: Requesting resume data extraction from LLM...")
        extracted_json = await self._call_ollama_async(prompt) # Await async call

        if extracted_json:
            try:
                if isinstance(extracted_json, dict):
                     resume_data = ResumeData(**extracted_json)
                     log.info("ASYNC: Successfully parsed extracted resume data.")
                     log.debug(f"Extracted skills: T:{len(resume_data.technical_skills)} M:{len(resume_data.management_skills)}")
                     log.debug(f"Extracted experience years: {resume_data.total_years_experience}")
                     return resume_data
                else:
                     log.error(f"ASYNC: LLM response for resume extraction was not a dictionary: {type(extracted_json)}")
                     return None
            except Exception as e:
                log.error(f"ASYNC: Failed to validate extracted resume data: {e}", exc_info=True)
                log.error(f"ASYNC: Invalid JSON received for resume: {extracted_json}")
                return None
        else:
            log.error("ASYNC: Failed to get valid JSON response from LLM for resume extraction.")
            return None

    # --- ASYNC Suitability Analysis ---
    async def analyze_suitability_async(self, resume_data: ResumeData, job_data: Dict[str, Any]) -> Optional[JobAnalysisResult]:
        """
        ASYNC: Analyzes job suitability against resume data using the LLM.
        Expects LLM to return nested JSON, extracts and validates 'analysis' part.
        Returns only the validated JobAnalysisResult object or None.
        """
        if not resume_data: log.warning("Missing structured resume data for suitability analysis."); return None
        # Check description BEFORE formatting prompt
        if not job_data or not job_data.get("description"):
             job_title_for_log = job_data.get('title', 'N/A') if job_data else 'N/A'
             log.warning(f"Missing job data or description for job: {job_title_for_log}. Skipping analysis.")
             return None

        try:
            resume_data_json = resume_data.model_dump_json(indent=2)
            job_data_json = json.dumps(job_data, indent=2, default=str)
            prompt = self.suitability_prompt_template.format(
                resume_data_json=resume_data_json,
                job_data_json=job_data_json )
        except Exception as e:
            log.error(f"Error preparing data for suitability prompt: {e}", exc_info=True)
            return None

        job_title = job_data.get('title', 'N/A')
        log.info(f"ASYNC: Requesting suitability analysis from LLM for job: {job_title}")
        combined_json_response = await self._call_ollama_async(prompt) # Await async call

        if not combined_json_response or not isinstance(combined_json_response, dict):
            log.error(f"ASYNC: Failed to get valid JSON dictionary response from LLM for suitability analysis for job: {job_title}.")
            log.error(f"Raw response received from Ollama: {combined_json_response}") # Log raw response on failure
            return None

        analysis_data = combined_json_response.get("analysis")
        if not analysis_data or not isinstance(analysis_data, dict):
            log.error(f"ASYNC: LLM response JSON did not contain a valid 'analysis' dictionary for job: {job_title}.")
            log.debug(f"Full LLM response received: {combined_json_response}")
            return None

        try:
            analysis_result = JobAnalysisResult(**analysis_data)
            log.info(f"ASYNC: Suitability score for '{job_title}': {analysis_result.suitability_score}%")
            return analysis_result
        except Exception as e:
            log.error(f"ASYNC: Failed to validate LLM analysis result for job: {job_title}. Error: {e}", exc_info=True)
            log.error(f"ASYNC: Invalid 'analysis' JSON structure received: {analysis_data}")
            return None