import ollama
import json
import logging
import time
import os
from typing import Dict, Optional, Any

# --- CORRECTED IMPORT ---
# Import Pydantic models from the models.py file within the analysis package
from analysis.models import ResumeData, JobAnalysisResult, AnalyzedJob
# --- END CORRECTION ---

import config # Import our central configuration

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
    """Handles interaction with Ollama for resume and job analysis."""

    def __init__(self):
        self.client = ollama.Client(host=config.OLLAMA_BASE_URL, timeout=config.OLLAMA_REQUEST_TIMEOUT)
        self.resume_prompt_template = load_prompt(config.RESUME_PROMPT_FILE)
        self.suitability_prompt_template = load_prompt(config.SUITABILITY_PROMPT_FILE)
        self._check_connection_and_model()

    # --- _check_connection_and_model method remains unchanged ---
    def _check_connection_and_model(self):
        """Checks Ollama connection and ensures the configured model is available."""
        try:
            log.info(f"Checking Ollama connection at {config.OLLAMA_BASE_URL}...")
            self.client.ps()
            log.info("Ollama connection successful (basic check passed).")

            log.info("Fetching list of local Ollama models...")
            ollama_list_response = self.client.list()
            log.debug(f"Raw Ollama list() response type: {type(ollama_list_response)}")
            log.debug(f"Raw Ollama list() response content: {ollama_list_response}")

            models_data = ollama_list_response.get('models', [])
            if not isinstance(models_data, list):
                log.error(f"Ollama list response 'models' key did not contain a list. Found type: {type(models_data)}")
                models_data = []

            local_models = []
            for m in models_data:
                if hasattr(m, 'model') and isinstance(m.model, str) and m.model:
                    local_models.append(m.model)
                elif isinstance(m, dict) and m.get('name'):
                     log.warning(f"Found dictionary item in model list (unexpected): {m}. Using 'name' key.")
                     local_models.append(m.get('name'))
                else:
                    log.warning(f"Could not extract model name from item in Ollama models list: {m} (Type: {type(m)})")

            log.info(f"Successfully parsed local models: {local_models}")

            if config.OLLAMA_MODEL not in local_models:
                log.warning(f"Model '{config.OLLAMA_MODEL}' not found in parsed local models list: {local_models}")
                log.info(f"Attempting to pull model '{config.OLLAMA_MODEL}'. This may take time...")
                try:
                    self._pull_model_with_progress(config.OLLAMA_MODEL)

                    log.info("Re-fetching model list after pull...")
                    updated_list_response = self.client.list()
                    updated_models_data = updated_list_response.get('models', [])
                    updated_names = []
                    for m_upd in updated_models_data:
                         if hasattr(m_upd, 'model') and isinstance(m_upd.model, str) and m_upd.model:
                              updated_names.append(m_upd.model)
                         elif isinstance(m_upd, dict) and m_upd.get('name'):
                              updated_names.append(m_upd.get('name'))

                    log.debug(f"Model list after pull: {updated_names}")
                    if config.OLLAMA_MODEL not in updated_names:
                         log.error(f"Model '{config.OLLAMA_MODEL}' still not found after attempting pull and re-checking list.")
                         time.sleep(2)
                         final_list_response = self.client.list()
                         final_models_data = final_list_response.get('models', [])
                         final_names = [m_final.model for m_final in final_models_data if hasattr(m_final, 'model')]
                         if config.OLLAMA_MODEL not in final_names:
                             log.error(f"Final check failed. Model '{config.OLLAMA_MODEL}' unavailable.")
                             raise ConnectionError(f"Ollama model pull seemed complete but model '{config.OLLAMA_MODEL}' not listed after delay.")
                         else:
                              log.info("Model found after delay.")

                except Exception as pull_err:
                    log.error(f"Failed to pull or verify Ollama model '{config.OLLAMA_MODEL}': {pull_err}", exc_info=True)
                    raise ConnectionError(f"Required Ollama model '{config.OLLAMA_MODEL}' unavailable and pull failed.") from pull_err
            else:
                log.info(f"Using configured Ollama model: {config.OLLAMA_MODEL}")

        except (ollama.ResponseError, ConnectionError, TimeoutError) as conn_e:
             log.error(f"Failed to connect or communicate with Ollama at {config.OLLAMA_BASE_URL}. Is Ollama running? Error: {conn_e}", exc_info=True)
             raise ConnectionError(f"Ollama connection/setup failed: {conn_e}") from conn_e
        except Exception as e:
            log.error(f"An unexpected error occurred during Ollama connection/setup: {e}", exc_info=True)
            raise ConnectionError(f"Ollama connection/setup failed unexpectedly: {e}") from e

    # --- _pull_model_with_progress method remains unchanged ---
    def _pull_model_with_progress(self, model_name: str):
        """Pulls an Ollama model, showing progress."""
        current_digest = ""
        status = ""
        try:
            for progress in self.client.pull(model_name, stream=True):
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
        finally: print()

    # --- _call_ollama method remains unchanged ---
    def _call_ollama(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Calls Ollama API with retry logic, expects JSON output."""
        log.debug(f"Sending request to Ollama model {config.OLLAMA_MODEL}. Prompt length: {len(prompt)} chars.")
        if len(prompt) > config.MAX_PROMPT_CHARS:
             log.warning(f"Prompt length ({len(prompt)} chars) exceeds threshold ({config.MAX_PROMPT_CHARS}). May risk context window issues.")

        last_exception = None
        for attempt in range(config.OLLAMA_MAX_RETRIES):
            try:
                response = self.client.chat(
                    model=config.OLLAMA_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    format='json',
                    options={'temperature': 0.1}
                )
                content = response['message']['content']
                log.debug(f"Ollama raw response (first 500 chars): {content[:500]}...")
                try:
                    content_strip = content.strip()
                    if content_strip.startswith("```json"):
                        content_strip = content_strip[7:]; content_strip = content_strip[:-3] if content_strip.endswith("```") else content_strip
                    elif content_strip.startswith("```"):
                         content_strip = content_strip[3:]; content_strip = content_strip[:-3] if content_strip.endswith("```") else content_strip
                    result = json.loads(content_strip.strip())
                    log.debug("Successfully parsed JSON response from Ollama.")
                    return result
                except json.JSONDecodeError as json_err:
                    log.warning(f"Failed to decode JSON response (Attempt {attempt + 1}): {json_err}")
                    log.debug(f"Problematic Ollama response content: {content}")
                    last_exception = json_err
            except (ollama.ResponseError, TimeoutError, ConnectionError) as conn_err:
                log.warning(f"Ollama API communication error (Attempt {attempt + 1}): {conn_err}")
                last_exception = conn_err
            except Exception as e:
                log.error(f"Unexpected error calling Ollama API (Attempt {attempt + 1}): {e}", exc_info=True)
                last_exception = e

            if attempt < config.OLLAMA_MAX_RETRIES - 1:
                delay = config.OLLAMA_RETRY_DELAY * (2 ** attempt)
                log.info(f"Retrying Ollama call in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                 log.error(f"Ollama call failed after {config.OLLAMA_MAX_RETRIES} attempts.")
                 if last_exception: log.error(f"Last error encountered: {last_exception}")
        return None

    # --- extract_resume_data method remains unchanged ---
    def extract_resume_data(self, resume_text: str) -> Optional[ResumeData]:
        """Extracts structured data from resume text using the LLM."""
        MAX_RESUME_CHARS_FOR_LLM = 15000 # Added limit example
        if not resume_text or not resume_text.strip():
            log.warning("Resume text is empty, cannot extract data.")
            return None

        if len(resume_text) > MAX_RESUME_CHARS_FOR_LLM:
            log.warning(f"Resume text length ({len(resume_text)}) exceeds limit ({MAX_RESUME_CHARS_FOR_LLM}). Truncating.")
            resume_text_for_prompt = resume_text[:MAX_RESUME_CHARS_FOR_LLM]
        else:
            resume_text_for_prompt = resume_text

        prompt = self.resume_prompt_template.format(resume_text=resume_text_for_prompt)
        log.info("Requesting resume data extraction from LLM...")
        extracted_json = self._call_ollama(prompt)

        if extracted_json:
            try:
                if isinstance(extracted_json, dict):
                     resume_data = ResumeData(**extracted_json)
                     log.info("Successfully parsed extracted resume data.")
                     log.debug(f"Extracted skills: T:{len(resume_data.technical_skills)} M:{len(resume_data.management_skills)}")
                     log.debug(f"Extracted experience years: {resume_data.total_years_experience}")
                     return resume_data
                else:
                     log.error(f"LLM response for resume extraction was not a dictionary: {type(extracted_json)}")
                     return None
            except Exception as e:
                log.error(f"Failed to validate extracted resume data: {e}", exc_info=True)
                log.error(f"Invalid JSON received for resume: {extracted_json}")
                return None
        else:
            log.error("Failed to get valid JSON response from LLM for resume extraction.")
            return None

    # --- analyze_suitability method remains unchanged ---
    def analyze_suitability(self, resume_data: ResumeData, job_data: Dict[str, Any]) -> Optional[JobAnalysisResult]:
        """
        Analyzes job suitability against resume data using the LLM.
        Expects LLM to return a nested JSON with 'original_job_data' and 'analysis' keys.
        Returns only the validated JobAnalysisResult object or None.
        """
        if not resume_data:
             log.warning("Missing structured resume data for suitability analysis.")
             return None
        if not job_data or not job_data.get("description"):
             log.warning(f"Missing job data or description for job: {job_data.get('title', 'N/A')}. Skipping analysis.")
             return None

        try:
            resume_data_json = resume_data.model_dump_json(indent=2)
            job_data_json = json.dumps(job_data, indent=2, default=str)
            prompt = self.suitability_prompt_template.format(
                resume_data_json=resume_data_json,
                job_data_json=job_data_json
            )
        except Exception as e:
            log.error(f"Error preparing data for suitability prompt: {e}", exc_info=True)
            return None

        log.info(f"Requesting suitability analysis from LLM for job: {job_data.get('title', 'N/A')}")
        combined_json_response = self._call_ollama(prompt)

        if not combined_json_response or not isinstance(combined_json_response, dict):
            log.error("Failed to get valid JSON dictionary response from LLM for suitability analysis.")
            # --- ADDED LOGGING ---
            log.error(f"Raw response received from Ollama: {combined_json_response}")
            # --- END ADD ---
            return None

        analysis_data = combined_json_response.get("analysis")
        if not analysis_data or not isinstance(analysis_data, dict):
            log.error("LLM response JSON did not contain a valid 'analysis' dictionary.")
            log.debug(f"Full LLM response received: {combined_json_response}")
            return None

        try:
            analysis_result = JobAnalysisResult(**analysis_data)
            log.info(f"Suitability score for '{job_data.get('title', 'N/A')}': {analysis_result.suitability_score}%")
            return analysis_result
        except Exception as e:
            log.error(f"Failed to validate LLM analysis result: {e}", exc_info=True)
            log.error(f"Invalid 'analysis' JSON structure received: {analysis_data}")
            return None