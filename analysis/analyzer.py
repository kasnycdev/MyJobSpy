import ollama
import json
import logging
import time
import os
from typing import Dict, Optional, Any

from .models import ResumeData, JobAnalysisResult
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
        raise # Re-raise the error as this is critical
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

    def _check_connection_and_model(self):
        """Checks Ollama connection and ensures the configured model is available."""
        try:
            log.info(f"Checking Ollama connection at {config.OLLAMA_BASE_URL}...")
            # First, just test basic connection without assuming list structure
            self.client.ps() # Use 'ps' or another simple command to check connection status
            log.info("Ollama connection successful (basic check passed).")

            # Now, get the list of models and inspect carefully
            log.info("Fetching list of local Ollama models...")
            ollama_list_response = self.client.list()
            log.debug(f"Raw Ollama list response: {ollama_list_response}") # Log the full response for debugging

            # Safely extract the list of model dictionaries
            models_data = ollama_list_response.get('models', []) # Default to empty list if 'models' key missing
            if not isinstance(models_data, list):
                log.error(f"Ollama list response 'models' key did not contain a list. Found type: {type(models_data)}")
                models_data = [] # Treat invalid data as empty list

            # Safely extract model names using .get() and filter out None/empty results
            local_models = []
            for m in models_data:
                if isinstance(m, dict):
                    model_name = m.get('name') # Use .get() for safe access
                    if model_name: # Ensure name is not None or empty
                        local_models.append(model_name)
                else:
                    log.warning(f"Found non-dictionary item in Ollama models list: {m}")

            log.info(f"Successfully parsed local models: {local_models}")

            # Check if the desired model exists in the extracted list
            if config.OLLAMA_MODEL not in local_models:
                log.warning(f"Model '{config.OLLAMA_MODEL}' not found in parsed local models list: {local_models}")
                log.info(f"Attempting to pull model '{config.OLLAMA_MODEL}'. This may take time...")
                try:
                    self._pull_model_with_progress(config.OLLAMA_MODEL)
                    # Verify it exists after pulling
                    updated_list = self.client.list().get('models', [])
                    updated_names = [m.get('name') for m in updated_list if isinstance(m, dict) and m.get('name')]
                    if config.OLLAMA_MODEL not in updated_names:
                         log.error(f"Model '{config.OLLAMA_MODEL}' still not found after attempting pull.")
                         raise ConnectionError(f"Ollama model pull seemed complete but model '{config.OLLAMA_MODEL}' not listed.")

                except Exception as pull_err:
                    log.error(f"Failed to pull Ollama model '{config.OLLAMA_MODEL}': {pull_err}", exc_info=True)
                    # Raise a specific error indicating the model is unavailable
                    raise ConnectionError(f"Required Ollama model '{config.OLLAMA_MODEL}' unavailable and pull failed.") from pull_err
            else:
                log.info(f"Using configured Ollama model: {config.OLLAMA_MODEL}")

        except (ollama.ResponseError, ConnectionError, TimeoutError) as conn_e:
             log.error(f"Failed to connect or communicate with Ollama at {config.OLLAMA_BASE_URL}. Is Ollama running? Error: {conn_e}", exc_info=True)
             raise ConnectionError(f"Ollama connection/setup failed: {conn_e}") from conn_e
        except Exception as e:
            log.error(f"An unexpected error occurred during Ollama connection/setup: {e}", exc_info=True)
            # Wrap unexpected errors for clarity
            raise ConnectionError(f"Ollama connection/setup failed unexpectedly: {e}") from e
    def _pull_model_with_progress(self, model_name: str):
        """Pulls an Ollama model, showing progress."""
        current_digest = ""
        status = ""
        try:
            for progress in self.client.pull(model_name, stream=True):
                digest = progress.get("digest", "")
                if digest != current_digest and current_digest != "":
                    print() # Newline after completion message
                if digest:
                    current_digest = digest
                    status = progress.get('status', '')
                    print(f"Pulling {model_name}: {status}", end='\r')
                else:
                    status = progress.get('status', '')
                    print(f"Pulling {model_name}: {status}") # Print other messages like downloading, verifying

                if progress.get('error'):
                    raise Exception(f"Pull error: {progress['error']}")

                # Check completion status explicitly
                if 'status' in progress and 'success' in progress['status'].lower():
                    # Sometimes success message doesn't have digest, handle here
                    print() # Ensure newline after final status
                    log.info(f"Successfully pulled model {model_name}")
                    break
        except Exception as e:
             print() # Ensure newline after potential error
             log.error(f"Error during model pull: {e}")
             raise
        finally:
             print() # Final newline ensures clean state


    def _call_ollama(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Calls the Ollama API with retry logic and expects JSON output.
        Handles potential JSON parsing errors and retries.
        """
        log.debug(f"Sending request to Ollama model {config.OLLAMA_MODEL}. Prompt length: {len(prompt)} chars.")
        if len(prompt) > config.MAX_PROMPT_CHARS:
             log.warning(f"Prompt length ({len(prompt)} chars) exceeds threshold ({config.MAX_PROMPT_CHARS}). May risk context window issues.")
             # Consider truncating prompt here if necessary

        last_exception = None
        for attempt in range(config.OLLAMA_MAX_RETRIES):
            try:
                response = self.client.chat(
                    model=config.OLLAMA_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    format='json', # Request JSON output format
                    options={'temperature': 0.1} # Lower temperature for more deterministic JSON
                )
                content = response['message']['content']
                log.debug(f"Ollama raw response (first 500 chars): {content[:500]}...")

                try:
                    # Handle potential markdown code blocks around JSON
                    content_strip = content.strip()
                    if content_strip.startswith("```json"):
                        content_strip = content_strip[7:]
                        if content_strip.endswith("```"):
                            content_strip = content_strip[:-3]
                    elif content_strip.startswith("```"):
                         content_strip = content_strip[3:]
                         if content_strip.endswith("```"):
                              content_strip = content_strip[:-3]

                    result = json.loads(content_strip.strip())
                    log.debug("Successfully parsed JSON response from Ollama.")
                    return result
                except json.JSONDecodeError as json_err:
                    log.warning(f"Failed to decode JSON response from Ollama (Attempt {attempt + 1}): {json_err}")
                    log.debug(f"Problematic Ollama response content: {content}")
                    last_exception = json_err
                    # Retry on JSON errors as LLM might fix formatting

            except (ollama.ResponseError, TimeoutError, ConnectionError) as conn_err:
                log.warning(f"Ollama API communication error (Attempt {attempt + 1}): {conn_err}")
                last_exception = conn_err
            except Exception as e:
                log.error(f"Unexpected error calling Ollama API (Attempt {attempt + 1}): {e}", exc_info=True)
                last_exception = e
                # Potentially stop retrying on unexpected errors? For now, continue.

            if attempt < config.OLLAMA_MAX_RETRIES - 1:
                delay = config.OLLAMA_RETRY_DELAY * (2 ** attempt)
                log.info(f"Retrying Ollama call in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                 log.error(f"Ollama call failed after {config.OLLAMA_MAX_RETRIES} attempts.")
                 if last_exception:
                     log.error(f"Last error encountered: {last_exception}")

        return None # Failed after all retries

    def extract_resume_data(self, resume_text: str) -> Optional[ResumeData]:
        """Extracts structured data from resume text using the LLM."""
        if not resume_text or not resume_text.strip():
            log.warning("Resume text is empty, cannot extract data.")
            return None

        prompt = self.resume_prompt_template.format(resume_text=resume_text)
        log.info("Requesting resume data extraction from LLM...")
        extracted_json = self._call_ollama(prompt)

        if extracted_json:
            try:
                resume_data = ResumeData(**extracted_json)
                log.info("Successfully parsed extracted resume data.")
                log.debug(f"Extracted skills: {resume_data.skills}")
                log.debug(f"Extracted experience years: {resume_data.total_years_experience}")
                return resume_data
            except Exception as e: # Catches Pydantic validation errors
                log.error(f"Failed to validate extracted resume data: {e}", exc_info=True)
                log.error(f"Invalid JSON received for resume: {extracted_json}")
                return None
        else:
            log.error("Failed to get valid JSON response from LLM for resume extraction.")
            return None

    def analyze_suitability(self, resume_data: ResumeData, job_data: Dict[str, Any]) -> Optional[JobAnalysisResult]:
        """Analyzes job suitability against resume data using the LLM."""
        if not resume_data:
             log.warning("Missing structured resume data for suitability analysis.")
             return None
        if not job_data or not job_data.get("description"): # Need at least a description
             log.warning(f"Missing job data or description for job: {job_data.get('title', 'N/A')}. Skipping analysis.")
             return None

        try:
            resume_data_json = resume_data.model_dump_json(indent=2)
            # Create a focused job context for the prompt
            job_context = {
                "title": job_data.get("title"),
                "company": job_data.get("company"),
                "location": job_data.get("location"),
                "description": job_data.get("description"),
                "required_skills": job_data.get("skills"), # Assuming jobspy provides 'skills'
                "qualifications": job_data.get("qualifications"), # May not always be present
                "salary_text": job_data.get("salary"), # Assuming jobspy provides 'salary' field
                "job_type": job_data.get("job_type") or job_data.get("employment_type"),
                "work_model": job_data.get("work_model"), # May need inference
                "benefits_text": job_data.get("benefits") # Assuming jobspy provides 'benefits'
            }
            # Remove keys with None values to keep prompt cleaner
            job_context_filtered = {k: v for k, v in job_context.items() if v is not None and v != ''}
            job_data_json = json.dumps(job_context_filtered, indent=2)

            prompt = self.suitability_prompt_template.format(
                resume_data_json=resume_data_json,
                job_data_json=job_data_json
            )
        except Exception as e:
            log.error(f"Error preparing data for suitability prompt: {e}", exc_info=True)
            return None

        log.info(f"Requesting suitability analysis from LLM for job: {job_data.get('title', 'N/A')}")
        analysis_json = self._call_ollama(prompt)

        if analysis_json:
            try:
                analysis_result = JobAnalysisResult(**analysis_json)
                log.info(f"Suitability score for '{job_data.get('title', 'N/A')}': {analysis_result.suitability_score}%")
                return analysis_result
            except Exception as e: # Catches Pydantic validation errors
                log.error(f"Failed to validate LLM analysis result: {e}", exc_info=True)
                log.error(f"Invalid JSON received for analysis: {analysis_json}")
                return None
        else:
            log.error("Failed to get valid JSON response from LLM for suitability analysis.")
            return None