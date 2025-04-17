import ollama
import json
import logging
from typing import Dict, Optional
import time

from .models import ResumeData, JobAnalysisResult
from .prompts import RESUME_EXTRACTION_PROMPT_TEMPLATE, SUITABILITY_ANALYSIS_PROMPT_TEMPLATE
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_REQUEST_TIMEOUT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResumeAnalyzer:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL, timeout=OLLAMA_REQUEST_TIMEOUT)
        self._check_connection()

    def _check_connection(self):
        """Checks if the Ollama server is reachable and the model exists."""
        try:
            logging.info(f"Connecting to Ollama at {OLLAMA_BASE_URL}...")
            self.client.list() # Simple command to check connection
            logging.info("Ollama connection successful.")
            # Check if model exists locally
            local_models = [m['name'] for m in self.client.list()['models']]
            if OLLAMA_MODEL not in local_models:
                logging.warning(f"Model '{OLLAMA_MODEL}' not found locally.")
                logging.info(f"Attempting to pull model '{OLLAMA_MODEL}'. This may take a while...")
                try:
                    self._pull_model(OLLAMA_MODEL)
                except Exception as pull_err:
                     logging.error(f"Failed to pull model '{OLLAMA_MODEL}'. Please ensure Ollama is running and the model name is correct. Error: {pull_err}")
                     raise ConnectionError(f"Ollama model '{OLLAMA_MODEL}' not available and pull failed.") from pull_err
            else:
                 logging.info(f"Using Ollama model: {OLLAMA_MODEL}")

        except Exception as e:
            logging.error(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}. Ensure Ollama is running. Error: {e}")
            raise ConnectionError(f"Ollama connection failed: {e}") from e

    def _pull_model(self, model_name: str):
        """Helper to pull the model with progress"""
        current_digest = ""
        for progress in self.client.pull(model_name, stream=True):
            digest = progress.get("digest", "")
            if digest != current_digest and current_digest != "":
                print() # Newline afer completion message
            if digest:
                 current_digest = digest
                 print(f"Pulling {model_name}: {progress.get('status', '')}", end='\r')
            else:
                 print(f"Pulling {model_name}: {progress.get('status', '')}") # Print other messages

            if progress.get('error'):
                 raise Exception(f"Pull error: {progress['error']}")

            # Handle final status message potentially not having digest
            if progress.get('status') == 'success' and not digest:
                print(f"\nPull complete for {model_name}")
                break
        print() # Ensure newline after final status


    def _call_ollama(self, prompt: str, retries: int = 2, delay: int = 5) -> Optional[Dict]:
        """Calls the Ollama API with retry logic and expects JSON output."""
        for attempt in range(retries + 1):
            try:
                logging.debug(f"Sending request to Ollama (Attempt {attempt + 1}/{retries + 1}). Prompt length: {len(prompt)}")
                response = self.client.chat(
                    model=OLLAMA_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    format='json', # Request JSON output format
                    options={'temperature': 0.2} # Lower temperature for more deterministic JSON
                )
                content = response['message']['content']
                logging.debug(f"Ollama raw response: {content[:500]}...") # Log start of response

                # Try to parse the JSON response
                try:
                    # Sometimes the response might have ```json ... ``` markers
                    if content.strip().startswith("```json"):
                        content = content.strip()[7:-3].strip()
                    elif content.strip().startswith("```"):
                         content = content.strip()[3:-3].strip()

                    result = json.loads(content)
                    logging.debug("Successfully parsed JSON response from Ollama.")
                    return result
                except json.JSONDecodeError as json_err:
                    logging.error(f"Failed to decode JSON response from Ollama: {json_err}")
                    logging.error(f"Problematic response content: {content}")
                    # Don't retry on JSON decode error unless it's the last attempt? Or maybe always retry?
                    # Let's retry, the LLM might fix it on the next try.
                    if attempt == retries:
                        return None # Failed after all retries

            except Exception as e:
                logging.error(f"Error calling Ollama API (Attempt {attempt + 1}): {e}")
                if attempt == retries:
                    return None # Failed after all retries

            if attempt < retries:
                logging.warning(f"Retrying Ollama call in {delay} seconds...")
                time.sleep(delay)

        return None # Should not be reached if retries > 0, but safety return

    def extract_resume_data(self, resume_text: str) -> Optional[ResumeData]:
        """Extracts structured data from resume text using the LLM."""
        if not resume_text:
            logging.warning("Resume text is empty, cannot extract data.")
            return None

        prompt = RESUME_EXTRACTION_PROMPT_TEMPLATE.format(resume_text=resume_text)
        logging.info("Requesting resume data extraction from LLM...")
        extracted_json = self._call_ollama(prompt)

        if extracted_json:
            try:
                resume_data = ResumeData(**extracted_json)
                logging.info("Successfully parsed extracted resume data.")
                return resume_data
            except Exception as e: # Catches Pydantic validation errors
                logging.error(f"Failed to validate extracted resume data against Pydantic model: {e}")
                logging.error(f"Extracted JSON: {extracted_json}")
                return None
        else:
            logging.error("Failed to get valid JSON response from LLM for resume extraction.")
            return None

    def analyze_suitability(self, resume_data: ResumeData, job_data: dict) -> Optional[JobAnalysisResult]:
        """Analyzes job suitability against resume data using the LLM."""
        if not resume_data or not job_data:
            logging.warning("Missing resume data or job data for suitability analysis.")
            return None

        try:
            # Prepare data for the prompt
            resume_data_json = resume_data.model_dump_json(indent=2)
            # Try to pass only essential fields to the LLM to save context, but fall back to full dict
            essential_job_keys = ['title', 'company', 'location', 'description', 'required_skills', 'qualifications', 'salary_text', 'benefits_text', 'job_type', 'work_model']
            job_context = {k: job_data.get(k) for k in essential_job_keys if k in job_data}
            if not job_context.get('description'): # Ensure description is included if possible
                job_context['full_data_fallback'] = job_data # Maybe include full data if description missing

            job_data_json = json.dumps(job_context, indent=2)

            prompt = SUITABILITY_ANALYSIS_PROMPT_TEMPLATE.format(
                resume_data_json=resume_data_json,
                job_data_json=job_data_json
            )
        except Exception as e:
            logging.error(f"Error preparing data for suitability prompt: {e}")
            return None

        logging.info(f"Requesting suitability analysis from LLM for job: {job_data.get('title', 'N/A')}")
        analysis_json = self._call_ollama(prompt)

        if analysis_json:
            try:
                analysis_result = JobAnalysisResult(**analysis_json)
                logging.info(f"Suitability score for '{job_data.get('title', 'N/A')}': {analysis_result.suitability_score}%")
                return analysis_result
            except Exception as e: # Catches Pydantic validation errors
                logging.error(f"Failed to validate LLM analysis result against Pydantic model: {e}")
                logging.error(f"Analysis JSON received: {analysis_json}")
                return None
        else:
            logging.error("Failed to get valid JSON response from LLM for suitability analysis.")
            return None