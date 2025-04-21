# analysis/analyzer.py
import ollama
import json
import logging
from typing import Dict, Any, Optional, List
from models import JobAnalysisResult, KeywordAnalysis, MatchDetails # Import updated Pydantic models
from datetime import datetime

logger = logging.getLogger(__name__)

class JobAnalyzer:
    def __init__(self, model_name: str, ollama_base_url: Optional[str] = None,
                 resume_prompt_path: Optional[str] = None,
                 suitability_prompt_path: Optional[str] = None):
        self.model_name = model_name
        self.client = ollama.Client(host=ollama_base_url) if ollama_base_url else ollama.Client()
        self.resume_prompt_template = self._load_prompt(resume_prompt_path)
        self.suitability_prompt_template = self._load_prompt(suitability_prompt_path)
        logger.info(f"JobAnalyzer initialized with model: {model_name}")

    def _load_prompt(self, file_path: Optional[str]) -> Optional[str]:
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"Prompt file not found or not specified: {file_path}")
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading prompt file {file_path}: {e}", exc_info=True)
            return None

    def check_connection(self) -> bool:
        """Checks basic connection to the Ollama server."""
        try:
            self.client.list() # Simple command to test connection
            logger.info("Ollama connection successful (basic check passed).")
            # You might want a more robust check, like ensuring the specific model exists
            # models = self.client.list().get('models', [])
            # if not any(m['name'] == self.model_name for m in models):
            #     logger.error(f"Model '{self.model_name}' not found locally in Ollama.")
            #     return False
            return True
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}", exc_info=True)
            return False

    def extract_resume_data(self, resume_text: str) -> Optional[Dict[str, Any]]:
        """Extracts structured data from resume text using LLM."""
        if not self.resume_prompt_template:
            logger.error("Resume extraction prompt template is missing.")
            return None

        prompt = self.resume_prompt_template.format(resume_text=resume_text)
        logger.info("Requesting resume data extraction from LLM...")
        logger.debug(f"Sending request to Ollama model {self.model_name}. Prompt length: {len(prompt)} chars.")

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format='json' # Request JSON output directly if model supports it
            )
            content = response['message']['content']
            logger.debug(f"Ollama raw response (first 500 chars): {content[:500]} ...")

            # Attempt to parse the JSON content
            extracted_data = json.loads(content)
            logger.debug("Successfully parsed JSON response from Ollama.")
            logger.info("Successfully parsed extracted resume data.")
            return extracted_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from Ollama for resume extraction: {e}")
            logger.error(f"LLM response content was: {content}") # Log the faulty content
            return None
        except Exception as e:
            logger.error(f"Error during resume data extraction: {e}", exc_info=True)
            return None

    def analyze_suitability(self,
                             structured_resume: Dict[str, Any],
                             job_details: Dict[str, Any],
                             user_profile: Dict[str, Any]) -> Optional[JobAnalysisResult]:
        """Analyzes job suitability using LLM based on structured resume and job details."""
        if not self.suitability_prompt_template:
            logger.error("Job suitability prompt template is missing.")
            return None

        # Prepare input JSON strings for the prompt
        try:
            structured_resume_json = json.dumps(structured_resume, indent=2)
            job_details_json = json.dumps(job_details, indent=2, default=str) # Use default=str for dates etc.
        except Exception as json_err:
            logger.error(f"Failed to serialize input data to JSON for LLM prompt: {json_err}", exc_info=True)
            return None

        # Format the prompt with all necessary data
        prompt = self.suitability_prompt_template.format(
            structured_resume_json=structured_resume_json,
            job_details_json=job_details_json,
            # --- Pass user profile data ---
            user_salary_min=user_profile.get('DESIRED_SALARY_MIN', 'N/A'),
            user_salary_max=user_profile.get('DESIRED_SALARY_MAX', 'N/A'),
            must_have_skills=", ".join(user_profile.get('MUST_HAVE_SKILLS', [])),
            # --- Pass job salary data if available ---
            job_salary_min=job_details.get('min_amount', 'N/A'),
            job_salary_max=job_details.get('max_amount', 'N/A')
        )

        job_title = job_details.get('title', 'Unknown Job')
        logger.info(f"Requesting suitability analysis from LLM for job: {job_title}")
        logger.debug(f"Sending request to Ollama model {self.model_name}. Prompt length: {len(prompt)} chars.")

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            content = response['message']['content']
            logger.debug(f"Ollama raw response (first 500 chars): {content[:500]} ...")

            # Attempt to parse the JSON content
            analysis_data = json.loads(content)
            logger.debug("Successfully parsed JSON response from Ollama.")

            # --- Parse the *enhanced* structure ---
            keyword_analysis_data = analysis_data.get('keyword_analysis', {})
            skill_match_data = analysis_data.get('skill_match_details', {})
            exp_match_data = analysis_data.get('experience_match_details', {})
            qual_match_data = analysis_data.get('qualification_match_details', {})

            analysis_result = JobAnalysisResult(
                suitability_score=analysis_data.get('suitability_score'),
                justification=analysis_data.get('justification'),
                keyword_analysis=KeywordAnalysis(
                    matched_required=keyword_analysis_data.get('matched_required', []),
                    missing_required=keyword_analysis_data.get('missing_required', []),
                    missing_preferred=keyword_analysis_data.get('missing_preferred', [])
                ),
                 skill_match_details=MatchDetails(
                      assessment=skill_match_data.get('assessment'),
                      reasoning=skill_match_data.get('reasoning')
                 ),
                 experience_match_details=MatchDetails(
                      assessment=exp_match_data.get('assessment'),
                      reasoning=exp_match_data.get('reasoning')
                 ),
                 qualification_match_details=MatchDetails(
                      assessment=qual_match_data.get('assessment'),
                      reasoning=qual_match_data.get('reasoning')
                 ),
                salary_alignment=analysis_data.get('salary_alignment'),
                alignment_details=analysis_data.get('alignment_details'),
                date_analyzed=datetime.now() # Redundant due to default_factory, but explicit
            )
            return analysis_result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from Ollama for job analysis '{job_title}': {e}")
            logger.error(f"LLM response content was: {content}")
            return None # Return None on failure
        except Exception as e:
            logger.error(f"Error during suitability analysis for job '{job_title}': {e}", exc_info=True)
            return None