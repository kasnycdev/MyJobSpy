# main_matcher.py
import logging
from typing import List, Dict, Any, Optional
from analysis.analyzer import JobAnalyzer # Assuming analyzer class path
# --- MODIFIED IMPORT ---
from models.models import CombinedJobResult, JobAnalysisResult, OriginalJobData # Import models
# --- END MODIFIED IMPORT ---
import json
import os
from tqdm import tqdm # For progress bar
import fitz # PyMuPDF

logger = logging.getLogger(__name__)

def load_and_extract_resume(resume_filepath: Optional[str], analyzer: JobAnalyzer) -> Optional[Dict[str, Any]]:
    """Loads resume text and calls LLM for structured data extraction."""
    if not resume_filepath:
        logger.error("Resume filepath not provided.")
        return None

    logger.info(f"Processing resume file: {resume_filepath}")
    if not os.path.exists(resume_filepath):
        logger.error(f"Resume file not found: {resume_filepath}")
        return None

    resume_text = ""
    try:
        if resume_filepath.lower().endswith('.txt'):
             with open(resume_filepath, 'r', encoding='utf-8') as f:
                  resume_text = f.read()
             logger.info(f"Successfully read TXT resume ({len(resume_text)} chars).")
        elif resume_filepath.lower().endswith('.pdf'):
             logger.info(f"Parsing PDF resume: {os.path.basename(resume_filepath)}")
             doc = fitz.open(resume_filepath)
             for page in doc:
                 resume_text += page.get_text()
             doc.close()
             if not resume_text:
                  logger.warning(f"PyMuPDF parsed 0 characters from {resume_filepath}. Check PDF content/format.")
             else:
                  logger.info(f"Successfully parsed PDF text ({len(resume_text)} chars).")
        else:
            logger.error(f"Unsupported resume file format: {resume_filepath}. Only .txt and .pdf supported currently.")
            return None

        if not resume_text:
             logger.error("Resume text could not be extracted or is empty.")
             return None

        # Call the analyzer to extract structured data
        structured_data = analyzer.extract_resume_data(resume_text)
        if structured_data:
             logger.info("Successfully extracted structured data from resume.")
             # Optional save
             # try:
             #      os.makedirs("./output", exist_ok=True)
             #      with open("./output/structured_resume.json", "w", encoding='utf-8') as f:
             #           json.dump(structured_data, f, indent=4)
             # except Exception as save_err:
             #      logger.warning(f"Could not save structured resume JSON: {save_err}")
        else:
             logger.error("Failed to get structured data from resume via LLM.")

        return structured_data

    except ImportError:
        logger.error("PyMuPDF (fitz) not installed. Cannot parse PDF. Install with 'pip install PyMuPDF'")
        return None
    except Exception as e:
        logger.error(f"Error loading/processing resume {resume_filepath}: {e}", exc_info=True)
        return None


def analyze_jobs(analyzer: JobAnalyzer,
                 structured_resume: Dict[str, Any],
                 jobs_list: List[Dict[str, Any]],
                 user_profile: Dict[str, Any]
                 ) -> List[CombinedJobResult]:
    """
    Analyzes a list of job dictionaries against the structured resume.
    """
    results = []
    if not jobs_list:
        logger.warning("No jobs provided for analysis.")
        return results
    if not structured_resume:
        logger.error("Structured resume data is missing, cannot perform analysis.")
        # Create entries with original data but no analysis
        for job_data_dict in jobs_list:
             try:
                  original_data = OriginalJobData(**job_data_dict)
                  results.append(CombinedJobResult(original_job_data=original_data, analysis=None))
             except Exception as e:
                  logger.error(f"Could not create entry for job ID {job_data_dict.get('id', 'N/A')} even without analysis: {e}")
        return results


    logger.info(f"Starting analysis of {len(jobs_list)} jobs...")

    for job_data_dict in tqdm(jobs_list, desc="Analyzing jobs"):
        job_title = job_data_dict.get('title', 'Unknown Job')
        job_id = job_data_dict.get('id', 'N/A')
        analysis_result: Optional[JobAnalysisResult] = None # Default to None
        original_data: Optional[OriginalJobData] = None

        try:
            # 1. Validate/Create OriginalJobData model
            original_data = OriginalJobData(**job_data_dict)

            # 2. Call the analyzer's suitability function
            analysis_result = analyzer.analyze_suitability(
                structured_resume,
                original_data.model_dump(mode='json'), # Pass validated dict
                user_profile
            )

            if not analysis_result:
                 logger.warning(f"Analysis returned None for job: {job_title} (ID: {job_id})")
                 # Keep analysis_result as None

        except Exception as e:
            logger.error(f"Error during analysis pipeline for job '{job_title}' (ID: {job_id}): {e}", exc_info=True)
            # Attempt to create a result with error state if possible
            if original_data is None: # If error happened during OriginalJobData parsing
                 try: original_data = OriginalJobData(**job_data_dict) # Try again or use partial
                 except: original_data = OriginalJobData(id=job_id, title=f"Error Processing: {job_title}") # Fallback
            # Create an analysis object indicating failure
            analysis_result = JobAnalysisResult(suitability_score=0, justification=f"Error during analysis: {str(e)[:200]}")

        # 3. Combine into the final result object (analysis might be None)
        if original_data: # Ensure we have at least original data to append
             combined_result = CombinedJobResult(
                 original_job_data=original_data,
                 analysis=analysis_result
             )
             results.append(combined_result)
        else:
             logger.error(f"Failed to process original data for job ID {job_id}, skipping.")


    logger.info(f"Finished analyzing jobs. {len(results)} results processed.")
    return results