# main_matcher.py
import logging
from typing import List, Dict, Any, Optional
from analysis.analyzer import JobAnalyzer # Assuming analyzer class path
from models import CombinedJobResult, JobAnalysisResult, OriginalJobData # Import models
import json
import os
from tqdm import tqdm # For progress bar

logger = logging.getLogger(__name__)

def load_and_extract_resume(resume_filepath: str, analyzer: JobAnalyzer) -> Optional[Dict[str, Any]]:
    """Loads resume text and calls LLM for structured data extraction."""
    logger.info(f"Processing resume file: {resume_filepath}")
    if not os.path.exists(resume_filepath):
        logger.error(f"Resume file not found: {resume_filepath}")
        return None

    try:
        # Basic text extraction (replace with PDF/DOCX parser if needed)
        if resume_filepath.lower().endswith('.txt'):
             with open(resume_filepath, 'r', encoding='utf-8') as f:
                  resume_text = f.read()
        elif resume_filepath.lower().endswith('.pdf'):
             # Add robust PDF parsing logic here (e.g., using PyPDF2, pdfminer.six)
             logger.info(f"Parsing PDF resume: {os.path.basename(resume_filepath)}")
             # Placeholder - replace with actual PDF parsing
             try:
                  import fitz # PyMuPDF
                  doc = fitz.open(resume_filepath)
                  resume_text = ""
                  for page in doc:
                       resume_text += page.get_text()
                  doc.close()
                  logger.info(f"Successfully parsed PDF text ({len(resume_text)} chars).")
             except ImportError:
                  logger.error("PyMuPDF not installed. Cannot parse PDF. Install with 'pip install PyMuPDF'")
                  return None
             except Exception as pdf_err:
                   logger.error(f"Error parsing PDF {resume_filepath}: {pdf_err}", exc_info=True)
                   return None
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
             # Optionally save the structured resume to a file for inspection/reuse
             # with open("./output/structured_resume.json", "w") as f:
             #     json.dump(structured_data, f, indent=4)
        else:
             logger.error("Failed to get structured data from resume via LLM.")

        return structured_data

    except Exception as e:
        logger.error(f"Error loading/processing resume {resume_filepath}: {e}", exc_info=True)
        return None


def analyze_jobs(analyzer: JobAnalyzer,
                 structured_resume: Dict[str, Any],
                 jobs_list: List[Dict[str, Any]],
                 user_profile: Dict[str, Any] # Pass user profile
                 ) -> List[CombinedJobResult]:
    """
    Analyzes a list of job dictionaries against the structured resume.

    Args:
        analyzer: The JobAnalyzer instance.
        structured_resume: The dictionary representing the parsed resume.
        jobs_list: A list of dictionaries, each representing a job posting (processed).
        user_profile: Dictionary containing user preferences (salary, skills).

    Returns:
        A list of CombinedJobResult objects containing original data and analysis.
    """
    results = []
    if not jobs_list:
        logger.warning("No jobs provided for analysis.")
        return results

    logger.info(f"Starting analysis of {len(jobs_list)} jobs...")

    # Wrap the loop with tqdm for a progress bar
    for job_data_dict in tqdm(jobs_list, desc="Analyzing jobs"):
        try:
            # Validate and create OriginalJobData model instance
            # This helps catch type errors early
            original_data = OriginalJobData(**job_data_dict)

            # Call the analyzer's suitability function
            analysis_result: Optional[JobAnalysisResult] = analyzer.analyze_suitability(
                structured_resume,
                original_data.model_dump(), # Pass the validated dict
                user_profile # Pass user profile to analysis
            )

            if analysis_result:
                 logger.debug(f"Analysis successful for job: {original_data.title}")
            else:
                 # Handle analysis failure for this specific job
                 logger.warning(f"Analysis failed for job: {original_data.title} (ID: {original_data.id})")
                 # Create a CombinedJobResult with None for analysis
                 analysis_result = None # Ensure it's None

            combined_result = CombinedJobResult(
                original_job_data=original_data,
                analysis=analysis_result # This will be None if analysis failed
            )
            results.append(combined_result)

        except Exception as e:
            job_title = job_data_dict.get('title', 'Unknown Job')
            job_id = job_data_dict.get('id', 'N/A')
            logger.error(f"Critical error processing job '{job_title}' (ID: {job_id}): {e}", exc_info=True)
            # Optionally create a result entry indicating the error
            try:
                 error_original_data = OriginalJobData(**job_data_dict) # Try to parse original data at least
                 error_analysis = JobAnalysisResult(suitability_score=0, justification=f"Error during processing: {e}")
                 results.append(CombinedJobResult(original_job_data=error_original_data, analysis=error_analysis))
            except Exception as inner_e:
                 logger.error(f"Could not even create error entry for job ID {job_id}: {inner_e}")

    logger.info(f"Finished analyzing jobs. {len(results)} results processed.")
    return results

# Note: The main execution logic calling these functions is now in run_pipeline.py