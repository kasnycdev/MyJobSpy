import logging
import json
import os
import argparse
# --- CORRECTED IMPORT ---
from analysis.analyzer import ResumeAnalyzer # Use the correct class name
# --- END CORRECTION ---
from typing import List, Dict, Any, Optional, Tuple

from parsers.resume_parser import parse_resume
from parsers.job_parser import load_job_mandates # Kept for direct JSON input if needed
from analysis.models import ResumeData, AnalyzedJob, JobAnalysisResult # Assuming these are defined correctly
from filtering.filter import apply_filters
import config

# Setup logger
log = logging.getLogger(__name__)

# --- CORRECTED TYPE HINT ---
def load_and_extract_resume(resume_path: str, analyzer: ResumeAnalyzer) -> Optional[ResumeData]:
# --- END CORRECTION ---
    """Loads resume, parses text, and extracts structured data."""
    log.info(f"Processing resume file: {resume_path}")
    resume_text = parse_resume(resume_path)
    # Uncomment the next line ONLY for temporary debugging if parsing seems wrong
    # log.debug(f"--- Start Resume Text ---\n{resume_text}\n--- End Resume Text ---")
    if not resume_text:
        log.error("Failed to parse resume text.")
        return None
    structured_resume_data = analyzer.extract_resume_data(resume_text)
    if not structured_resume_data:
        log.error("Failed to extract structured data from resume.")
        return None
    log.info("Successfully extracted structured data from resume.")
    return structured_resume_data

# --- CORRECTED TYPE HINT ---
def analyze_jobs(
    analyzer: ResumeAnalyzer,
# --- END CORRECTION ---
    structured_resume_data: ResumeData,
    job_list: List[Dict[str, Any]]
) -> List[AnalyzedJob]:
    """Analyzes a list of jobs against the resume data."""
    analyzed_results: list[AnalyzedJob] = []
    total_jobs = len(job_list)
    log.info(f"Starting analysis of {total_jobs} jobs...")

    try: from rich.progress import track
    except ImportError:
        def track(iterable, description=""): log.info(description); return iterable

    for i, job_dict in enumerate(track(job_list, description="Analyzing jobs...")):
        job_title = job_dict.get('title', 'N/A')
        log.debug(f"Analyzing job {i+1}/{total_jobs}: {job_title}")

        analysis_result = analyzer.analyze_suitability(structured_resume_data, job_dict)

        if not analysis_result:
            log.warning(f"Analysis failed for job: {job_title}")
            # Create placeholder with score 0
            try:
                analysis_result_placeholder = JobAnalysisResult(
                    suitability_score=0, # Use 0
                    justification="Analysis failed or LLM response was invalid / Job data insufficient.",
                    skill_match=None, experience_match=None, qualification_match=None,
                    salary_alignment="N/A", benefit_alignment="N/A", missing_keywords=[] )
                analysis_result = analysis_result_placeholder
            except Exception as placeholder_err:
                log.error(f"CRITICAL: Failed to create placeholder JobAnalysisResult: {placeholder_err}", exc_info=True)
                continue # Skip appending this job

        analyzed_job = AnalyzedJob(original_job_data=job_dict, analysis=analysis_result)
        analyzed_results.append(analyzed_job)

    log.info(f"Analysis complete. Successfully analyzed (or attempted) {len(analyzed_results)} jobs.")
    return analyzed_results


def apply_filters_sort_and_save(
    analyzed_results: List[AnalyzedJob],
    output_path: str,
    filter_args: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Applies filters, sorts, and saves the final results."""
    jobs_to_filter = [res.original_job_data for res in analyzed_results]

    if filter_args:
        log.info("Applying post-analysis filters...")
        filtered_original_jobs = apply_filters(jobs_to_filter, **filter_args)
        log.info(f"{len(filtered_original_jobs)} jobs passed filters.")

        # Map back - consider using a unique job ID if available from scraping
        filtered_keys = set()
        for job in filtered_original_jobs:
             key = (job.get('url', job.get('job_url')), job.get('title'), job.get('company'), job.get('location'))
             filtered_keys.add(key)
        final_filtered_results = []
        for res in analyzed_results:
             original_job = res.original_job_data
             key = (original_job.get('url', original_job.get('job_url')), original_job.get('title'), original_job.get('company'), original_job.get('location'))
             if key in filtered_keys: final_filtered_results.append(res)
    else:
        final_filtered_results = analyzed_results

    log.info("Sorting results by suitability score...")
    final_filtered_results.sort(
        key=lambda x: x.analysis.suitability_score if x.analysis else 0, reverse=True )

    final_results_json = [result.model_dump(mode='json') for result in final_filtered_results] # Use mode='json' for better serialization

    output_dir = os.path.dirname(output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results_json, f, indent=4)
        log.info(f"Successfully saved {len(final_results_json)} analyzed jobs to {output_path}")
    except Exception as e: # Catch broader errors during save
        log.error(f"Error writing output file {output_path}: {e}", exc_info=True)
        log.debug(f"Problematic data structure (first item): {final_results_json[0] if final_results_json else 'N/A'}")

    return final_results_json

# --- Main execution block (for potential direct testing/use) ---
def main():
    """Main function for standalone execution."""
    # --- Argument parsing remains unchanged ---
    parser = argparse.ArgumentParser(description="Analyze pre-existing job JSON against a resume.")
    parser.add_argument("--resume", required=True, help="Path to the resume file (.docx or .pdf)")
    parser.add_argument("--jobs", required=True, help="Path to the JSON file containing job mandates (list of objects)")
    parser.add_argument("--output", default=config.DEFAULT_ANALYSIS_JSON, help="Path for the output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")
    parser.add_argument("--min-salary", type=int, help="Minimum desired annual salary for filtering.")
    parser.add_argument("--max-salary", type=int, help="Maximum desired annual salary for filtering.")
    parser.add_argument("--filter-remote-country", help="Filter REMOTE jobs within a specific country.")
    parser.add_argument("--filter-proximity-location", help="Reference location for proximity filtering.")
    parser.add_argument("--filter-proximity-range", type=float, help="Distance in miles for proximity.")
    parser.add_argument("--filter-proximity-models", default="Hybrid,On-site", help="Work models for proximity.")
    parser.add_argument("--filter-work-models", help="Standard work models (e.g., 'Remote,Hybrid').")
    parser.add_argument("--filter-job-types", help="Comma-separated job types (e.g., 'Full-time')")
    args = parser.parse_args()

    # --- Logging setup remains unchanged ---
    log_level = logging.DEBUG if args.verbose else config.LOG_LEVEL
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    log.info("Starting standalone analysis process...")

    try:
        analyzer = ResumeAnalyzer() # Use correct class name
    except Exception as e: log.error(f"Failed to initialize analyzer: {e}", exc_info=True); return

    structured_resume = load_and_extract_resume(args.resume, analyzer)
    if not structured_resume: log.error("Exiting due to resume processing failure."); return

    log.info(f"Loading jobs from JSON file: {args.jobs}")
    job_list = load_job_mandates(args.jobs)
    if not job_list: log.error("No jobs loaded from JSON file. Exiting."); return

    analyzed_results = analyze_jobs(analyzer, structured_resume, job_list)

    # --- Filter args population remains unchanged ---
    filter_args_dict = {}
    if args.min_salary is not None: filter_args_dict['salary_min'] = args.min_salary
    if args.max_salary is not None: filter_args_dict['salary_max'] = args.max_salary
    if args.filter_work_models: filter_args_dict['work_models'] = [wm.strip().lower() for wm in args.filter_work_models.split(',')]
    if args.filter_job_types: filter_args_dict['job_types'] = [jt.strip().lower() for jt in args.filter_job_types.split(',')]
    if args.filter_remote_country: filter_args_dict['filter_remote_country'] = args.filter_remote_country.strip()
    if args.filter_proximity_location:
         if args.filter_proximity_range is None: parser.error("--filter-proximity-range is required with --filter-proximity-location.")
         filter_args_dict['filter_proximity_location'] = args.filter_proximity_location.strip()
         filter_args_dict['filter_proximity_range'] = args.filter_proximity_range
         filter_args_dict['filter_proximity_models'] = [pm.strip().lower() for pm in args.filter_proximity_models.split(',')]
    elif args.filter_proximity_range is not None: parser.error("--filter-proximity-location is required with --filter-proximity-range.")

    apply_filters_sort_and_save(analyzed_results, args.output, filter_args_dict)

    log.info("Standalone analysis finished.")

if __name__ == "__main__":
    main()