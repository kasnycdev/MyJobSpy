import logging
import json
import os
import argparse
from typing import List, Dict, Any, Optional, Tuple

from parsers.resume_parser import parse_resume
from parsers.job_parser import load_job_mandates # Kept for direct JSON input if needed
from analysis.analyzer import ResumeAnalyzer
from analysis.models import ResumeData, AnalyzedJob, JobAnalysisResult
from filtering.filter import apply_filters
import config

# Setup logger
log = logging.getLogger(__name__)

def load_and_extract_resume(resume_path: str, analyzer: ResumeAnalyzer) -> Optional[ResumeData]:
    """Loads resume, parses text, and extracts structured data."""
    log.info(f"Processing resume file: {resume_path}")
    resume_text = parse_resume(resume_path)
    if not resume_text:
        log.error("Failed to parse resume text.")
        return None
    structured_resume_data = analyzer.extract_resume_data(resume_text)
    if not structured_resume_data:
        log.error("Failed to extract structured data from resume.")
        return None
    log.info("Successfully extracted structured data from resume.")
    return structured_resume_data

def analyze_jobs(
    analyzer: ResumeAnalyzer,
    structured_resume_data: ResumeData,
    job_list: List[Dict[str, Any]]
) -> List[AnalyzedJob]:
    """Analyzes a list of jobs against the resume data."""
    analyzed_results: list[AnalyzedJob] = []
    total_jobs = len(job_list)
    log.info(f"Starting analysis of {total_jobs} jobs...")

    from rich.progress import track # Use rich progress bar here

    for i, job_dict in enumerate(track(job_list, description="Analyzing jobs...")):
        job_title = job_dict.get('title', 'N/A')
        log.debug(f"Analyzing job {i+1}/{total_jobs}: {job_title}") # Debug level for less noise

        # Ensure job_dict is suitable (e.g., has description) - Analyzer does this now
        analysis_result = analyzer.analyze_suitability(structured_resume_data, job_dict)

        analysis_status = "success" if analysis_result else "failed"
        if not analysis_result:
            log.warning(f"Analysis failed for job: {job_title}")
            # Create a placeholder analysis if desired, or skip
            analysis_result = JobAnalysisResult(
                 suitability_score=0, # Indicate failure
                 justification="Analysis failed or was skipped.",
                 skill_match=None, experience_match=None, qualification_match=None,
                 salary_alignment="N/A", benefit_alignment="N/A", missing_keywords=[]
             )

        # Combine original job data with analysis results
        # Add analysis status if needed
        # job_dict["analysis_status"] = analysis_status # Optional: Add status to original data
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

    # Filtering needs the data in a list of dicts format that filter.py expects
    # We need to decide: Filter based on original data or analyzed data?
    # Let's filter based on original_job_data for salary, location etc.
    jobs_to_filter = [res.original_job_data for res in analyzed_results]

    if filter_args:
        log.info("Applying post-analysis filters...")
        filtered_original_jobs = apply_filters(jobs_to_filter, **filter_args)
        log.info(f"{len(filtered_original_jobs)} jobs passed filters.")
        # Now, map back to the full AnalyzedJob objects
        filtered_titles_urls = {(job.get('title', ''), job.get('url', '')) for job in filtered_original_jobs}
        final_filtered_results = [
            res for res in analyzed_results
            if (res.original_job_data.get('title', ''), res.original_job_data.get('url', '')) in filtered_titles_urls
        ]
    else:
        final_filtered_results = analyzed_results # No filters applied

    # Sort the filtered results by suitability score (descending)
    # Handle potential -1 scores from failed analyses (place them last)
    log.info("Sorting results by suitability score...")
    final_filtered_results.sort(
        key=lambda x: x.analysis.suitability_score if x.analysis else -1,
        reverse=True
    )

    # Convert Pydantic models to dictionaries for JSON serialization
    final_results_json = [result.model_dump() for result in final_filtered_results]

    # Save Output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results_json, f, indent=4)
        log.info(f"Successfully saved {len(final_results_json)} analyzed and filtered jobs to {output_path}")
    except IOError as e:
        log.error(f"Error writing output file {output_path}: {e}")
    except TypeError as e:
        log.error(f"Error serializing results to JSON: {e}. Check data types in models.", exc_info=True)
        log.debug(f"Problematic data structure (first item): {final_results_json[0] if final_results_json else 'N/A'}")

    return final_results_json # Return the final list of dicts

# --- Main execution block (for potential direct testing/use) ---
def main():
    """Main function for standalone execution (less common now)."""
    parser = argparse.ArgumentParser(description="Analyze pre-existing job JSON against a resume.")
    parser.add_argument("--resume", required=True, help="Path to the resume file (.docx or .pdf)")
    parser.add_argument("--jobs", required=True, help="Path to the JSON file containing job mandates (list of objects)")
    parser.add_argument("--output", default=config.DEFAULT_ANALYSIS_JSON, help="Path for the output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")

    # Add filtering arguments mirroring run_pipeline.py
    parser.add_argument("--min-salary", type=int, help="Minimum desired annual salary for filtering.")
    parser.add_argument("--max-salary", type=int, help="Maximum desired annual salary for filtering.")
    parser.add_argument("--filter-locations", help="Comma-separated list of desired locations for filtering (e.g., 'New York,Remote')")
    parser.add_argument("--filter-work-models", help="Comma-separated list of desired work models for filtering (e.g., 'Remote,Hybrid')")
    parser.add_argument("--filter-job-types", help="Comma-separated list of desired job types for filtering (e.g., 'Full-time')")

    args = parser.parse_args()

    # Setup logging based on verbosity
    log_level = logging.DEBUG if args.verbose else config.LOG_LEVEL
    logging.basicConfig(level=log_level, format=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    # Consider using RichHandler here too if running standalone
    # from rich.logging import RichHandler
    # logging.basicConfig(level=log_level, format=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT, handlers=[RichHandler()])


    log.info("Starting standalone analysis process...")

    try:
        analyzer = ResumeAnalyzer()
    except Exception as e:
        log.error(f"Failed to initialize analyzer: {e}", exc_info=True)
        return

    structured_resume = load_and_extract_resume(args.resume, analyzer)
    if not structured_resume:
        log.error("Exiting due to resume processing failure.")
        return

    log.info(f"Loading jobs from JSON file: {args.jobs}")
    job_list = load_job_mandates(args.jobs) # Using the original JSON loader here
    if not job_list:
        log.error("No jobs loaded from JSON file. Exiting.")
        return

    analyzed_results = analyze_jobs(analyzer, structured_resume, job_list)

    filter_args = {}
    if args.min_salary is not None: filter_args['salary_min'] = args.min_salary
    if args.max_salary is not None: filter_args['salary_max'] = args.max_salary
    if args.filter_locations: filter_args['locations'] = [loc.strip() for loc in args.filter_locations.split(',')]
    if args.filter_work_models: filter_args['work_models'] = [wm.strip() for wm in args.filter_work_models.split(',')]
    if args.filter_job_types: filter_args['job_types'] = [jt.strip() for jt in args.filter_job_types.split(',')]

    apply_filters_sort_and_save(analyzed_results, args.output, filter_args)

    log.info("Standalone analysis finished.")

if __name__ == "__main__":
    # This allows running `python main_matcher.py --resume ... --jobs ...`
    main()