# main_matcher.py
import logging
import json
import os
import argparse
import asyncio # Import asyncio
from analysis.analyzer import ResumeAnalyzer # Correct class name
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path # Use pathlib

from parsers.resume_parser import parse_resume
from parsers.job_parser import load_job_mandates
# Ensure correct Pydantic models are imported
from analysis.models import ResumeData, AnalyzedJob, JobAnalysisResult
from filtering.filter import apply_filters
# Use CFG for configuration values loaded from YAML
# Import DEFAULT path for standalone mode default output
from config import CFG, DEFAULT_ANALYSIS_JSON

log = logging.getLogger(__name__)

# --- load_and_extract_resume (async version - unchanged) ---
async def load_and_extract_resume(resume_path: str, analyzer: ResumeAnalyzer) -> Optional[ResumeData]:
    """ASYNC Loads resume, parses text, and extracts structured data."""
    log.info(f"Processing resume file: {resume_path}")
    try:
        # Run synchronous I/O in a thread pool executor
        loop = asyncio.get_running_loop()
        resume_text = await loop.run_in_executor(None, parse_resume, resume_path)
    except FileNotFoundError:
        log.error(f"Resume file not found at {resume_path}")
        return None
    except Exception as parse_err:
         log.error(f"Error parsing resume file {resume_path}: {parse_err}", exc_info=True)
         return None

    if not resume_text: log.error("Parsed empty resume text."); return None

    # Call the async extraction method
    structured_resume_data = await analyzer.extract_resume_data(resume_text) # Await async call
    if not structured_resume_data: log.error("Failed to extract structured data from resume."); return None

    log.info("Successfully extracted structured data from resume.")
    return structured_resume_data


# --- analyze_jobs (REVISED to use asyncio.gather) ---
async def analyze_jobs(
    analyzer: ResumeAnalyzer,
    structured_resume_data: ResumeData,
    job_list: List[Dict[str, Any]]
) -> List[AnalyzedJob]:
    """ASYNC Analyzes a list of jobs against the resume data concurrently using asyncio.gather."""
    analyzed_results: list[AnalyzedJob] = []
    total_jobs = len(job_list)
    if total_jobs == 0:
         log.warning("No jobs provided for analysis.")
         return []
    log.info(f"Starting ASYNC analysis of {total_jobs} jobs using asyncio.gather...")

    # --- Create a wrapper for robust error handling per task ---
    async def analyze_single_job_wrapper(job_dict):
        """Wrapper to call analyze_suitability and return placeholder on error/None."""
        job_title = job_dict.get('title', 'N/A') # Get title for logging
        analysis_result: Optional[JobAnalysisResult] = None
        try:
            # Await the actual analysis function
            analysis_result = await analyzer.analyze_suitability(structured_resume_data, job_dict)

            if not analysis_result:
                # This case handles graceful failures within analyze_suitability (e.g., missing desc)
                # OR if the LLM call itself returned None after retries.
                log.warning(f"Analysis failed or skipped for job: {job_title}")
                analysis_result = JobAnalysisResult(
                    suitability_score=0, justification="Analysis failed/skipped (LLM/Data issue).",
                    skill_match=None, experience_match=None, qualification_match=None,
                    salary_alignment="N/A", benefit_alignment="N/A", missing_keywords=[])
        except Exception as task_exc:
            # This catches unexpected errors during the await analyzer.analyze_suitability call
            log.error(f"Error processing analysis task for job '{job_title}': {task_exc}", exc_info=True)
            # Create placeholder on unexpected task error
            analysis_result = JobAnalysisResult(
                suitability_score=0, justification=f"Task Error: {type(task_exc).__name__}",
                skill_match=None, experience_match=None, qualification_match=None,
                salary_alignment="N/A", benefit_alignment="N/A", missing_keywords=[])

        # Always return an AnalyzedJob object, even if analysis failed
        return AnalyzedJob(original_job_data=job_dict, analysis=analysis_result)
    # --- End wrapper function ---

    # Create coroutines using the wrapper
    coroutines = [analyze_single_job_wrapper(job_dict) for job_dict in job_list]

    # Try using Rich Progress Bar
    try:
        from rich.progress import track
        from rich import get_console
        RICH_AVAILABLE_IN_ASYNC = True
        # Rich track doesn't directly support async iterables well for this pattern.
        # We'll use gather and log progress manually or just let gather run.
        log.info("Using asyncio.gather (Rich track not ideal for this gather pattern)...")

    except ImportError:
        RICH_PROGRESS_AVAILABLE = False
        log.warning("Rich progress bar unavailable for async analysis. Using basic logging.")

    # --- Use asyncio.gather to run all wrappers concurrently ---
    # return_exceptions=False because the wrapper now handles exceptions internally
    # and always returns an AnalyzedJob object.
    log.info(f"Submitting {len(coroutines)} analysis tasks to asyncio.gather...")
    start_time = time.time()

    # Run tasks - consider adding a timeout to gather if needed
    # results_list will contain AnalyzedJob objects (successful or placeholders)
    analyzed_results = await asyncio.gather(*coroutines)

    end_time = time.time()
    duration = end_time - start_time
    jobs_per_sec = total_jobs / duration if duration > 0 else 0
    log.info(f"Async analysis via gather complete. Processed {len(analyzed_results)}/{total_jobs} jobs in {duration:.2f} seconds ({jobs_per_sec:.2f} jobs/sec).")

    successful_analyses = sum(1 for res in analyzed_results if res.analysis and res.analysis.suitability_score > 0)
    log.info(f"Generated {successful_analyses} successful analysis results with score > 0.")

    return analyzed_results


# --- apply_filters_sort_and_save (Keep previous version with min_score handling) ---
def apply_filters_sort_and_save(
    analyzed_results: List[AnalyzedJob],
    output_path: str,
    filter_args: Dict[str, Any] # Combined dict of all filter criteria
) -> List[Dict[str, Any]]:
    """Applies filters (including min score), sorts, and saves results."""
    min_score = filter_args.pop('min_score', 0) # Extract min_score, default 0

    # 1. Apply standard/location/custom filters first
    jobs_to_filter = [res.original_job_data for res in analyzed_results]
    standard_filter_args = filter_args # Use remaining args

    if standard_filter_args:
        log.info("Applying standard/location/custom filters...")
        filtered_original_jobs = apply_filters(jobs_to_filter, **standard_filter_args)
        log.info(f"{len(filtered_original_jobs)} jobs passed standard filters.")
        # Map back using a key (URL or composite) - Ensure this key is reliable enough
        filtered_keys = set()
        for job in filtered_original_jobs:
             # Using URL as primary key, fallback to composite if URL missing
             key = job.get('url', job.get('job_url'))
             if not key:
                 key = (job.get('title'), job.get('company'), job.get('location'))
             filtered_keys.add(key)

        intermediate_filtered_results = []
        for res in analyzed_results:
             original_job = res.original_job_data
             key = original_job.get('url', original_job.get('job_url'))
             if not key:
                 key = (original_job.get('title'), original_job.get('company'), original_job.get('location'))
             if key in filtered_keys:
                  intermediate_filtered_results.append(res)
        log.debug(f"Intermediate results after standard filters: {len(intermediate_filtered_results)}")
    else:
        intermediate_filtered_results = analyzed_results # No standard filters applied

    # 2. Apply minimum score filter
    log.info(f"Applying minimum score filter (>= {min_score})...")
    # Ensure analysis score is not None before comparing
    score_filtered_results = [
        res for res in intermediate_filtered_results
        if res.analysis and res.analysis.suitability_score is not None and res.analysis.suitability_score >= min_score
    ]
    log.info(f"{len(score_filtered_results)} jobs passed minimum score filter.")
    final_filtered_results = score_filtered_results

    # 3. Sort the final filtered results
    log.info("Sorting final results by suitability score...")
    # Ensure analysis score is not None before sorting
    final_filtered_results.sort(
        key=lambda x: x.analysis.suitability_score if x.analysis and x.analysis.suitability_score is not None else 0,
        reverse=True )

    # 4. Convert and Save
    final_results_json = [result.model_dump(mode='json') for result in final_filtered_results]
    output_path_obj = Path(output_path) # Use pathlib consistently
    output_dir = output_path_obj.parent
    if output_dir: output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path_obj, 'w', encoding='utf-8') as f: json.dump(final_results_json, f, indent=4)
        log.info(f"Successfully saved {len(final_results_json)} final jobs to {output_path_obj}")
    except Exception as e:
        log.error(f"Error writing output file {output_path_obj}: {e}", exc_info=True)

    return final_results_json

# --- Main execution block (async runner - unchanged) ---
async def async_main(args):
    """Async version of the main logic for standalone run."""
    log.info("Starting standalone ASYNC analysis process...")
    analyzer = None
    try:
        # Initialize Analyzer (can raise errors)
        analyzer = ResumeAnalyzer()

        # Load/Cache Resume (can raise errors)
        structured_resume = await load_and_extract_resume(args.resume, analyzer)
        if not structured_resume:
            # Error logged within function, exit cleanly
            return

        # Load Jobs (can raise errors)
        log.info(f"Loading jobs from JSON file: {args.jobs}")
        try:
             loop = asyncio.get_running_loop()
             job_list = await loop.run_in_executor(None, load_job_mandates, args.jobs)
        except FileNotFoundError: log.error(f"Jobs file not found: {args.jobs}"); return
        except Exception as e: log.error(f"Error loading jobs JSON: {e}"); return
        if not job_list: log.error("No jobs loaded. Exiting."); return

        # Analyze Jobs (can raise errors within tasks, handled by wrapper)
        analyzed_results = await analyze_jobs(analyzer, structured_resume, job_list)

        # Filter/Save (sync, but wrap just in case)
        try:
            filter_args_dict = {}
            # Populate filter_args_dict including all new filters
            if args.min_salary is not None: filter_args_dict['salary_min'] = args.min_salary
            if args.max_salary is not None: filter_args_dict['salary_max'] = args.max_salary
            if args.filter_work_models: filter_args_dict['work_models'] = [wm.strip().lower() for wm in args.filter_work_models.split(',')]
            if args.filter_job_types: filter_args_dict['job_types'] = [jt.strip().lower() for jt in args.filter_job_types.split(',')]
            if args.filter_companies: filter_args_dict['filter_companies'] = [c.strip().lower() for c in args.filter_companies.split(',')]
            if args.exclude_companies: filter_args_dict['exclude_companies'] = [c.strip().lower() for c in args.exclude_companies.split(',')]
            if args.filter_title_keywords: filter_args_dict['filter_title_keywords'] = [k.strip().lower() for k in args.filter_title_keywords.split(',')]
            if args.filter_date_after: filter_args_dict['filter_date_after'] = args.filter_date_after
            if args.filter_date_before: filter_args_dict['filter_date_before'] = args.filter_date_before
            filter_args_dict['min_score'] = args.min_score # Pass min_score in dict
            if args.filter_remote_country: filter_args_dict['filter_remote_country'] = args.filter_remote_country.strip()
            if args.filter_proximity_location:
                 filter_args_dict['filter_proximity_location'] = args.filter_proximity_location.strip()
                 filter_args_dict['filter_proximity_range'] = args.filter_proximity_range
                 filter_args_dict['filter_proximity_models'] = [pm.strip().lower() for pm in args.filter_proximity_models.split(',')]

            # Call apply_filters_sort_and_save
            apply_filters_sort_and_save(analyzed_results, args.output, filter_args_dict)
        except Exception as e:
            log.error(f"Error during filtering/saving results: {e}", exc_info=True)

    except Exception as e:
        # Catch any other unexpected errors during the main async flow
        log.error(f"An unexpected error occurred in async_main: {e}", exc_info=True)
    finally:
         # Ensure client is closed even if errors occurred
         if analyzer and hasattr(analyzer, 'close'):
             await analyzer.close()
    log.info("Standalone analysis finished.")

def main():
    """Parses args and runs the async main function."""
    # --- Argument Parsing (Keep all filters defined) ---
    parser = argparse.ArgumentParser(description="Analyze pre-existing job JSON against a resume.")
    parser.add_argument("--resume", required=True, help="Path to resume file.")
    parser.add_argument("--jobs", required=True, help="Path to jobs JSON file.")
    parser.add_argument("--output", default=str(DEFAULT_ANALYSIS_JSON), help="Output JSON file path.") # Use default from config
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")
    parser.add_argument("--force-resume-reparse", action="store_true", help="Ignore cached resume data.") # Keep cache flag

    # Standard Filters
    filter_group = parser.add_argument_group('Standard Filtering Options')
    filter_group.add_argument("--min-salary", type=int, help="Min desired salary.")
    filter_group.add_argument("--max-salary", type=int, help="Max desired salary.")
    filter_group.add_argument("--filter-work-models", help="Comma-separated work models (e.g., 'Remote,Hybrid').")
    filter_group.add_argument("--filter-job-types", help="Comma-separated job types (e.g., 'Full-time').")
    filter_group.add_argument("--filter-companies", help="Include ONLY these companies (comma-sep, case-insensitive).")
    filter_group.add_argument("--exclude-companies", help="EXCLUDE these companies (comma-sep, case-insensitive).")
    filter_group.add_argument("--filter-title-keywords", help="Require ANY of these keywords in title (comma-sep, case-insensitive).")
    filter_group.add_argument("--filter-date-after", help="Include jobs posted ON or AFTER YYYY-MM-DD.")
    filter_group.add_argument("--filter-date-before", help="Include jobs posted ON or BEFORE YYYY-MM-DD.")
    filter_group.add_argument("--min-score", type=int, default=0, help="Minimum suitability score filter (0-100). Default 0.")

    # Advanced Location Filters
    adv_loc_group = parser.add_argument_group('Advanced Location Filtering')
    adv_loc_group.add_argument("--filter-remote-country", help="Filter REMOTE jobs within specific country.")
    adv_loc_group.add_argument("--filter-proximity-location", help="Reference location for proximity filtering.")
    adv_loc_group.add_argument("--filter-proximity-range", type=float, help="Distance in miles for proximity.")
    adv_loc_group.add_argument("--filter-proximity-models", default="Hybrid,On-site", help="Work models for proximity.")

    args = parser.parse_args()

    # --- Logging setup (Set level based on args AFTER parsing) ---
    log_level_arg = logging.DEBUG if args.verbose else CFG.get('logging', {}).get('level', 'INFO').upper()
    try: logging.getLogger().setLevel(log_level_arg); log.info(f"Log level set to: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")
    except ValueError: log.error(f"Invalid log level: {log_level_arg}. Using INFO."); logging.getLogger().setLevel(logging.INFO)

    # --- Run the main async logic ---
    try:
        # Check Python version for asyncio.run if needed (usually fine for 3.7+)
        if sys.version_info >= (3, 7):
            asyncio.run(async_main(args))
        else:
            # Fallback for older Python versions if strictly necessary
            loop = asyncio.get_event_loop()
            loop.run_until_complete(async_main(args))
    except KeyboardInterrupt:
        print(); log.warning("[yellow]Execution interrupted by user (Ctrl+C).[/yellow]" if RICH_AVAILABLE else "Execution interrupted.")
        # No explicit sys.exit needed here, KeyboardInterrupt termination handles it

if __name__ == "__main__":
    main()
