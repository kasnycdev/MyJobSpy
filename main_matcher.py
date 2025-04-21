import logging
import json
import os
import argparse
import asyncio # Import asyncio
from typing import List, Dict, Any, Optional, Tuple

# Use ResumeAnalyzer which now has async methods
from analysis.analyzer import ResumeAnalyzer
from parsers.resume_parser import parse_resume
from parsers.job_parser import load_job_mandates
from analysis.models import ResumeData, AnalyzedJob, JobAnalysisResult
from filtering.filter import apply_filters # filter functions remain synchronous
import config

# Setup logger
log = logging.getLogger(__name__)

# --- ASYNC Resume Loading/Extraction ---
async def load_and_extract_resume_async(resume_path: str, analyzer: ResumeAnalyzer) -> Optional[ResumeData]:
    """ASYNC: Loads resume, parses text, and extracts structured data."""
    log.info(f"Processing resume file: {resume_path}")
    # Parsing is CPU bound, keep it synchronous for now
    try:
        resume_text = parse_resume(resume_path)
    except Exception as e:
        log.error(f"Failed to parse resume text from {resume_path}: {e}", exc_info=True)
        return None

    if not resume_text:
        log.error("Parsed resume text is empty.")
        return None

    # Extraction uses LLM, make it async
    structured_resume_data = await analyzer.extract_resume_data_async(resume_text)
    if not structured_resume_data:
        log.error("Failed to extract structured data from resume.")
        return None
    log.info("Successfully extracted structured data from resume.")
    return structured_resume_data

# --- ASYNC Job Analysis ---
async def analyze_jobs_async(
    analyzer: ResumeAnalyzer,
    structured_resume_data: ResumeData,
    job_list: List[Dict[str, Any]]
) -> List[AnalyzedJob]:
    """ASYNC: Analyzes a list of jobs concurrently against the resume data."""
    analyzed_results: list[AnalyzedJob] = []
    total_jobs = len(job_list)
    log.info(f"Starting ASYNC analysis of {total_jobs} jobs...")

    # Create tasks for all analysis calls
    tasks = []
    for job_dict in job_list:
        # Create a coroutine object for each job analysis
        coro = analyzer.analyze_suitability_async(structured_resume_data, job_dict)
        # Create a task from the coroutine
        task = asyncio.create_task(coro)
        tasks.append(task)

    # Run tasks concurrently and gather results
    # Use rich progress bar here
    try:
        from rich.progress import Progress
        progress_context = Progress(
                "[progress.description]{task.description}",
                SpinnerColumn(),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TextColumn("[progress.completed]{task.completed} of {task.total} analyzed"),
                TimeElapsedColumn() )
    except ImportError:
         log.warning("Rich library not found, progress bar disabled.")
         progress_context = None # Fallback


    results_or_exceptions = []
    if progress_context:
         with progress_context as progress:
              analysis_task_tracker = progress.add_task("[cyan]Analyzing Jobs...", total=len(tasks))
              results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)
              progress.update(analysis_task_tracker, completed=len(tasks)) # Mark as complete after gather
    else: # Run without progress bar
         log.info(f"Running {len(tasks)} analysis tasks concurrently...")
         results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

    log.info("Processing analysis results...")
    # Process results, pairing back with original job data
    for i, result_or_exc in enumerate(results_or_exceptions):
        job_dict = job_list[i] # Get corresponding original job
        job_title = job_dict.get('title', 'N/A')
        analysis_result = None

        if isinstance(result_or_exc, Exception):
            log.warning(f"Analysis task for job '{job_title}' failed with exception: {result_or_exc}", exc_info=result_or_exc)
        elif result_or_exc is None:
            log.warning(f"Analysis task for job '{job_title}' returned None (likely LLM failure or missing description).")
        else:
            # Successfully received JobAnalysisResult object
            analysis_result = result_or_exc

        # Create placeholder if analysis failed or returned None
        if analysis_result is None:
            try:
                analysis_result_placeholder = JobAnalysisResult(
                    suitability_score=0, # Use 0
                    justification="Analysis task failed, returned None, or job description was missing.",
                    skill_match=None, experience_match=None, qualification_match=None,
                    salary_alignment="N/A", benefit_alignment="N/A", missing_keywords=[] )
                analysis_result = analysis_result_placeholder
            except Exception as placeholder_err:
                log.error(f"CRITICAL: Failed to create placeholder for failed analysis of '{job_title}': {placeholder_err}", exc_info=True)
                continue # Skip this job entirely

        analyzed_job = AnalyzedJob(original_job_data=job_dict, analysis=analysis_result)
        analyzed_results.append(analyzed_job)

    log.info(f"ASYNC analysis complete. Processed {len(analyzed_results)} jobs.")
    return analyzed_results


# --- apply_filters_sort_and_save remains synchronous ---
def apply_filters_sort_and_save(
    analyzed_results: List[AnalyzedJob],
    output_path: str,
    filter_args: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Applies filters, sorts, and saves the final results."""
    # (Function content remains the same as previous version)
    jobs_to_filter = [res.original_job_data for res in analyzed_results]
    if filter_args:
        log.info("Applying post-analysis filters...")
        filtered_original_jobs = apply_filters(jobs_to_filter, **filter_args)
        log.info(f"{len(filtered_original_jobs)} jobs passed filters.")
        filtered_keys = set()
        for job in filtered_original_jobs:
             key = (job.get('url', job.get('job_url')), job.get('title'), job.get('company'), job.get('location'))
             filtered_keys.add(key)
        final_filtered_results = []
        for res in analyzed_results:
             original_job = res.original_job_data
             key = (original_job.get('url', original_job.get('job_url')), original_job.get('title'), original_job.get('company'), original_job.get('location'))
             if key in filtered_keys: final_filtered_results.append(res)
    else: final_filtered_results = analyzed_results
    log.info("Sorting results by suitability score...")
    final_filtered_results.sort(key=lambda x: x.analysis.suitability_score if x.analysis else 0, reverse=True )
    final_results_json = [result.model_dump(mode='json') for result in final_filtered_results]
    output_dir = os.path.dirname(output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(final_results_json, f, indent=4)
        log.info(f"Successfully saved {len(final_results_json)} analyzed jobs to {output_path}")
    except Exception as e:
        log.error(f"Error writing output file {output_path}: {e}", exc_info=True)
        log.debug(f"Problematic data structure (first item): {final_results_json[0] if final_results_json else 'N/A'}")
    return final_results_json


# --- Main execution block updated for async ---
async def main_async(): # Changed to async def
    """Async main function for standalone execution."""
    # (Argument parsing remains the same)
    parser = argparse.ArgumentParser(description="Analyze pre-existing job JSON against a resume.")
    # ... add all arguments ...
    args = parser.parse_args()

    # (Logging setup remains the same)
    log_level = logging.DEBUG if args.verbose else config.LOG_LEVEL
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    log.info("Starting standalone ASYNC analysis process...")

    try:
        analyzer = ResumeAnalyzer() # Instantiates class with sync/async clients
    except Exception as e: log.error(f"Failed to initialize analyzer: {e}", exc_info=True); return

    # Call async version of resume extraction
    structured_resume = await load_and_extract_resume_async(args.resume, analyzer)
    if not structured_resume: log.error("Exiting due to resume processing failure."); return

    log.info(f"Loading jobs from JSON file: {args.jobs}")
    job_list = load_job_mandates(args.jobs)
    if not job_list: log.error("No jobs loaded from JSON file. Exiting."); return

    # Call async version of job analysis
    analyzed_results = await analyze_jobs_async(analyzer, structured_resume, job_list)

    # (Filter args population remains the same)
    filter_args_dict = {}
    # ... populate filter_args_dict ...

    # Filtering/saving remains synchronous
    apply_filters_sort_and_save(analyzed_results, args.output, filter_args_dict)

    log.info("Standalone ASYNC analysis finished.")

if __name__ == "__main__":
    # Use asyncio.run() to execute the async main function
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
         print("\nStandalone analysis interrupted by user.")
         sys.exit(130)