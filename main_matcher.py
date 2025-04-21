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
from analysis.models import ResumeData, AnalyzedJob, JobAnalysisResult
from filtering.filter import apply_filters
# Use CFG for configuration values loaded from YAML
from config import CFG, DEFAULT_ANALYSIS_JSON

log = logging.getLogger(__name__)

# --- load_and_extract_resume becomes ASYNC ---
async def load_and_extract_resume(resume_path: str, analyzer: ResumeAnalyzer) -> Optional[ResumeData]:
    """ASYNC Loads resume, parses text, and extracts structured data."""
    log.info(f"Processing resume file: {resume_path}")
    try:
        # Run synchronous I/O in a thread pool executor to avoid blocking async loop
        loop = asyncio.get_running_loop()
        resume_text = await loop.run_in_executor(None, parse_resume, resume_path)
        # resume_text = parse_resume(resume_path) # Simpler if parse_resume is very fast

    except FileNotFoundError:
        log.error(f"Resume file not found at {resume_path}")
        return None
    except Exception as parse_err:
         log.error(f"Error parsing resume file {resume_path}: {parse_err}", exc_info=True)
         return None

    if not resume_text: log.error("Failed to parse resume text."); return None

    # Call the async extraction method
    structured_resume_data = await analyzer.extract_resume_data(resume_text) # Await async call
    if not structured_resume_data: log.error("Failed to extract structured data from resume."); return None

    log.info("Successfully extracted structured data from resume.")
    return structured_resume_data

# --- analyze_jobs becomes ASYNC ---
async def analyze_jobs(
    analyzer: ResumeAnalyzer,
    structured_resume_data: ResumeData,
    job_list: List[Dict[str, Any]]
) -> List[AnalyzedJob]:
    """ASYNC Analyzes a list of jobs against the resume data concurrently."""
    analyzed_results: list[AnalyzedJob] = []
    total_jobs = len(job_list)
    if total_jobs == 0:
         log.warning("No jobs provided for analysis.")
         return []
    log.info(f"Starting ASYNC analysis of {total_jobs} jobs...")

    # Create tasks for all jobs
    tasks = []
    for job_dict in job_list:
        task = asyncio.create_task(analyzer.analyze_suitability(structured_resume_data, job_dict))
        tasks.append((task, job_dict)) # Store task and original job data

    # Process tasks as they complete with Rich progress
    processed_count = 0
    try:
        # Use Rich progress bar if available
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskID
        RICH_PROGRESS_AVAILABLE = True
    except ImportError:
        RICH_PROGRESS_AVAILABLE = False
        log.warning("Rich progress bar unavailable for async analysis.")

    async def _process_tasks_rich():
        nonlocal processed_count, analyzed_results # Allow modification
        with Progress( SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                       TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), transient=False, ) as progress:
            analysis_task_id: TaskID = progress.add_task("[cyan]Analyzing jobs...", total=total_jobs)
            # Use asyncio.as_completed to process tasks as they finish
            for future in asyncio.as_completed([t[0] for t in tasks]):
                # Find original job data associated with this future - less efficient but works
                # A better way might be to store future->job_dict mapping
                original_job_dict = None
                for task_tuple in tasks:
                    if task_tuple[0] == future:
                        original_job_dict = task_tuple[1]
                        break

                job_title = original_job_dict.get('title', 'N/A') if original_job_dict else "Unknown Job"
                try:
                    analysis_result = await future # Get result (or raise exception)
                    if not analysis_result:
                        log.warning(f"Analysis failed for job: {job_title}")
                        analysis_result = JobAnalysisResult(suitability_score=0, justification="Analysis failed/skipped.",
                                                            skill_match=None, experience_match=None, qualification_match=None,
                                                            salary_alignment="N/A", benefit_alignment="N/A", missing_keywords=[])
                    analyzed_job = AnalyzedJob(original_job_data=original_job_dict, analysis=analysis_result)
                    analyzed_results.append(analyzed_job)
                except Exception as task_exc:
                     log.error(f"Error processing analysis task for job '{job_title}': {task_exc}", exc_info=True)
                     # Append a failed state if needed, or just skip
                     # Optionally create placeholder here too if needed
                finally:
                     processed_count += 1
                     progress.update(analysis_task_id, advance=1)

    async def _process_tasks_basic():
         nonlocal processed_count, analyzed_results
         # Process sequentially if rich is not available (simpler fallback)
         for i, (task_future, job_dict) in enumerate(tasks):
            job_title = job_dict.get('title', 'N/A')
            if i % 20 == 0 or i == total_jobs -1: # Log progress occasionally
                log.info(f"Analyzing job {i+1}/{total_jobs} ('{job_title}')...")
            try:
                analysis_result = await task_future
                if not analysis_result:
                    log.warning(f"Analysis failed for job: {job_title}")
                    analysis_result = JobAnalysisResult(suitability_score=0, justification="Analysis failed/skipped.",
                                                        skill_match=None, experience_match=None, qualification_match=None,
                                                        salary_alignment="N/A", benefit_alignment="N/A", missing_keywords=[])
                analyzed_job = AnalyzedJob(original_job_data=job_dict, analysis=analysis_result)
                analyzed_results.append(analyzed_job)
            except Exception as task_exc:
                log.error(f"Error processing analysis task for job '{job_title}': {task_exc}", exc_info=True)
            finally:
                processed_count += 1


    # Run the appropriate processing function
    if RICH_PROGRESS_AVAILABLE:
         await _process_tasks_rich()
    else:
         await _process_tasks_basic()

    log.info(f"Async analysis complete. Processed {processed_count}/{total_jobs} jobs. Successfully analyzed: {len(analyzed_results)}")
    return analyzed_results


# --- apply_filters_sort_and_save adds MIN_SCORE filter ---
def apply_filters_sort_and_save(
    analyzed_results: List[AnalyzedJob],
    output_path: str,
    filter_args: Dict[str, Any] # Combined dict of all filter criteria
) -> List[Dict[str, Any]]:
    """Applies filters (including min score), sorts, and saves results."""
    min_score = filter_args.pop('min_score', None) # Extract min_score, remove from standard args

    # 1. Apply standard/location filters first using the remaining filter_args
    jobs_to_filter = [res.original_job_data for res in analyzed_results]

    if filter_args: # Check if any standard/location filters remain
        log.info("Applying standard/location/custom filters...")
        filtered_original_jobs = apply_filters(jobs_to_filter, **filter_args)
        log.info(f"{len(filtered_original_jobs)} jobs passed standard/location/custom filters.")
        # Map back
        filtered_keys = set()
        for job in filtered_original_jobs:
             key = (job.get('url', job.get('job_url')), job.get('title'), job.get('company'), job.get('location'))
             filtered_keys.add(key)
        intermediate_filtered_results = []
        for res in analyzed_results:
             original_job = res.original_job_data
             key = (original_job.get('url', original_job.get('job_url')), original_job.get('title'), original_job.get('company'), original_job.get('location'))
             if key in filtered_keys: intermediate_filtered_results.append(res)
    else:
        intermediate_filtered_results = analyzed_results # No standard filters applied

    # 2. Apply minimum score filter (if provided)
    if min_score is not None:
         log.info(f"Applying minimum score filter (>= {min_score})...")
         score_filtered_results = [
             res for res in intermediate_filtered_results
             # Ensure analysis exists and score is not None before comparing
             if res.analysis and res.analysis.suitability_score is not None and res.analysis.suitability_score >= min_score
         ]
         log.info(f"{len(score_filtered_results)} jobs passed minimum score filter.")
         final_filtered_results = score_filtered_results
    else:
         final_filtered_results = intermediate_filtered_results


    # 3. Sort the final filtered results
    log.info("Sorting final results by suitability score...")
    final_filtered_results.sort(
        key=lambda x: x.analysis.suitability_score if x.analysis and x.analysis.suitability_score is not None else 0,
        reverse=True )

    # 4. Convert and Save
    final_results_json = [result.model_dump(mode='json') for result in final_filtered_results]
    output_path_obj = Path(output_path) # Use pathlib
    output_dir = output_path_obj.parent
    if output_dir: output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path_obj, 'w', encoding='utf-8') as f: json.dump(final_results_json, f, indent=4)
        log.info(f"Successfully saved {len(final_results_json)} final jobs to {output_path_obj}")
    except Exception as e:
        log.error(f"Error writing output file {output_path_obj}: {e}", exc_info=True)
        log.debug(f"Problematic data structure (first item): {final_results_json[0] if final_results_json else 'N/A'}")

    return final_results_json

# --- Main execution block (becomes async runner) ---
async def async_main(args):
    """Async version of the main logic for standalone run."""
    log.info("Starting standalone ASYNC analysis process...")
    analyzer = None # Define upfront for finally block
    try:
        analyzer = ResumeAnalyzer()
        structured_resume = await load_and_extract_resume(args.resume, analyzer)
        if not structured_resume: log.error("Exiting due to resume processing failure."); return

        log.info(f"Loading jobs from JSON file: {args.jobs}")
        try:
             # Run sync file loading in executor
             loop = asyncio.get_running_loop()
             job_list = await loop.run_in_executor(None, load_job_mandates, args.jobs)
        except FileNotFoundError: log.error(f"Jobs file not found: {args.jobs}"); return
        except Exception as e: log.error(f"Error loading jobs JSON: {e}"); return
        if not job_list: log.error("No jobs loaded from JSON file. Exiting."); return

        analyzed_results = await analyze_jobs(analyzer, structured_resume, job_list)

        # Filtering/saving is sync
        filter_args_dict = {}
        # --- Populate filter_args_dict including new filters ---
        if args.min_salary is not None: filter_args_dict['salary_min'] = args.min_salary
        if args.max_salary is not None: filter_args_dict['salary_max'] = args.max_salary
        if args.filter_work_models: filter_args_dict['work_models'] = [wm.strip().lower() for wm in args.filter_work_models.split(',')]
        if args.filter_job_types: filter_args_dict['job_types'] = [jt.strip().lower() for jt in args.filter_job_types.split(',')]
        if args.filter_companies: filter_args_dict['filter_companies'] = [c.strip() for c in args.filter_companies.split(',')]
        if args.exclude_companies: filter_args_dict['exclude_companies'] = [c.strip() for c in args.exclude_companies.split(',')]
        if args.filter_title_keywords: filter_args_dict['filter_title_keywords'] = [k.strip() for k in args.filter_title_keywords.split(',')]
        if args.filter_date_after: filter_args_dict['filter_date_after'] = args.filter_date_after
        if args.filter_date_before: filter_args_dict['filter_date_before'] = args.filter_date_before
        filter_args_dict['min_score'] = args.min_score # Add min_score here too for consistency if needed by apply_filters directly
        if args.filter_remote_country: filter_args_dict['filter_remote_country'] = args.filter_remote_country.strip()
        if args.filter_proximity_location:
             filter_args_dict['filter_proximity_location'] = args.filter_proximity_location.strip()
             filter_args_dict['filter_proximity_range'] = args.filter_proximity_range
             filter_args_dict['filter_proximity_models'] = [pm.strip().lower() for pm in args.filter_proximity_models.split(',')]
        # ---

        apply_filters_sort_and_save(analyzed_results, args.output, filter_args_dict, args.min_score) # Pass min_score separately

    finally:
         if analyzer: await analyzer.close() # Close async client

    log.info("Standalone analysis finished.")

def main():
    """Parses args and runs the async main function."""
    # --- Add NEW filter arguments ---
    parser = argparse.ArgumentParser(description="Analyze pre-existing job JSON against a resume.")
    parser.add_argument("--resume", required=True, help="Path to resume file.")
    parser.add_argument("--jobs", required=True, help="Path to jobs JSON file.")
    parser.add_argument("--output", default=str(DEFAULT_ANALYSIS_JSON), help="Output JSON file path.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    parser.add_argument("--min-salary", type=int, help="Min desired salary.")
    parser.add_argument("--max-salary", type=int, help="Max desired salary.")
    parser.add_argument("--filter-work-models", help="Standard work models.")
    parser.add_argument("--filter-job-types", help="Comma-separated job types.")
    parser.add_argument("--filter-companies", help="Include ONLY these companies (comma-sep).")
    parser.add_argument("--exclude-companies", help="EXCLUDE these companies (comma-sep).")
    parser.add_argument("--filter-title-keywords", help="Require ANY of these keywords in title (comma-sep).")
    parser.add_argument("--filter-date-after", help="Include jobs posted ON or AFTER YYYY-MM-DD.")
    parser.add_argument("--filter-date-before", help="Include jobs posted ON or BEFORE YYYY-MM-DD.")
    parser.add_argument("--min-score", type=int, default=0, help="Minimum suitability score filter (0-100). Default 0.")
    parser.add_argument("--filter-remote-country", help="Filter REMOTE jobs in country.")
    parser.add_argument("--filter-proximity-location", help="Reference location for proximity.")
    parser.add_argument("--filter-proximity-range", type=float, help="Distance in miles for proximity.")
    parser.add_argument("--filter-proximity-models", default="Hybrid,On-site", help="Work models for proximity.")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else CFG['logging']['level']
    # Ensure logger level is set correctly based on args BEFORE running async_main
    logging.getLogger().setLevel(log_level.upper())

    # Run the async main function using asyncio.run()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
         print(); log.warning("[yellow]Standalone execution interrupted.[/yellow]" if RICH_AVAILABLE else "Standalone execution interrupted.")

if __name__ == "__main__":
    main()