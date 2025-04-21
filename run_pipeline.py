# run_pipeline.py
import argparse
import logging
import json
import os
import sys
import hashlib
import traceback
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

# Use the jobspy library for scraping
try: import jobspy; from jobspy import scrape_jobs
except ImportError: print("CRITICAL ERROR: 'jobspy' library not found..."); sys.exit(1)

# Import analysis components
try:
    from main_matcher import load_and_extract_resume, analyze_jobs, apply_filters_sort_and_save
    from analysis.analyzer import ResumeAnalyzer
    from analysis.models import ResumeData
except ImportError as e: print(f"CRITICAL ERROR: Could not import analysis functions: {e}"); sys.exit(1)
except Exception as e: print(f"CRITICAL ERROR during imports: {e}"); traceback.print_exc(); sys.exit(1)

# Load configuration
try: import config
except Exception as e: print(f"CRITICAL ERROR: Failed to import config.py: {e}"); traceback.print_exc(); sys.exit(1)

# Rich for UX
try: from rich.console import Console; from rich.logging import RichHandler; from rich.table import Table; RICH_AVAILABLE = True
except ImportError: RICH_AVAILABLE = False; print("Warning: 'rich' library not found...")

# Setup logging
log = logging.getLogger(__name__)
log_level_str = config.CFG.get('logging', {}).get('level', 'INFO').upper() # Get level from config first
if RICH_AVAILABLE:
    logging.basicConfig(level=log_level_str, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True, show_path=False)], force=True)
    console = Console()
else:
     logging.basicConfig(level=log_level_str, format='%(asctime)s-%(levelname)s-%(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
if logging.getLogger().getEffectiveLevel() > logging.DEBUG: logging.getLogger("geopy").setLevel(logging.INFO)


# --- scrape_jobs_with_jobspy function (unchanged) ---
def scrape_jobs_with_jobspy(search_terms, location, sites, results_wanted, hours_old, country_indeed, proxies=None, offset=0) -> Optional[pd.DataFrame]:
    # ... (Keep previous version) ...
    pass

# --- convert_and_save_scraped function (unchanged) ---
def convert_and_save_scraped(jobs_df: pd.DataFrame, output_path: Path) -> List[Dict[str, Any]]:
    # ... (Keep previous version) ...
    pass

# --- print_summary_table function (unchanged) ---
def print_summary_table(results_json: Optional[List[Dict[str, Any]]], top_n: int = 10):
    # ... (Keep previous version) ...
    pass

# --- Resume Caching Helper ---
def get_resume_hash(filepath: str) -> Optional[str]:
    """Calculates SHA256 hash of a file's content."""
    try:
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as file:
            while chunk := file.read(4096): hasher.update(chunk) # Walrus operator Py 3.8+
        return hasher.hexdigest()
    except FileNotFoundError: log.error(f"Resume file not found at {filepath} for hashing."); return None
    except Exception as e: log.error(f"Error hashing resume file {filepath}: {e}", exc_info=True); return None

# --- Main Execution Logic (Async Wrapper) ---
async def async_pipeline_logic(args):
    """The core logic wrapped in an async function."""
    config.ensure_required_dirs() # Ensure output and cache dirs exist

    # --- Determine scrape location ---
    scrape_location = None
    if args.filter_remote_country: scrape_location = args.filter_remote_country.strip(); log.info(f"Using country '{scrape_location}' as primary scrape location (from --filter-remote-country).")
    elif args.filter_proximity_location: scrape_location = args.filter_proximity_location.strip(); log.info(f"Using proximity target '{scrape_location}' as primary scrape location.")
    elif args.location: scrape_location = args.location; log.info(f"Using provided --location '{scrape_location}' as primary scrape location.")
    # Validation ensures one is set

    # --- Step 1: Scrape Jobs ---
    log.info("--- Stage 1: Scraping Jobs ---")
    scraper_sites = [site.strip().lower() for site in args.sites.split(',')]
    proxy_list = [p.strip() for p in args.proxies.split(',')] if args.proxies else None
    jobs_df = scrape_jobs_with_jobspy(
        search_terms=args.search, location=scrape_location, sites=scraper_sites, results_wanted=args.results,
        hours_old=args.hours_old, country_indeed=args.country_indeed, proxies=proxy_list, offset=args.offset )

    if jobs_df is None or jobs_df.empty:
        log.warning("Scraping yielded no results. Pipeline finished.")
        # ... (handle empty analysis output file) ...
        return # Exit async function

    # --- Step 2: Convert and Save Scraped Data ---
    log.info("--- Stage 2: Processing Scraped Data ---")
    # Use Path object from config
    scraped_jobs_path = config.PROJECT_ROOT / config.CFG['output']['directory'] / config.CFG['output']['scraped_json_filename']
    jobs_list = convert_and_save_scraped(jobs_df, scraped_jobs_path)
    if not jobs_list: raise RuntimeError("Failed to convert or save scraped data.")

    # --- Step 3: Initialize Analyzer and Load/Cache Resume ---
    log.info("--- Stage 3: Initializing Analyzer and Processing Resume ---")
    analyzer: Optional[ResumeAnalyzer] = None
    structured_resume: Optional[ResumeData] = None
    try:
        analyzer = ResumeAnalyzer() # Can raise errors during init/check

        # --- Resume Caching Logic ---
        resume_hash = get_resume_hash(args.resume)
        cache_filepath = None
        if resume_hash:
             cache_filename = f"{resume_hash}.json"
             cache_filepath = config.RESUME_CACHE_DIR / cache_filename # Use Path object from config
             log.debug(f"Resume hash: {resume_hash}, Cache file path: {cache_filepath}")

        if not args.force_resume_reparse and cache_filepath and cache_filepath.exists():
            log.info(f"Loading structured resume data from cache: {cache_filepath}")
            try:
                with open(cache_filepath, 'r', encoding='utf-8') as f: cached_data_json = f.read()
                structured_resume = ResumeData.model_validate_json(cached_data_json)
                log.info("Successfully loaded resume data from cache.")
            except Exception as e: log.warning(f"Failed to load resume from cache: {e}. Re-parsing.", exc_info=True); structured_resume = None
        else:
             if cache_filepath: log.info(f"Cache not found or --force-resume-reparse. Parsing resume...")
             else: log.info("Resume hash failed, parsing resume...")
             structured_resume = None # Ensure it's None if not loaded

        if not structured_resume: # If cache load failed or skipped
            structured_resume = await load_and_extract_resume(args.resume, analyzer)
            if structured_resume and cache_filepath: # Save to cache if parsing succeeded
                try:
                    with open(cache_filepath, 'w', encoding='utf-8') as f: f.write(structured_resume.model_dump_json(indent=2))
                    log.info(f"Saved structured resume data to cache: {cache_filepath}")
                except Exception as cache_e: log.warning(f"Failed to save resume cache: {cache_e}", exc_info=True)
        # --- End Resume Caching Logic ---

        if not structured_resume: raise RuntimeError("Failed to load and extract structured data from resume.")

        # --- Step 4: Analyze Jobs ---
        log.info("--- Stage 4: Analyzing Job Suitability (Async) ---")
        analyzed_results = await analyze_jobs(analyzer, structured_resume, jobs_list)
        if not analyzed_results: log.warning("Analysis step produced no results.")

        # --- Step 5: Apply Filters, Sort, and Save ---
        log.info("--- Stage 5: Filtering, Sorting, and Saving Results ---")
        filter_args_dict = {}
        # Populate filter_args_dict including ALL filters
        if args.min_salary is not None: filter_args_dict['salary_min'] = args.min_salary
        if args.max_salary is not None: filter_args_dict['salary_max'] = args.max_salary
        if args.filter_work_models: filter_args_dict['work_models'] = [wm.strip().lower() for wm in args.filter_work_models.split(',')]
        if args.filter_job_types: filter_args_dict['job_types'] = [jt.strip().lower() for jt in args.filter_job_types.split(',')]
        if args.filter_companies: filter_args_dict['filter_companies'] = [c.strip().lower() for c in args.filter_companies.split(',')]
        if args.exclude_companies: filter_args_dict['exclude_companies'] = [c.strip().lower() for c in args.exclude_companies.split(',')]
        if args.filter_title_keywords: filter_args_dict['filter_title_keywords'] = [k.strip().lower() for k in args.filter_title_keywords.split(',')]
        if args.filter_date_after: filter_args_dict['filter_date_after'] = args.filter_date_after
        if args.filter_date_before: filter_args_dict['filter_date_before'] = args.filter_date_before
        filter_args_dict['min_score'] = args.min_score # Pass min_score in dict for apply_filters_sort_and_save
        if args.filter_remote_country: filter_args_dict['filter_remote_country'] = args.filter_remote_country.strip()
        if args.filter_proximity_location:
             filter_args_dict['filter_proximity_location'] = args.filter_proximity_location.strip()
             filter_args_dict['filter_proximity_range'] = args.filter_proximity_range
             filter_args_dict['filter_proximity_models'] = [pm.strip().lower() for pm in args.filter_proximity_models.split(',')]

        final_results_list_dict = apply_filters_sort_and_save(
            analyzed_results, args.analysis_output, filter_args_dict )

        # --- Step 6: Print Summary Table ---
        log.info("--- Stage 6: Final Summary ---")
        print_summary_table(final_results_list_dict, top_n=10)

        log.info(f"[bold green]Pipeline Run Finished Successfully[/bold green]" if RICH_AVAILABLE else "Pipeline Run Finished Successfully")

    finally:
        # Ensure async client is closed even if errors occurred during logic
        if analyzer and hasattr(analyzer, 'close'):
             await analyzer.close()


# --- Main Entry Point ---
def main():
    """Parses args and runs the async pipeline logic."""
    parser = argparse.ArgumentParser(
        description="Run Job Scraping (via JobSpy) and GenAI Analysis Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    # --- Argument Groups ---
    scrape_group = parser.add_argument_group('Scraping Options (JobSpy)')
    scrape_group.add_argument("--search", required=True, help="Job title, keywords, or company.")
    scrape_group.add_argument("--location", default=None, help="Primary location for scraping. Overridden if --filter-remote-country.")
    scrape_group.add_argument("--sites", default=",".join(config.CFG['scraping']['default_sites']), help="Comma-separated sites.")
    scrape_group.add_argument("--results", type=int, default=config.CFG['scraping']['default_results_limit'], help="Approx total jobs per site.")
    scrape_group.add_argument("--hours-old", type=int, default=config.CFG['scraping']['default_hours_old'], help="Max job age in hours (0=disable).")
    scrape_group.add_argument("--country-indeed", default=config.CFG['scraping']['default_country_indeed'], help="Country for Indeed search.")
    scrape_group.add_argument("--proxies", help="Comma-separated proxies.")
    scrape_group.add_argument("--offset", type=int, default=0, help="Search results offset.")
    scrape_group.add_argument("--scraped-jobs-file", default=str(config.DEFAULT_SCRAPED_JSON), help="Intermediate file for scraped jobs.")

    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument("--resume", required=True, help="Path to the resume file.")
    analysis_group.add_argument("--analysis-output", default=str(config.DEFAULT_ANALYSIS_JSON), help="Final analysis output JSON.")
    analysis_group.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")
    analysis_group.add_argument("--force-resume-reparse", action="store_true", help="Ignore cached resume data and re-parse.")

    filter_group = parser.add_argument_group('Filtering Options (Applied After Analysis)')
    filter_group.add_argument("--min-salary", type=int, help="Minimum desired annual salary.")
    filter_group.add_argument("--max-salary", type=int, help="Maximum desired annual salary.")
    filter_group.add_argument("--filter-work-models", help="Standard work models (e.g., 'Remote,Hybrid').")
    filter_group.add_argument("--filter-job-types", help="Comma-separated job types (e.g., 'Full-time')")
    filter_group.add_argument("--filter-companies", help="Include ONLY these companies (comma-sep, case-insensitive).")
    filter_group.add_argument("--exclude-companies", help="EXCLUDE these companies (comma-sep, case-insensitive).")
    filter_group.add_argument("--filter-title-keywords", help="Require ANY of these keywords in title (comma-sep, case-insensitive).")
    filter_group.add_argument("--filter-date-after", help="Include jobs posted ON or AFTER YYYY-MM-DD.")
    filter_group.add_argument("--filter-date-before", help="Include jobs posted ON or BEFORE YYYY-MM-DD.")
    filter_group.add_argument("--min-score", type=int, default=0, help="Minimum suitability score filter (0-100). Default 0.")

    adv_loc_group = parser.add_argument_group('Advanced Location Filtering')
    adv_loc_group.add_argument("--filter-remote-country", help="Filter REMOTE jobs within specific country.")
    adv_loc_group.add_argument("--filter-proximity-location", help="Reference location for proximity filtering.")
    adv_loc_group.add_argument("--filter-proximity-range", type=float, help="Distance in miles for proximity.")
    adv_loc_group.add_argument("--filter-proximity-models", default="Hybrid,On-site", help="Work models for proximity.")

    args = parser.parse_args()

    # --- Setup Logging Level Based on Args AFTER parsing ---
    final_log_level = logging.DEBUG if args.verbose else config.LOG_LEVEL.upper()
    try:
        logging.getLogger().setLevel(final_log_level) # Set root logger level
        log.info(f"Log level set to: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")
    except ValueError:
         log.error(f"Invalid log level configured: {final_log_level}. Using INFO.")
         logging.getLogger().setLevel(logging.INFO)


    # --- Run the main async logic ---
    try:
        if sys.version_info >= (3, 7):
            asyncio.run(async_pipeline_logic(args))
        else:
            # Fallback needed for Python 3.6, though 3.9+ is recommended
            log.warning("Using legacy asyncio event loop runner (Python < 3.7).")
            loop = asyncio.get_event_loop()
            loop.run_until_complete(async_pipeline_logic(args))
    except KeyboardInterrupt:
        print(); log.warning("[yellow]Execution interrupted by user (Ctrl+C).[/yellow]" if RICH_AVAILABLE else "Execution interrupted.")
        # Allow finally block in async_pipeline_logic to run if possible
        sys.exit(130)
    except (RuntimeError, ValueError, ConnectionError, FileNotFoundError) as e:
         # Catch specific critical errors raised by the pipeline
         log.critical(f"Pipeline failed with critical error: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
         # Catch-all for truly unexpected errors
         log.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
         sys.exit(1)

if __name__ == "__main__":
    main()