# run_pipeline.py
import argparse
import logging
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# ... (other imports: jobspy, main_matcher functions, config, rich) ...
import config # Central configuration
from rich.console import Console
from rich.logging import RichHandler
# ...

log = logging.getLogger(__name__)
console = Console()

# --- scrape_jobs_with_jobspy function remains unchanged ---
# --- convert_and_save_scraped function remains unchanged ---
# --- print_summary_table function remains unchanged ---

# --- Main Execution ---
def run_pipeline():
    parser = argparse.ArgumentParser(
        description="Run Job Scraping (via JobSpy) and GenAI Analysis Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Scraping Arguments (Unchanged) ---
    scrape_group = parser.add_argument_group('Scraping Options (JobSpy)')
    scrape_group.add_argument("--search", required=True, help="Job title, keywords, or company.")
    # ... (other scraping args: location, sites, results, hours-old, etc.) ...
    scrape_group.add_argument("--scraped-jobs-file", default=config.DEFAULT_SCRAPED_JSON, help="Intermediate file for scraped jobs.")

    # --- Analysis Arguments (Unchanged) ---
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument("--resume", required=True, help="Path to the resume file.")
    analysis_group.add_argument("--analysis-output", default=config.DEFAULT_ANALYSIS_JSON, help="Final analysis output JSON.")
    analysis_group.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")

    # --- Filtering Arguments ---
    filter_group = parser.add_argument_group('Filtering Options (Applied After Analysis)')
    # Standard Filters (Unchanged)
    filter_group.add_argument("--min-salary", type=int, help="Minimum desired annual salary.")
    filter_group.add_argument("--max-salary", type=int, help="Maximum desired annual salary.")
    filter_group.add_argument("--filter-locations", help="DEPRECATED (use proximity/remote filters). Comma-separated locations (e.g., 'New York,Remote')") # Mark old one as deprecated if desired
    filter_group.add_argument("--filter-work-models", help="Standard work models (e.g., 'Remote,Hybrid'). Applied *in addition* to advanced filters.")
    filter_group.add_argument("--filter-job-types", help="Comma-separated job types (e.g., 'Full-time')")

    # --- NEW Advanced Location Filters ---
    adv_loc_group = parser.add_argument_group('Advanced Location Filtering')
    adv_loc_group.add_argument("--filter-remote-country",
                               help="Filter for REMOTE jobs within a specific country (e.g., 'USA', 'Canada', 'United Kingdom'). Requires geocoding job locations.")
    adv_loc_group.add_argument("--filter-proximity-location",
                               help="Reference location (city, state, address) for proximity filtering (e.g., 'New York, NY', 'London, UK').")
    adv_loc_group.add_argument("--filter-proximity-range", type=float,
                               help="Distance in miles (e.g., 50.0) to filter jobs around --filter-proximity-location.")
    adv_loc_group.add_argument("--filter-proximity-models", default="Hybrid,On-site",
                               help="Work models allowed for proximity filtering (e.g., 'Hybrid,On-site' or 'On-site').")
    # --- End New Args ---

    args = parser.parse_args()

    # --- Setup Logging Level (Unchanged) ---
    log_level = logging.DEBUG if args.verbose else config.LOG_LEVEL
    logging.getLogger().setLevel(log_level)
    log.info(f"[bold green]Starting Pipeline Run[/bold green] ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    config.ensure_output_dir()

    # --- Step 1: Scrape Jobs (Unchanged) ---
    # ... (scraper setup and call to scrape_jobs_with_jobspy) ...
    jobs_df = scrape_jobs_with_jobspy(...)
    if jobs_df is None or jobs_df.empty:
        # ... (handle no results) ...
        sys.exit(0)

    # --- Step 2: Convert and Save Scraped Data (Unchanged) ---
    jobs_list = convert_and_save_scraped(jobs_df, args.scraped_jobs_file)
    if not jobs_list:
        # ... (handle save failure) ...
        sys.exit(1)

    # --- Step 3: Initialize Analyzer and Load Resume (Unchanged) ---
    try:
        analyzer = ResumeAnalyzer()
    except Exception as e:
        log.critical(f"Failed to initialize ResumeAnalyzer: {e}.", exc_info=True)
        sys.exit(1)

    structured_resume = load_and_extract_resume(args.resume, analyzer)
    if not structured_resume:
        log.critical("Failed to load and extract data from resume.", exc_info=True)
        sys.exit(1)

    # --- Step 4: Analyze Jobs (Unchanged) ---
    analyzed_results = analyze_jobs(analyzer, structured_resume, jobs_list)
    if not analyzed_results:
         log.warning("Analysis step produced no results.")

    # --- Step 5: Apply Filters, Sort, and Save ---
    # --- UPDATED Filter Args Population ---
    filter_args_dict = {}
    # Standard filters
    if args.min_salary is not None: filter_args_dict['salary_min'] = args.min_salary
    if args.max_salary is not None: filter_args_dict['salary_max'] = args.max_salary
    # Note: We keep --filter-work-models and --filter-job-types as they can apply alongside advanced location filters
    if args.filter_work_models: filter_args_dict['work_models'] = [wm.strip() for wm in args.filter_work_models.split(',')]
    if args.filter_job_types: filter_args_dict['job_types'] = [jt.strip() for jt in args.filter_job_types.split(',')]

    # Advanced Location Filters
    if args.filter_remote_country: filter_args_dict['filter_remote_country'] = args.filter_remote_country.strip()
    if args.filter_proximity_location:
        if args.filter_proximity_range is None:
             log.error("Usage Error: --filter-proximity-range is required when using --filter-proximity-location.")
             sys.exit(1)
        filter_args_dict['filter_proximity_location'] = args.filter_proximity_location.strip()
        filter_args_dict['filter_proximity_range'] = args.filter_proximity_range
        filter_args_dict['filter_proximity_models'] = [pm.strip().lower() for pm in args.filter_proximity_models.split(',')] if args.filter_proximity_models else ['hybrid', 'on-site']
    elif args.filter_proximity_range is not None:
         log.error("Usage Error: --filter-proximity-location is required when using --filter-proximity-range.")
         sys.exit(1)
    # --- End Filter Args Population ---

    final_results_list_dict = apply_filters_sort_and_save(
        analyzed_results,
        args.analysis_output,
        filter_args_dict # Pass the dictionary containing all active filters
    )

    # --- Step 6: Print Summary Table (Unchanged) ---
    log.info("[bold blue]Pipeline Summary:[/bold blue]")
    print_summary_table(final_results_list_dict, top_n=10)

    log.info(f"[bold green]Pipeline Run Finished[/bold green]")

if __name__ == "__main__":
    run_pipeline()