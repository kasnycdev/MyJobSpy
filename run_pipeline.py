import argparse
import logging
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# Use the jobspy library for scraping
try:
    import jobspy
    from jobspy import scrape_jobs # Explicit import
except ImportError:
    # This error is critical, exit immediately before logging is fully set up
    print("CRITICAL ERROR: 'jobspy' library not found. Please install it via requirements.txt")
    sys.exit(1)

# Import analysis components from main_matcher
try:
    from main_matcher import load_and_extract_resume, analyze_jobs, apply_filters_sort_and_save
    from analysis.analyzer import ResumeAnalyzer # Use the correct class name
except ImportError as e:
    # This error is also critical
    print(f"CRITICAL ERROR: Could not import analysis functions from main_matcher or analyzer: {e}")
    print("Ensure main_matcher.py and analysis/analyzer.py are in the correct path and define the expected classes/functions.")
    sys.exit(1)

import config # Central configuration

# Rich for UX
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not found. Falling back to basic logging/output.")

# Setup logging using Rich if available, otherwise basic
log = logging.getLogger(__name__) # Get logger instance first
if RICH_AVAILABLE:
    logging.basicConfig(
        level=config.LOG_LEVEL, # Set level initially
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
    )
    console = Console()
else:
     logging.basicConfig(
        level=config.LOG_LEVEL, # Set level initially
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
# Suppress noisy libraries AFTER basicConfig is called
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING) # Often noisy too


# --- scrape_jobs_with_jobspy function (add specific try/except) ---
def scrape_jobs_with_jobspy(
    search_terms: str,
    location: str,
    sites: list[str],
    results_wanted: int,
    hours_old: int,
    country_indeed: str,
    proxies: Optional[list[str]] = None,
    offset: int = 0
    ) -> Optional[pd.DataFrame]:
    """Uses the jobspy library to scrape jobs, with better logging and error handling."""
    log.info(f"[bold blue]Starting job scraping via JobSpy...[/bold blue]" if RICH_AVAILABLE else "Starting job scraping via JobSpy...")
    log.info(f"Search: '[cyan]{search_terms}[/cyan]' | Location: '[cyan]{location}[/cyan]' | Sites: {sites}" if RICH_AVAILABLE else f"Search: '{search_terms}' | Location: '{location}' | Sites: {sites}")
    log.info(f"Params: Results ≈{results_wanted}, Max Age={hours_old}h, Indeed Country='{country_indeed}', Offset={offset}")
    if proxies: log.info(f"Using {len(proxies)} proxies.")

    try:
        jobs_df = scrape_jobs(
            site_name=sites, search_term=search_terms, location=location, results_wanted=results_wanted,
            hours_old=hours_old, country_indeed=country_indeed, proxies=proxies, offset=offset,
            verbose=1, # Use jobspy's verbose logging
            description_format="markdown", linkedin_fetch_description=True, linkedin_company_ids=None
        )

        if jobs_df is None or jobs_df.empty:
            log.warning("Jobspy scraping returned no results or failed silently.")
            return None
        else:
            log.info(f"Jobspy scraping successful. Found {len(jobs_df)} jobs.")
            log.debug(f"DataFrame columns: {jobs_df.columns.tolist()}")
            required_cols = ['title', 'company', 'location', 'description', 'job_url']
            missing_cols = [col for col in required_cols if col not in jobs_df.columns]
            if missing_cols: log.warning(f"Jobspy output DataFrame is missing expected columns: {missing_cols}")
            return jobs_df

    except ImportError as ie:
         # This case should ideally be caught at the top level import
         log.critical(f"Import error during scraping function call: {ie}. Ensure all jobspy dependencies are installed.")
         return None
    # Add specific exceptions JobSpy might raise if known, otherwise catch broadly
    except Exception as e:
        # Log the specific error encountered during scraping
        log.error(f"An unexpected error occurred during jobspy scraping execution: {e}", exc_info=True)
        return None # Indicate failure


# --- convert_and_save_scraped function (add specific try/except) ---
def convert_and_save_scraped(jobs_df: pd.DataFrame, output_path: str) -> List[Dict[str, Any]]:
    """Converts DataFrame to list of dicts and saves to JSON, handling date objects."""
    log.info(f"Converting DataFrame to list and saving to {output_path}")
    try:
        # --- Data Processing ---
        rename_map = {'job_url': 'url','job_type': 'employment_type','salary': 'salary_text','benefits': 'benefits_text'}
        actual_rename_map = {k: v for k, v in rename_map.items() if k in jobs_df.columns}
        if actual_rename_map: jobs_df = jobs_df.rename(columns=actual_rename_map); log.debug(f"Renamed DataFrame columns: {actual_rename_map}")
        possible_date_columns = ['date_posted', 'posted_date', 'date']
        for col in possible_date_columns:
            if col in jobs_df.columns:
                log.debug(f"Processing potential date column '{col}' for JSON serialization.")
                if pd.api.types.is_datetime64_any_dtype(jobs_df[col]) or jobs_df[col].dtype == 'object':
                    try:
                        jobs_df[col] = pd.to_datetime(jobs_df[col], errors='coerce')
                        jobs_df[col] = jobs_df[col].dt.strftime('%Y-%m-%d')
                        log.debug(f"Successfully converted column '{col}' dates to string format.")
                    except Exception as date_err:
                        log.warning(f"Could not fully process date column '{col}'. Error: {date_err}. Attempting direct string conversion.")
                        jobs_df[col] = jobs_df[col].astype(str)
                else: log.debug(f"Column '{col}' is not datetime or object type ({jobs_df[col].dtype}), skipping date conversion.")
        essential_cols = ['title','company','location','description','url','salary_text','employment_type','benefits_text','skills','date_posted']
        for col in essential_cols:
            if col not in jobs_df.columns: log.warning(f"Column '{col}' missing from scraped data, adding as empty."); jobs_df[col] = ''
        jobs_df = jobs_df.fillna(''); log.debug("Filled NA/NaN values with empty strings.")
        jobs_list = jobs_df.to_dict('records'); log.debug(f"Converted DataFrame to list of {len(jobs_list)} dictionaries.")

        # --- File Saving ---
        output_dir = os.path.dirname(output_path);
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(jobs_list, f, indent=4)
        log.info(f"Saved {len(jobs_list)} scraped jobs to {output_path}"); return jobs_list

    except TypeError as json_err:
         log.error(f"JSON Serialization Error during conversion/saving: {json_err}", exc_info=True)
         # Try to log problematic record (if jobs_list was created)
         if 'jobs_list' in locals():
             for i, record in enumerate(jobs_list):
                  try: json.dumps(record)
                  except TypeError:
                       log.error(f"Problematic record at index {i}: {record}")
                       for k, v in record.items(): log.error(f"  Field '{k}': Type={type(v)}, Value='{str(v)[:100]}...'")
                       break
         return [] # Return empty list on save failure
    except (IOError, OSError) as file_err:
        log.error(f"File system error saving scraped jobs to {output_path}: {file_err}", exc_info=True)
        return []
    except Exception as e:
        # Catch unexpected errors during the conversion/saving process
        log.error(f"Unexpected error converting/saving scraped jobs: {e}", exc_info=True); return []


# --- print_summary_table function (add basic check) ---
def print_summary_table(results_json: Optional[List[Dict[str, Any]]], top_n: int = 10):
    """Prints a summary table of top results using Rich."""
    if not results_json: # Check if list is None or empty
        if RICH_AVAILABLE: console.print("[yellow]No analysis results to summarize.[/yellow]")
        else: log.warning("No analysis results to summarize.")
        return

    # Existing logic using rich... make sure console is defined or handled if rich is not available
    if not RICH_AVAILABLE:
         log.info("Top results (rich table unavailable):")
         # Basic print fallback
         for i, result in enumerate(results_json[:top_n]):
             analysis = result.get('analysis', {})
             original = result.get('original_job_data', {})
             score = analysis.get('suitability_score', -1)
             if score is None or score == 0: continue
             log.info(f"  {i+1}. Score: {score}% | Title: {original.get('title', 'N/A')} | Company: {original.get('company', 'N/A')} | URL: {original.get('url', '#')}")
         return

    # Rich table logic (keep as before)
    table = Table(title=f"Top {min(top_n, len(results_json))} Job Matches", show_header=True, header_style="bold magenta", show_lines=False)
    table.add_column("Score", style="dim", width=6, justify="right"); table.add_column("Title", style="bold", min_width=20); table.add_column("Company"); table.add_column("Location"); table.add_column("URL", overflow="fold", style="cyan")
    count = 0
    for result in results_json:
        if count >= top_n: break
        analysis = result.get('analysis', {}); original = result.get('original_job_data', {})
        score = analysis.get('suitability_score', -1)
        if score is None or score == 0: continue
        score_str = f"{score}%"
        table.add_row(score_str, original.get('title', 'N/A'), original.get('company', 'N/A'), original.get('location', 'N/A'), original.get('url', '#'))
        count += 1
    if count == 0: console.print("[yellow]No successfully analyzed jobs with score > 0 to display in summary.[/yellow]")
    else: console.print(table)


# --- Main Execution ---
def run_pipeline():
    # --- Argument Parsing (remains unchanged) ---
    parser = argparse.ArgumentParser(
        description="Run Job Scraping (via JobSpy) and GenAI Analysis Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    scrape_group = parser.add_argument_group('Scraping Options (JobSpy)')
    scrape_group.add_argument("--search", required=True, help="Job title, keywords, or company.")
    scrape_group.add_argument("--location", default=None, help="Primary location for scraping. Overridden if --filter-remote-country.")
    scrape_group.add_argument("--sites", default=",".join(config.DEFAULT_SCRAPE_SITES), help="Comma-separated sites.")
    scrape_group.add_argument("--results", type=int, default=config.DEFAULT_RESULTS_LIMIT, help="Approx total jobs per site.")
    scrape_group.add_argument("--hours-old", type=int, default=config.DEFAULT_HOURS_OLD, help="Max job age in hours (0=disable).")
    scrape_group.add_argument("--country-indeed", default=config.DEFAULT_COUNTRY_INDEED, help="Country for Indeed search.")
    scrape_group.add_argument("--proxies", help="Comma-separated proxies.")
    scrape_group.add_argument("--offset", type=int, default=0, help="Search results offset.")
    scrape_group.add_argument("--scraped-jobs-file", default=config.DEFAULT_SCRAPED_JSON, help="Intermediate file for scraped jobs.")
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument("--resume", required=True, help="Path to the resume file.")
    analysis_group.add_argument("--analysis-output", default=config.DEFAULT_ANALYSIS_JSON, help="Final analysis output JSON.")
    analysis_group.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")
    filter_group = parser.add_argument_group('Filtering Options (Applied After Analysis)')
    filter_group.add_argument("--min-salary", type=int, help="Minimum desired annual salary.")
    filter_group.add_argument("--max-salary", type=int, help="Maximum desired annual salary.")
    filter_group.add_argument("--filter-work-models", help="Standard work models (e.g., 'Remote,Hybrid').")
    filter_group.add_argument("--filter-job-types", help="Comma-separated job types (e.g., 'Full-time')")
    adv_loc_group = parser.add_argument_group('Advanced Location Filtering')
    adv_loc_group.add_argument("--filter-remote-country", help="Filter REMOTE jobs within specific country.")
    adv_loc_group.add_argument("--filter-proximity-location", help="Reference location for proximity filtering.")
    adv_loc_group.add_argument("--filter-proximity-range", type=float, help="Distance in miles for proximity.")
    adv_loc_group.add_argument("--filter-proximity-models", default="Hybrid,On-site", help="Work models for proximity.")
    args = parser.parse_args()

    # --- Setup Logging Level ---
    log_level = logging.DEBUG if args.verbose else config.LOG_LEVEL
    logging.getLogger().setLevel(log_level) # Set root logger level *after* parsing args

    log.info(f"[bold green]Starting Pipeline Run[/bold green] ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})" if RICH_AVAILABLE else f"Starting Pipeline Run ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    # --- Main Pipeline Logic Wrapped in try...except ---
    try:
        config.ensure_output_dir()
        # --- Validate argument combinations ---
        # ... (validation logic remains unchanged) ...
        if not args.location and not args.filter_remote_country and not args.filter_proximity_location: parser.error("Ambiguous location: Specify --location OR --filter-remote-country OR --filter-proximity-location.")
        if args.filter_proximity_location and args.filter_remote_country: parser.error("Conflicting filters: Cannot use --filter-proximity-location and --filter-remote-country.")
        if args.filter_proximity_location and args.filter_proximity_range is None: parser.error("--filter-proximity-range is required with --filter-proximity-location.")
        if args.filter_proximity_range is not None and not args.filter_proximity_location: parser.error("--filter-proximity-location is required with --filter-proximity-range.")

        # --- Determine scrape location ---
        # ... (scrape location logic remains unchanged) ...
        scrape_location = None
        if args.filter_remote_country: scrape_location = args.filter_remote_country.strip(); log.info(f"Using country '{scrape_location}' as primary scrape location (from --filter-remote-country).")
        elif args.filter_proximity_location: scrape_location = args.filter_proximity_location.strip(); log.info(f"Using proximity target '{scrape_location}' as primary scrape location.")
        elif args.location: scrape_location = args.location; log.info(f"Using provided --location '{scrape_location}' as primary scrape location.")

        # --- Step 1: Scrape Jobs ---
        log.info("--- Stage 1: Scraping Jobs ---")
        scraper_sites = [site.strip().lower() for site in args.sites.split(',')]
        proxy_list = [p.strip() for p in args.proxies.split(',')] if args.proxies else None
        jobs_df = scrape_jobs_with_jobspy(
            search_terms=args.search, location=scrape_location, sites=scraper_sites, results_wanted=args.results,
            hours_old=args.hours_old, country_indeed=args.country_indeed, proxies=proxy_list, offset=args.offset )

        if jobs_df is None or jobs_df.empty:
            log.warning("Scraping yielded no results. Pipeline halted.")
            analysis_output_dir = os.path.dirname(args.analysis_output);
            if analysis_output_dir: os.makedirs(analysis_output_dir, exist_ok=True)
            with open(args.analysis_output, 'w', encoding='utf-8') as f: json.dump([], f)
            log.info(f"Empty analysis results file created at {args.analysis_output}"); sys.exit(0)

        # --- Step 2: Convert and Save Scraped Data ---
        log.info("--- Stage 2: Processing Scraped Data ---")
        jobs_list = convert_and_save_scraped(jobs_df, args.scraped_jobs_file)
        if not jobs_list: log.error("Failed to convert or save scraped data. Pipeline halted."); sys.exit(1)

        # --- Step 3: Initialize Analyzer and Load Resume ---
        log.info("--- Stage 3: Initializing Analyzer and Processing Resume ---")
        analyzer: Optional[ResumeAnalyzer] = None # Define upfront
        try:
            analyzer = ResumeAnalyzer() # Use the correctly imported class name
        except Exception as e:
            # Catch specific errors if possible (e.g., ConnectionError from _check_connection)
            log.critical(f"Failed to initialize ResumeAnalyzer: {e}. Cannot proceed with analysis.", exc_info=True)
            sys.exit(1)

        structured_resume: Optional[Any] = None # Define upfront
        try:
            structured_resume = load_and_extract_resume(args.resume, analyzer)
        except FileNotFoundError as e:
            log.critical(f"Resume file not found: {args.resume}. Error: {e}", exc_info=True)
            sys.exit(1)
        except Exception as e: # Catch other resume processing errors
             log.critical(f"Failed during resume loading/extraction: {e}", exc_info=True)
             sys.exit(1)

        if not structured_resume:
            # This case is hit if load_and_extract_resume returns None (e.g., parsing/LLM error)
            log.critical("Failed to load and extract structured data from resume. Pipeline halted.")
            sys.exit(1)

        # --- Step 4: Analyze Jobs ---
        log.info("--- Stage 4: Analyzing Job Suitability ---")
        analyzed_results: List[Any] = [] # Define upfront
        try:
            # Pass the confirmed non-None structured_resume
            analyzed_results = analyze_jobs(analyzer, structured_resume, jobs_list)
            if not analyzed_results: log.warning("Analysis step produced no results (all jobs might have failed analysis).")
        except Exception as e:
            # Catch errors during the analysis loop itself
            log.critical(f"Critical error during job analysis phase: {e}", exc_info=True)
            sys.exit(1) # Exit if the whole analysis phase fails

        # --- Step 5: Apply Filters, Sort, and Save ---
        log.info("--- Stage 5: Filtering, Sorting, and Saving Results ---")
        final_results_list_dict: List[Dict[str, Any]] = [] # Define upfront
        try:
            filter_args_dict = {}
            if args.min_salary is not None: filter_args_dict['salary_min'] = args.min_salary
            if args.max_salary is not None: filter_args_dict['salary_max'] = args.max_salary
            if args.filter_work_models: filter_args_dict['work_models'] = [wm.strip().lower() for wm in args.filter_work_models.split(',')]
            if args.filter_job_types: filter_args_dict['job_types'] = [jt.strip().lower() for jt in args.filter_job_types.split(',')]
            if args.filter_remote_country: filter_args_dict['filter_remote_country'] = args.filter_remote_country.strip()
            if args.filter_proximity_location:
                 filter_args_dict['filter_proximity_location'] = args.filter_proximity_location.strip()
                 filter_args_dict['filter_proximity_range'] = args.filter_proximity_range
                 filter_args_dict['filter_proximity_models'] = [pm.strip().lower() for pm in args.filter_proximity_models.split(',')]

            final_results_list_dict = apply_filters_sort_and_save(
                analyzed_results, args.analysis_output, filter_args_dict )
        except Exception as e:
            # Catch errors during filtering/sorting/saving
            log.critical(f"Critical error during filtering/saving phase: {e}", exc_info=True)
            sys.exit(1)

        # --- Step 6: Print Summary Table ---
        log.info("--- Stage 6: Final Summary ---")
        try:
             print_summary_table(final_results_list_dict, top_n=10)
        except Exception as e:
             log.error(f"Error printing summary table: {e}", exc_info=True) # Don't exit for summary error

        log.info(f"[bold green]Pipeline Run Finished Successfully[/bold green]" if RICH_AVAILABLE else "Pipeline Run Finished Successfully")

    except KeyboardInterrupt:
        print(); log.warning("[yellow]Pipeline execution interrupted by user (Ctrl+C). Exiting gracefully.[/yellow]" if RICH_AVAILABLE else "Pipeline interrupted by user (Ctrl+C). Exiting.")
        sys.exit(130)
    except FileNotFoundError as e:
         # Catch file errors related to resume/jobs file earlier now, but keep as fallback
         log.critical(f"File not found error: {e}", exc_info=True)
         sys.exit(1)
    except ConnectionError as e:
         # Catch connection errors specifically (e.g., from Ollama or Geopy)
         log.critical(f"Network/Connection error during pipeline execution: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
         # Catch-all for truly unexpected errors during the main flow
         log.critical(f"An unexpected critical error occurred during pipeline execution: {e}", exc_info=True)
         sys.exit(1)

if __name__ == "__main__":
    run_pipeline()