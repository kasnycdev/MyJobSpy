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
    print("CRITICAL ERROR: 'jobspy' library not found. Please install it via requirements.txt")
    sys.exit(1)

# Import analysis components from main_matcher
try:
    from main_matcher import load_and_extract_resume, analyze_jobs, apply_filters_sort_and_save
    from analysis.analyzer import ResumeAnalyzer # Need analyzer class
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import analysis functions from main_matcher or analyzer: {e}")
    print("Ensure main_matcher.py and analysis/analyzer.py are in the correct path.")
    sys.exit(1)

import config # Central configuration

# Rich for UX
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Setup logging using Rich
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)] # show_path=False for cleaner logs
)
logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx verbosity
log = logging.getLogger(__name__) # Use project logger
console = Console()


# --- scrape_jobs_with_jobspy function remains unchanged ---
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
    """Uses the jobspy library to scrape jobs, with better logging."""
    log.info(f"[bold blue]Starting job scraping via JobSpy...[/bold blue]")
    log.info(f"Search: '[cyan]{search_terms}[/cyan]' | Location: '[cyan]{location}[/cyan]' | Sites: {sites}")
    log.info(f"Params: Results ≈{results_wanted}, Max Age={hours_old}h, Indeed Country='{country_indeed}', Offset={offset}")
    if proxies:
        log.info(f"Using {len(proxies)} proxies.")

    try:
        jobs_df = scrape_jobs(
            site_name=sites,
            search_term=search_terms,
            location=location,
            results_wanted=results_wanted,
            hours_old=hours_old,
            country_indeed=country_indeed,
            proxies=proxies,
            offset=offset,
            verbose=1,
            linkedin_fetch_description: True,
            description_format="markdown"
        )

        if jobs_df is None or jobs_df.empty:
            log.warning("Jobspy scraping returned no results or failed.")
            return None
        else:
            log.info(f"Jobspy scraping successful. Found {len(jobs_df)} jobs.")
            log.debug(f"DataFrame columns: {jobs_df.columns.tolist()}")
            required_cols = ['title', 'company', 'location', 'description', 'job_url']
            missing_cols = [col for col in required_cols if col not in jobs_df.columns]
            if missing_cols:
                 log.warning(f"Jobspy output DataFrame is missing expected columns: {missing_cols}")
            return jobs_df

    except ImportError as ie:
         log.critical(f"Import error during scraping: {ie}. Ensure all jobspy dependencies are installed.")
         return None
    except Exception as e:
        log.error(f"An error occurred during jobspy scraping: {e}", exc_info=True)
        return None


# --- convert_and_save_scraped function remains unchanged ---
def convert_and_save_scraped(jobs_df: pd.DataFrame, output_path: str) -> List[Dict[str, Any]]:
    """Converts DataFrame to list of dicts and saves to JSON, handling date objects."""
    log.info(f"Converting DataFrame to list and saving to {output_path}")

    rename_map = {
        'job_url': 'url',
        'job_type': 'employment_type',
        'salary': 'salary_text',
        'benefits': 'benefits_text'
    }
    actual_rename_map = {k: v for k, v in rename_map.items() if k in jobs_df.columns}
    if actual_rename_map:
        jobs_df = jobs_df.rename(columns=actual_rename_map)
        log.debug(f"Renamed DataFrame columns: {actual_rename_map}")

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
            else:
                 log.debug(f"Column '{col}' is not datetime or object type ({jobs_df[col].dtype}), skipping date conversion.")

    essential_cols = ['title', 'company', 'location', 'description', 'url',
                      'salary_text', 'employment_type', 'benefits_text', 'skills',
                      'date_posted']
    for col in essential_cols:
         if col not in jobs_df.columns:
              log.warning(f"Column '{col}' missing from scraped data, adding as empty.")
              jobs_df[col] = ''

    jobs_df = jobs_df.fillna('')
    log.debug("Filled NA/NaN values with empty strings.")
    jobs_list = jobs_df.to_dict('records')
    log.debug(f"Converted DataFrame to list of {len(jobs_list)} dictionaries.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jobs_list, f, indent=4)
        log.info(f"Saved {len(jobs_list)} scraped jobs to {output_path}")
        return jobs_list
    except TypeError as json_err:
         log.error(f"JSON Serialization Error even after date conversion: {json_err}", exc_info=True)
         for i, record in enumerate(jobs_list):
              try: json.dumps(record)
              except TypeError:
                   log.error(f"Problematic record at index {i}: {record}")
                   for k, v in record.items(): log.error(f"  Field '{k}': Type={type(v)}, Value='{str(v)[:100]}...'")
                   break
         return []
    except Exception as e:
        log.error(f"Error saving scraped jobs to JSON file {output_path}: {e}", exc_info=True)
        return []


# --- print_summary_table function remains unchanged ---
def print_summary_table(results_json: List[Dict[str, Any]], top_n: int = 10):
    """Prints a summary table of top results using Rich."""
    if not results_json:
        console.print("[yellow]No analysis results to summarize.[/yellow]")
        return

    table = Table(title=f"Top {min(top_n, len(results_json))} Job Matches", show_header=True, header_style="bold magenta", show_lines=False)
    table.add_column("Score", style="dim", width=6, justify="right")
    table.add_column("Title", style="bold", min_width=20)
    table.add_column("Company")
    table.add_column("Location")
    table.add_column("URL", overflow="fold", style="cyan")

    count = 0
    for result in results_json:
        if count >= top_n: break
        analysis = result.get('analysis', {})
        original = result.get('original_job_data', {})
        score = analysis.get('suitability_score', -1)
        # Updated logic: Check if score is 0 (placeholder for failed analysis) or None (unexpected case)
        if score is None or score == 0: continue # Skip placeholder/failed entries

        score_str = f"{score}%"
        table.add_row(
            score_str,
            original.get('title', 'N/A'),
            original.get('company', 'N/A'),
            original.get('location', 'N/A'),
            original.get('url', '#')
        )
        count += 1

    if count == 0:
         console.print("[yellow]No successfully analyzed jobs with score > 0 to display in summary.[/yellow]")
    else:
         console.print(table)


# --- Main Execution ---
def run_pipeline():
    parser = argparse.ArgumentParser(
        description="Run Job Scraping (via JobSpy) and GenAI Analysis Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Scraping Arguments ---
    scrape_group = parser.add_argument_group('Scraping Options (JobSpy)')
    scrape_group.add_argument("--search", required=True, help="Job title, keywords, or company.")
    # Removed default="Remote" to encourage specifying location or using filters
    scrape_group.add_argument("--location", default=None, help="Primary location for scraping (e.g., 'New York', 'Canada'). Overridden if --filter-remote-country is used.")
    scrape_group.add_argument("--sites", default=",".join(config.DEFAULT_SCRAPE_SITES),
                              help="Comma-separated sites (e.g., linkedin,indeed,zip_recruiter). Check JobSpy docs.")
    scrape_group.add_argument("--results", type=int, default=config.DEFAULT_RESULTS_LIMIT, help="Approx total jobs to fetch per site.")
    scrape_group.add_argument("--hours-old", type=int, default=config.DEFAULT_HOURS_OLD, help="Max job age in hours (0=disable).")
    scrape_group.add_argument("--country-indeed", default=config.DEFAULT_COUNTRY_INDEED, help="Country for Indeed search ('usa', 'uk', etc.).")
    scrape_group.add_argument("--proxies", help="Comma-separated proxies ('http://user:pass@host:port,...').")
    scrape_group.add_argument("--offset", type=int, default=0, help="Search results offset.")
    scrape_group.add_argument("--scraped-jobs-file", default=config.DEFAULT_SCRAPED_JSON, help="Intermediate file for scraped jobs.")

    # --- Analysis Arguments ---
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument("--resume", required=True, help="Path to the resume file.")
    analysis_group.add_argument("--analysis-output", default=config.DEFAULT_ANALYSIS_JSON, help="Final analysis output JSON.")
    analysis_group.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")

    # --- Filtering Arguments ---
    filter_group = parser.add_argument_group('Filtering Options (Applied After Analysis)')
    filter_group.add_argument("--min-salary", type=int, help="Minimum desired annual salary.")
    filter_group.add_argument("--max-salary", type=int, help="Maximum desired annual salary.")
    filter_group.add_argument("--filter-work-models", help="Standard work models (e.g., 'Remote,Hybrid').")
    filter_group.add_argument("--filter-job-types", help="Comma-separated job types (e.g., 'Full-time')")

    # --- Advanced Location Filters ---
    adv_loc_group = parser.add_argument_group('Advanced Location Filtering')
    adv_loc_group.add_argument("--filter-remote-country",
                               help="Filter for REMOTE jobs within a specific country (e.g., 'USA'). If set, this country is used for the primary scrape location.")
    adv_loc_group.add_argument("--filter-proximity-location",
                               help="Reference location for proximity filtering (e.g., 'New York, NY').")
    adv_loc_group.add_argument("--filter-proximity-range", type=float,
                               help="Distance in miles for proximity filtering.")
    adv_loc_group.add_argument("--filter-proximity-models", default="Hybrid,On-site",
                               help="Work models for proximity filtering (default: 'Hybrid,On-site').")

    args = parser.parse_args()

    # --- Validate argument combinations ---
    if not args.location and not args.filter_remote_country and not args.filter_proximity_location:
         parser.error("Ambiguous location: Please specify --location OR --filter-remote-country OR --filter-proximity-location for scraping.")
    if args.filter_proximity_location and args.filter_remote_country:
         parser.error("Conflicting filters: Cannot use --filter-proximity-location and --filter-remote-country simultaneously.")


    # --- Setup Logging Level ---
    log_level = logging.DEBUG if args.verbose else config.LOG_LEVEL
    logging.getLogger().setLevel(log_level)
    log.info(f"[bold green]Starting Pipeline Run[/bold green] ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    config.ensure_output_dir()

    # --- Step 1: Scrape Jobs ---
    scraper_sites = [site.strip().lower() for site in args.sites.split(',')]
    proxy_list = [p.strip() for p in args.proxies.split(',')] if args.proxies else None

    # --- Determine the location to use for the SCRAPE ---
    scrape_location = None
    if args.filter_remote_country:
        # Use the country specified in the filter for the scrape
        scrape_location = args.filter_remote_country.strip()
        log.info(f"Using country '{scrape_location}' as primary scrape location (from --filter-remote-country).")
    elif args.filter_proximity_location:
         # Use proximity location for scrape if country filter isn't set
         scrape_location = args.filter_proximity_location.strip()
         log.info(f"Using proximity target '{scrape_location}' as primary scrape location.")
    elif args.location:
        # Use the explicitly provided --location if no advanced filters dictate location
        scrape_location = args.location
        log.info(f"Using provided --location '{scrape_location}' as primary scrape location.")
    else:
         # This case should be caught by argument validation above, but added for safety
         log.error("Could not determine scrape location based on provided arguments.")
         sys.exit(1)
    # --- End location determination ---

    jobs_df = scrape_jobs_with_jobspy(
        search_terms=args.search,
        location=scrape_location,         # Use the determined scrape_location
        sites=scraper_sites,
        results_wanted=args.results,
        hours_old=args.hours_old,
        country_indeed=args.country_indeed,
        proxies=proxy_list,
        offset=args.offset
    )

    if jobs_df is None or jobs_df.empty:
        log.warning("Scraping yielded no results. Pipeline cannot continue.")
        # Ensure analysis output file is created even if empty
        analysis_output_dir = os.path.dirname(args.analysis_output)
        if analysis_output_dir: os.makedirs(analysis_output_dir, exist_ok=True)
        with open(args.analysis_output, 'w', encoding='utf-8') as f: json.dump([], f)
        log.info(f"Empty analysis results file created at {args.analysis_output}")
        sys.exit(0)

    # --- Step 2: Convert and Save Scraped Data ---
    jobs_list = convert_and_save_scraped(jobs_df, args.scraped_jobs_file)
    if not jobs_list:
         log.error("Failed to convert or save scraped data. Pipeline halted.")
         sys.exit(1)

    # --- Step 3: Initialize Analyzer and Load Resume ---
    try:
        analyzer = ResumeAnalyzer()
    except Exception as e:
        log.critical(f"Failed to initialize ResumeAnalyzer: {e}.", exc_info=True)
        sys.exit(1)

    structured_resume = load_and_extract_resume(args.resume, analyzer)
    if not structured_resume:
        log.critical("Failed to load and extract data from resume.", exc_info=True)
        sys.exit(1)

    # --- Step 4: Analyze Jobs ---
    analyzed_results = analyze_jobs(analyzer, structured_resume, jobs_list)
    if not analyzed_results:
         log.warning("Analysis step produced no results.")

    # --- Step 5: Apply Filters, Sort, and Save ---
    filter_args_dict = {}
    # Standard filters
    if args.min_salary is not None: filter_args_dict['salary_min'] = args.min_salary
    if args.max_salary is not None: filter_args_dict['salary_max'] = args.max_salary
    if args.filter_work_models: filter_args_dict['work_models'] = [wm.strip().lower() for wm in args.filter_work_models.split(',')]
    if args.filter_job_types: filter_args_dict['job_types'] = [jt.strip().lower() for jt in args.filter_job_types.split(',')]

    # Advanced Location Filters (Passed to filter function for post-processing)
    if args.filter_remote_country: filter_args_dict['filter_remote_country'] = args.filter_remote_country.strip()
    if args.filter_proximity_location:
        if args.filter_proximity_range is None:
             # This validation is now done earlier, but double-check doesn't hurt
             parser.error("--filter-proximity-range is required when using --filter-proximity-location.")
        filter_args_dict['filter_proximity_location'] = args.filter_proximity_location.strip()
        filter_args_dict['filter_proximity_range'] = args.filter_proximity_range
        filter_args_dict['filter_proximity_models'] = [pm.strip().lower() for pm in args.filter_proximity_models.split(',')]
    # No need for elif here, logic handled during arg parsing and scrape location setting

    final_results_list_dict = apply_filters_sort_and_save(
        analyzed_results,
        args.analysis_output,
        filter_args_dict
    )

    # --- Step 6: Print Summary Table ---
    log.info("[bold blue]Pipeline Summary:[/bold blue]")
    print_summary_table(final_results_list_dict, top_n=10)

    log.info(f"[bold green]Pipeline Run Finished[/bold green]")

if __name__ == "__main__":
    run_pipeline()