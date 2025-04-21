# run_pipeline.py
import argparse
import logging
import json
import os
import sys
import hashlib # For caching
import traceback # For detailed error logging
import asyncio # For running async functions
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path # Use pathlib for paths

# Use the jobspy library for scraping
try:
    import jobspy
    from jobspy import scrape_jobs
except ImportError:
    print("CRITICAL ERROR: 'jobspy' library not found. Please install it via requirements.txt")
    sys.exit(1)

# Import analysis components from main_matcher and analyzer
try:
    from main_matcher import load_and_extract_resume, analyze_jobs, apply_filters_sort_and_save
    from analysis.analyzer import ResumeAnalyzer
    from analysis.models import ResumeData # Import ResumeData for caching type hint
except ImportError as e:
    # This error is critical, exit before full logging setup
    print(f"CRITICAL ERROR: Could not import analysis functions from main_matcher or analyzer: {e}")
    print("Ensure main_matcher.py and analysis/analyzer.py are in the correct path and define the expected classes/functions.")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR during core imports: {e}")
    traceback.print_exc()
    sys.exit(1)

# Load configuration AFTER basic imports succeed
try:
    import config # Central configuration (loads CFG)
except Exception as e:
     print(f"CRITICAL ERROR: Failed to import configuration module 'config.py': {e}")
     traceback.print_exc()
     sys.exit(1)


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
# Get logger instance first - basicConfig below reconfigures the root logger
log = logging.getLogger(__name__)

# Access logging settings AFTER CFG is potentially loaded in config.py
log_level_str = config.CFG.get('logging', {}).get('level', 'INFO').upper()
log_format_str = "%(message)s" if RICH_AVAILABLE else '%(asctime)s - %(levelname)s - %(message)s'
log_date_fmt_str = "[%X]" if RICH_AVAILABLE else '%Y-%m-%d %H:%M:%S'

# Apply logging configuration
if RICH_AVAILABLE:
    logging.basicConfig(
        level=log_level_str, # Use level from CFG
        format=log_format_str,
        datefmt=log_date_fmt_str,
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True # Force reconfiguration if basicConfig was called implicitly before
    )
    console = Console()
else:
     logging.basicConfig(
        level=log_level_str, # Use level from CFG
        format=log_format_str,
        datefmt=log_date_fmt_str,
        force=True
    )

# Suppress noisy libraries AFTER main config is applied
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
# Suppress geopy debug unless verbose mode is explicitly set later
if log.getEffectiveLevel() > logging.DEBUG:
     logging.getLogger("geopy").setLevel(logging.INFO)


# --- scrape_jobs_with_jobspy function ---
def scrape_jobs_with_jobspy(
    search_terms: str, location: str, sites: list[str], results_wanted: int, hours_old: int,
    country_indeed: str, proxies: Optional[list[str]] = None, offset: int = 0
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
            verbose=1, description_format="markdown", linkedin_fetch_description=True ) # Use markdown for cleaner text generally
        if jobs_df is None or jobs_df.empty: log.warning("Jobspy scraping returned no results or failed."); return None
        else:
            log.info(f"Jobspy scraping successful. Found {len(jobs_df)} jobs.")
            log.debug(f"DataFrame columns: {jobs_df.columns.tolist()}")
            required_cols = ['title', 'company', 'location', 'description', 'job_url']
            missing_cols = [col for col in required_cols if col not in jobs_df.columns]
            if missing_cols: log.warning(f"Jobspy output DataFrame is missing expected columns: {missing_cols}")
            return jobs_df
    except ImportError as ie: log.critical(f"Import error during scraping func call: {ie}."); return None
    except Exception as e: log.error(f"Error during jobspy scraping execution: {e}", exc_info=True); return None


# --- convert_and_save_scraped function ---
def convert_and_save_scraped(jobs_df: pd.DataFrame, output_path: Path) -> List[Dict[str, Any]]:
    """Converts DataFrame to list of dicts and saves to JSON, handling date objects."""
    log.info(f"Converting DataFrame to list and saving to {output_path}")
    try:
        rename_map = {'job_url': 'url','job_type': 'employment_type','salary': 'salary_text','benefits': 'benefits_text'}
        actual_rename_map = {k: v for k, v in rename_map.items() if k in jobs_df.columns}
        if actual_rename_map: jobs_df = jobs_df.rename(columns=actual_rename_map); log.debug(f"Renamed DataFrame columns: {actual_rename_map}")
        possible_date_columns = ['date_posted', 'posted_date', 'date']
        for col in possible_date_columns:
            if col in jobs_df.columns:
                log.debug(f"Processing potential date column '{col}' for JSON serialization.")
                if pd.api.types.is_datetime64_any_dtype(jobs_df[col]) or jobs_df[col].dtype == 'object':
                    try: jobs_df[col] = pd.to_datetime(jobs_df[col], errors='coerce'); jobs_df[col] = jobs_df[col].dt.strftime('%Y-%m-%d'); log.debug(f"Conv col '{col}' dates to str.")
                    except Exception as date_err: log.warning(f"Could not process date col '{col}'. Err: {date_err}. Trying astype(str)."); jobs_df[col] = jobs_df[col].astype(str)
                else: log.debug(f"Col '{col}' not datetime/obj ({jobs_df[col].dtype}), skipping date conv.")
        essential_cols = ['title','company','location','description','url','salary_text','employment_type','benefits_text','skills','date_posted']
        for col in essential_cols:
            if col not in jobs_df.columns: log.warning(f"Col '{col}' missing, adding empty."); jobs_df[col] = ''
        jobs_df = jobs_df.fillna(''); log.debug("Filled NA/NaN values.")
        jobs_list = jobs_df.to_dict('records'); log.debug(f"Converted DataFrame to list of {len(jobs_list)} dicts.")
        output_dir = output_path.parent;
        if output_dir: output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(jobs_list, f, indent=4)
        log.info(f"Saved {len(jobs_list)} scraped jobs to {output_path}"); return jobs_list
    except TypeError as json_err: log.error(f"JSON Serialization Error: {json_err}", exc_info=True); return []
    except (IOError, OSError) as file_err: log.error(f"File error saving scraped jobs {output_path}: {file_err}", exc_info=True); return []
    except Exception as e: log.error(f"Unexpected error converting/saving scraped jobs: {e}", exc_info=True); return []


# --- print_summary_table function ---
def print_summary_table(results_json: Optional[List[Dict[str, Any]]], top_n: int = 10):
    """Prints a summary table of top results using Rich."""
    if not results_json:
        if RICH_AVAILABLE: console.print("[yellow]No analysis results to summarize.[/yellow]")
        else: log.warning("No analysis results to summarize.")
        return
    if not RICH_AVAILABLE:
         log.info("Top results (rich table unavailable):")
         for i, result in enumerate(results_json[:top_n]):
             analysis = result.get('analysis', {}); original = result.get('original_job_data', {})
             score = analysis.get('suitability_score', -1)
             if score is None or score == 0: continue
             log.info(f"  {i+1}. Score: {score}% | Title: {original.get('title', 'N/A')} | Comp: {original.get('company', 'N/A')} | URL: {original.get('url', '#')}")
         return
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
    if count == 0: console.print("[yellow]No successfully analyzed jobs with score > 0 to display.[/yellow]")
    else: console.print(table)

# --- Resume Caching Helper ---
def get_resume_hash(filepath: str) -> Optional[str]:
    """Calculates SHA256 hash of a file's content."""
    try:
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as file:
            while True:
                chunk = file.read(4096);
                if not chunk: break
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError: log.error(f"Resume file not found at {filepath} for hashing."); return None
    except Exception as e: log.error(f"Error hashing resume file {filepath}: {e}", exc_info=True); return None

# --- Main Execution Logic (Async Wrapper) ---
async def async_pipeline_logic(args):
    """The core logic wrapped in an async function."""
    config.ensure_required_dirs()

    # --- Determine scrape location (uses args directly) ---
    scrape_location = None
    if args.filter_remote_country: scrape_location = args.filter_remote_country.strip(); log.info(f"Using country '{scrape_location}' as primary scrape location (from --filter-remote-country).")
    elif args.filter_proximity_location: scrape_location = args.filter_proximity_location.strip(); log.info(f"Using proximity target '{scrape_location}' as primary scrape location.")
    elif args.location: scrape_location = args.location; log.info(f"Using provided --location '{scrape_location}' as primary scrape location.")

    # --- Step 1: Scrape Jobs ---
    log.info("--- Stage 1: Scraping Jobs ---")
    scraper_sites = [site.strip().lower() for site in args.sites.split(',')]
    proxy_list = [p.strip() for p in args.proxies.split(',')] if args.proxies else None
    # Scraping is often blocking I/O, run in executor if it becomes a bottleneck
    # loop = asyncio.get_running_loop()
    # jobs_df = await loop.run_in_executor(None, scrape_jobs_with_jobspy, ...) # Example
    jobs_df = scrape_jobs_with_jobspy(
        search_terms=args.search, location=scrape_location, sites=scraper_sites, results_wanted=args.results,
        hours_old=args.hours_old, country_indeed=args.country_indeed, proxies=proxy_list, offset=args.offset )

    if jobs_df is None or jobs_df.empty:
        log.warning("Scraping yielded no results. Pipeline finished.")
        analysis_output_path = Path(args.analysis_output) # Use Path object
        analysis_output_dir = analysis_output_path.parent;
        if analysis_output_dir: analysis_output_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_output_path, 'w', encoding='utf-8') as f: json.dump([], f)
        log.info(f"Empty analysis results file created at {analysis_output_path}"); return # Exit async function

    # --- Step 2: Convert and Save Scraped Data ---
    log.info("--- Stage 2: Processing Scraped Data ---")
    jobs_list = convert_and_save_scraped(jobs_df, Path(args.scraped_jobs_file))
    if not jobs_list: raise RuntimeError("Failed to convert or save scraped data.") # Raise to be caught by main try/except

    # --- Step 3: Initialize Analyzer and Load/Cache Resume ---
    log.info("--- Stage 3: Initializing Analyzer and Processing Resume ---")
    analyzer: Optional[ResumeAnalyzer] = None
    structured_resume: Optional[ResumeData] = None
    try:
        analyzer = ResumeAnalyzer()

        # --- Resume Caching Logic ---
        resume_hash = get_resume_hash(args.resume)
        cache_filepath = None
        if resume_hash:
             cache_filename = f"{resume_hash}.json"
             # Ensure cache dir comes from loaded config path object
             cache_filepath = config.RESUME_CACHE_DIR / cache_filename
             log.debug(f"Resume hash: {resume_hash}, Cache file path: {cache_filepath}")

        if not args.force_resume_reparse and cache_filepath and cache_filepath.exists():
            log.info(f"Loading structured resume data from cache: {cache_filepath}")
            try:
                with open(cache_filepath, 'r', encoding='utf-8') as f: cached_data_json = f.read()
                structured_resume = ResumeData.model_validate_json(cached_data_json)
                log.info("Successfully loaded resume data from cache.")
            except Exception as e:
                log.warning(f"Failed to load resume from cache {cache_filepath}: {e}. Re-parsing.", exc_info=True)
                structured_resume = None # Force re-parsing
        else:
             if not args.force_resume_reparse and cache_filepath: log.info("Resume cache not found. Parsing resume...")
             elif args.force_resume_reparse: log.info("--force-resume-reparse used. Parsing resume...")
             structured_resume = None # Ensure it's None if not loaded

        if not structured_resume: # If cache load failed or skipped
            structured_resume = await load_and_extract_resume(args.resume, analyzer) # Await the async call
            if structured_resume and cache_filepath: # Save to cache if parsing succeeded
                try:
                    with open(cache_filepath, 'w', encoding='utf-8') as f: f.write(structured_resume.model_dump_json(indent=2))
                    log.info(f"Saved structured resume data to cache: {cache_filepath}")
                except Exception as cache_e: log.warning(f"Failed to save resume data to cache {cache_filepath}: {cache_e}", exc_info=True)
        # --- End Resume Caching Logic ---

        if not structured_resume: raise RuntimeError("Failed to load and extract structured data from resume.")

        # --- Step 4: Analyze Jobs ---
        log.info("--- Stage 4: Analyzing Job Suitability (Async) ---")
        analyzed_results = await analyze_jobs(analyzer, structured_resume, jobs_list) # Await async call
        if not analyzed_results: log.warning("Analysis step produced no results.")

        # --- Step 5: Apply Filters, Sort, and Save ---
        log.info("--- Stage 5: Filtering, Sorting, and Saving Results ---")
        filter_args_dict = {}
        # Populate filter dict including NEW filters
        if args.min_salary is not None: filter_args_dict['salary_min'] = args.min_salary
        if args.max_salary is not None: filter_args_dict['salary_max'] = args.max_salary
        if args.filter_work_models: filter_args_dict['work_models'] = [wm.strip().lower() for wm in args.filter_work_models.split(',')]
        if args.filter_job_types: filter_args_dict['job_types'] = [jt.strip().lower() for jt in args.filter_job_types.split(',')]
        if args.filter_companies: filter_args_dict['filter_companies'] = [c.strip().lower() for c in args.filter_companies.split(',')] # Normalize case
        if args.exclude_companies: filter_args_dict['exclude_companies'] = [c.strip().lower() for c in args.exclude_companies.split(',')] # Normalize case
        if args.filter_title_keywords: filter_args_dict['filter_title_keywords'] = [k.strip().lower() for k in args.filter_title_keywords.split(',')] # Normalize case
        if args.filter_date_after: filter_args_dict['filter_date_after'] = args.filter_date_after
        if args.filter_date_before: filter_args_dict['filter_date_before'] = args.filter_date_before
        # Add min_score directly to filter_args_dict
        filter_args_dict['min_score'] = args.min_score
        # Advanced Location Filters
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
        # Ensure async client is closed if analyzer was initialized
        if analyzer and hasattr(analyzer, 'close'):
             await analyzer.close()


# --- Main Execution Entry Point ---
def main():
    """Parses args and runs the async pipeline logic."""
    parser = argparse.ArgumentParser(
        description="Run Job Scraping (via JobSpy) and GenAI Analysis Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    # --- Argument Groups ---
    scrape_group = parser.add_argument_group('Scraping Options (JobSpy)')
    scrape_group.add_argument("--search", required=True, help="Job title, keywords, or company.")
    scrape_group.add_argument("--location", default=None, help="Primary location for scraping. Overridden if --filter-remote-country.")
    # Use defaults from loaded CFG now
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
    filter_group.add_argument("--filter-companies", help="Include ONLY these companies (comma-sep).")
    filter_group.add_argument("--exclude-companies", help="EXCLUDE these companies (comma-sep).")
    filter_group.add_argument("--filter-title-keywords", help="Require ANY of these keywords in title (comma-sep).")
    filter_group.add_argument("--filter-date-after", help="Include jobs posted ON or AFTER YYYY-MM-DD.")
    filter_group.add_argument("--filter-date-before", help="Include jobs posted ON or BEFORE YYYY-MM-DD.")
    filter_group.add_argument("--min-score", type=int, default=0, help="Minimum suitability score filter (0-100). Default 0.")

    adv_loc_group = parser.add_argument_group('Advanced Location Filtering')
    adv_loc_group.add_argument("--filter-remote-country", help="Filter REMOTE jobs within specific country.")
    adv_loc_group.add_argument("--filter-proximity-location", help="Reference location for proximity filtering.")
    adv_loc_group.add_argument("--filter-proximity-range", type=float, help="Distance in miles for proximity.")
    adv_loc_group.add_argument("--filter-proximity-models", default="Hybrid,On-site", help="Work models for proximity.")

    args = parser.parse_args()

    # --- Setup Logging Level Based on Args ---
    log_level_arg = logging.DEBUG if args.verbose else config.CFG.get('logging', {}).get('level', 'INFO').upper()
    try:
        # Ensure the level name is valid before setting
        logging.getLogger().setLevel(log_level_arg)
        log.info(f"Log level set to: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")
    except ValueError:
         log.error(f"Invalid log level configured: {log_level_arg}. Using INFO.")
         logging.getLogger().setLevel(logging.INFO)


    # --- Run the main async logic ---
    try:
        asyncio.run(async_pipeline_logic(args))
    except KeyboardInterrupt:
        print(); log.warning("[yellow]Pipeline execution interrupted by user (Ctrl+C).[/yellow]" if RICH_AVAILABLE else "Pipeline interrupted by user (Ctrl+C).")
        sys.exit(130) # Standard exit code for Ctrl+C
    except RuntimeError as e:
         # Catch specific errors raised by the pipeline logic
         log.critical(f"Pipeline failed with runtime error: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
         # Catch-all for truly unexpected errors during the main flow
         log.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
         sys.exit(1)

if __name__ == "__main__":
    main()