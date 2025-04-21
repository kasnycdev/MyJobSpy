import argparse
import logging
import json
import os
import sys
import asyncio # Import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# Use the jobspy library for scraping
try:
    import jobspy
    from jobspy import scrape_jobs
except ImportError: print("CRITICAL ERROR: 'jobspy' library not found."); sys.exit(1)

# Import analysis components
try:
    from main_matcher import load_and_extract_resume_async, analyze_jobs_async, apply_filters_sort_and_save # Use async versions
    from analysis.analyzer import ResumeAnalyzer
except ImportError as e: print(f"CRITICAL ERROR: Could not import analysis functions: {e}"); sys.exit(1)

import config

# Rich for UX
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Setup logging using Rich
logging.basicConfig( level=config.LOG_LEVEL, format=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT, handlers=[RichHandler(rich_tracebacks=True, show_path=False)] )
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
console = Console()


# --- scrape_jobs_with_jobspy function remains synchronous ---
def scrape_jobs_with_jobspy(
    search_terms: str, location: str, sites: list[str], results_wanted: int, hours_old: int,
    country_indeed: str, proxies: Optional[list[str]] = None, offset: int = 0
    ) -> Optional[pd.DataFrame]:
    # (Function content remains the same as previous version)
    log.info(f"[bold blue]Starting job scraping via JobSpy...[/bold blue]")
    log.info(f"Search: '[cyan]{search_terms}[/cyan]' | Location: '[cyan]{location}[/cyan]' | Sites: {sites}")
    log.info(f"Params: Results ≈{results_wanted}, Max Age={hours_old}h, Indeed Country='{country_indeed}', Offset={offset}")
    if proxies: log.info(f"Using {len(proxies)} proxies.")
    try:
        jobs_df = scrape_jobs( site_name=sites, search_term=search_terms, location=location, results_wanted=results_wanted, hours_old=hours_old, country_indeed=country_indeed, proxies=proxies, offset=offset, verbose=1, description_format="markdown", linkedin_fetch_description=True)
        if jobs_df is None or jobs_df.empty: log.warning("Jobspy scraping returned no results or failed."); return None
        else:
            log.info(f"Jobspy scraping successful. Found {len(jobs_df)} jobs.")
            log.debug(f"DataFrame columns: {jobs_df.columns.tolist()}")
            # Ensure essential columns exist for later processing, prevents KeyErrors
            essential_scrape_cols = ['title', 'company', 'location', 'description', 'job_url', 'date_posted', 'job_type']
            for col in essential_scrape_cols:
                 if col not in jobs_df.columns:
                     log.warning(f"Essential column '{col}' missing from JobSpy output, adding as empty.")
                     jobs_df[col] = ''
            return jobs_df
    except ImportError as ie: log.critical(f"Import error during scraping: {ie}."); return None
    except Exception as e: log.error(f"An error occurred during jobspy scraping: {e}", exc_info=True); return None


# --- convert_and_save_scraped function remains synchronous ---
def convert_and_save_scraped(jobs_df: pd.DataFrame, output_path: str) -> List[Dict[str, Any]]:
    # (Function content remains the same as previous version)
    log.info(f"Converting DataFrame to list and saving to {output_path}")
    rename_map = {'job_url': 'url','job_type': 'employment_type','salary': 'salary_text','benefits': 'benefits_text'}
    actual_rename_map = {k: v for k, v in rename_map.items() if k in jobs_df.columns}; jobs_df = jobs_df.rename(columns=actual_rename_map)
    possible_date_columns = ['date_posted', 'posted_date', 'date']
    for col in possible_date_columns:
        if col in jobs_df.columns:
            if pd.api.types.is_datetime64_any_dtype(jobs_df[col]) or jobs_df[col].dtype == 'object':
                 try: jobs_df[col] = pd.to_datetime(jobs_df[col], errors='coerce'); jobs_df[col] = jobs_df[col].dt.strftime('%Y-%m-%d')
                 except Exception: jobs_df[col] = jobs_df[col].astype(str)
    essential_cols = ['title','company','location','description','url','salary_text','employment_type','benefits_text','skills','date_posted']
    for col in essential_cols:
         if col not in jobs_df.columns: log.warning(f"Column '{col}' missing, adding empty."); jobs_df[col] = ''
    jobs_df = jobs_df.fillna(''); jobs_list = jobs_df.to_dict('records')
    output_dir = os.path.dirname(output_path);
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(jobs_list, f, indent=4)
        log.info(f"Saved {len(jobs_list)} scraped jobs to {output_path}"); return jobs_list
    except Exception as e: log.error(f"Error saving scraped jobs JSON: {e}", exc_info=True); return []


# --- print_summary_table function remains synchronous ---
def print_summary_table(results_json: List[Dict[str, Any]], top_n: int = 10):
     # (Function content remains the same as previous version)
     if not results_json: console.print("[yellow]No analysis results to summarize.[/yellow]"); return
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


# --- Main ASYNC Execution Function ---
async def run_pipeline_async(): # Changed to async def
    # (Argument Parsing remains the same)
    parser = argparse.ArgumentParser( description="Run Job Scraping (JobSpy) & GenAI Analysis Pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter )
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

    # Setup Logging Level
    log_level = logging.DEBUG if args.verbose else config.LOG_LEVEL
    logging.getLogger().setLevel(log_level)
    log.info(f"[bold green]Starting ASYNC Pipeline Run[/bold green] ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    # Main Pipeline Logic Wrapped in try...except KeyboardInterrupt
    try:
        config.ensure_output_dir()
        # Validate argument combinations
        if not args.location and not args.filter_remote_country and not args.filter_proximity_location: parser.error("Ambiguous location: Specify --location OR --filter-remote-country OR --filter-proximity-location.")
        if args.filter_proximity_location and args.filter_remote_country: parser.error("Conflicting filters: Cannot use --filter-proximity-location and --filter-remote-country.")
        if args.filter_proximity_location and args.filter_proximity_range is None: parser.error("--filter-proximity-range required with --filter-proximity-location.")
        if args.filter_proximity_range is not None and not args.filter_proximity_location: parser.error("--filter-proximity-location required with --filter-proximity-range.")

        # Determine scrape location
        scrape_location = None
        if args.filter_remote_country: scrape_location = args.filter_remote_country.strip(); log.info(f"Using country '{scrape_location}' as primary scrape location.")
        elif args.filter_proximity_location: scrape_location = args.filter_proximity_location.strip(); log.info(f"Using proximity target '{scrape_location}' as primary scrape location.")
        elif args.location: scrape_location = args.location; log.info(f"Using provided --location '{scrape_location}' as primary scrape location.")

        # --- Step 1: Scrape Jobs (Synchronous) ---
        scraper_sites = [site.strip().lower() for site in args.sites.split(',')]
        proxy_list = [p.strip() for p in args.proxies.split(',')] if args.proxies else None
        jobs_df = scrape_jobs_with_jobspy( search_terms=args.search, location=scrape_location, sites=scraper_sites, results_wanted=args.results, hours_old=args.hours_old, country_indeed=args.country_indeed, proxies=proxy_list, offset=args.offset )
        if jobs_df is None or jobs_df.empty:
            log.warning("Scraping yielded no results. Exiting."); sys.exit(0) # Exit cleanly

        # --- Step 2: Convert and Save (Synchronous) ---
        jobs_list = convert_and_save_scraped(jobs_df, args.scraped_jobs_file)
        if not jobs_list: log.error("Failed to convert/save scraped data. Exiting."); sys.exit(1)

        # --- Step 3: Initialize Analyzer and Load Resume (Load Resume is ASYNC) ---
        try: analyzer = ResumeAnalyzer()
        except Exception as e: log.critical(f"Failed to initialize ResumeAnalyzer: {e}.", exc_info=True); sys.exit(1)

        structured_resume = await load_and_extract_resume_async(args.resume, analyzer) # Await async call
        if not structured_resume: log.critical("Failed to load/extract resume data. Exiting."); sys.exit(1)

        # --- Step 4: Analyze Jobs (ASYNC) ---
        analyzed_results = await analyze_jobs_async(analyzer, structured_resume, jobs_list) # Await async call
        if not analyzed_results: log.warning("Analysis step produced no results.")

        # --- Step 5: Apply Filters, Sort, and Save (Synchronous) ---
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

        final_results_list_dict = apply_filters_sort_and_save( analyzed_results, args.analysis_output, filter_args_dict )

        # --- Step 6: Print Summary Table ---
        log.info("[bold blue]Pipeline Summary:[/bold blue]")
        print_summary_table(final_results_list_dict, top_n=10)
        log.info(f"[bold green]Pipeline Run Finished Successfully[/bold green]")

    except KeyboardInterrupt: print(); log.warning("[yellow]Pipeline interrupted by user (Ctrl+C).[/yellow]"); sys.exit(130)
    except Exception as e: log.critical(f"Unexpected critical error during pipeline execution: {e}", exc_info=True); sys.exit(1)

# --- Update entry point to run the async function ---
if __name__ == "__main__":
    # Use asyncio.run() to start the async event loop and run the pipeline
    try:
        asyncio.run(run_pipeline_async())
    except KeyboardInterrupt:
        # Catch Ctrl+C at the top level if it wasn't caught inside run_pipeline_async
        print("\nExecution cancelled by user.")
        sys.exit(130)